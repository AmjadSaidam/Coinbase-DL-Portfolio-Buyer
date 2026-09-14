from dotenv import load_dotenv
load_dotenv()
import os
import requests
from requests.exceptions import RequestException
import pandas as pd
import time
# .py
import data_loaders.utils as utils

api_root = 'https://api.coinbase.com/api/v3/brokerage/market'
candle_limit = 350
request_limit = 10000
request_timeout = 30 # seconds; requests has no default timeout, a stalled connection would block forever
leading_gap_limit = pd.Timedelta(days = 1) # longest leading no-trade gap treated as thin liquidity rather than an unlisted product

# filter data
def data_standerdise(data_payload: list[dict] | dict[list],
                     symbol: str,
                     start_epoch: int,
                     end_epoch: int,
                     granularity: str):
    """forms dataframe with time index, cleans data and aligns index to time range"""
    data = pd.DataFrame(data_payload)
    if data.empty: # no candles at all in this window (e.g. asset not listed/traded yet) - fail loud instead of
        # a cryptic KeyError on 'start' below, or silently reindexing to an all-NaN column that would
        # poison every downstream tensor built from this symbol (ffill/bfill cannot recover with zero rows)
        raise ValueError(f"{symbol} returned no candles for the requested window "
                         f"{pd.Timestamp(start_epoch, unit = 's', tz = 'UTC')} - {pd.Timestamp(end_epoch, unit = 's', tz = 'UTC')} "
                         f"(likely not listed/traded yet for any part of this range)")
    data['start'] = pd.to_datetime(data['start'].astype(int), unit = 's', utc = True)
    data.set_index('start', inplace = True)
    data = data.astype(float) # OHLCV fields come back as strings from the API; CSV round-trips infer this implicitly, live callers do not
    data = data[~data.index.duplicated(keep = 'first')]
    data.sort_index(inplace = True) # guard against any residual out-of-order rows across window boundaries

    # reindex onto the full expected grid so every symbol returns the same number of rows
    expected_index = pd.date_range(start = pd.Timestamp(start_epoch, unit = 's', tz = 'UTC'),
                                    end = pd.Timestamp(end_epoch, unit = 's', tz = 'UTC'),
                                    freq = pd.Timedelta(seconds = utils.granularity_seconds[granularity]))
    first_candle = data.index.min()
    leading_gap = first_candle - expected_index[0] # coinbase omits buckets with no trades, so a thin pair can start late
    data = data.reindex(expected_index)
    data = data.ffill() # fill genuine no-trade gaps with the last best known price
    if leading_gap <= leading_gap_limit:
        data = data.bfill() # a short leading gap is just no-trade bars at the range start; ffill has no prior price, so take the first trade's
    else:
        # the product was not listed/tradeable for the leading part of the range: leave it NaN so
        # coinbase_price_return_data() aligns the whole universe on the first date every symbol trades
        print(f"warning: {symbol} has no candles for the first {leading_gap} of the requested range "
              f"(requested start {expected_index[0]}, first trade {first_candle}) - "
              f"not listed/tradeable yet, leading rows left NaN for universe alignment")

    return data

def coinbase_products(product_type: str = 'SPOT') -> set[str]:
    """product ids currently listed on the exchange (public endpoint, no auth)"""
    resp = requests.get(f'{api_root}/products', params = {'product_type': product_type}, timeout = request_timeout)
    resp.raise_for_status()
    return {product['product_id'] for product in resp.json().get('products', [])}

def process_requests(request_path: str,
                     requests_input: dict[str],
                     payloads: list):
    """single candles GET, appends the window's rows (chronological) to payloads"""
    resp = requests.get(request_path, params = requests_input, timeout = request_timeout)
    resp.raise_for_status()
    batch = resp.json()
    # coinbase returns candles newest -> oldest per window; reverse to chronological order
    rows = batch.get('candles', [])[::-1]
    payloads.extend(rows)

    return rows


def get_coinbase_candles(data_download_path: str,
                         file_name: str,
                         symbol: str,
                         start_date: pd.Timestamp,
                         end_date: pd.Timestamp,
                         granularity: str,
                         request_size: int = candle_limit,
                         request_delay: float = 0.2,
                         max_retries: int = 5):
    """"""
    # download path
    file_path = os.path.join(data_download_path, file_name)
    if file_path is not None and os.path.exists(file_path):
        return None

    if granularity not in utils.data_frequencies:
        raise ValueError(f'{granularity} not valid frequency, must be one of {utils.data_frequencies}')
    if not (0 < request_size <= candle_limit):
        raise ValueError(f'request_size must be in (0, {candle_limit}]')

    # timeframe in max-request seconds
    bar = utils.granularity_seconds[granularity]
    step = bar * (request_size - 1) # start/end are both inclusive, so this spans exactly request_size bar opens;
                                    # one bar more and the API silently drops the oldest candle past its cap

    # candles API uses unix seconds, not milliseconds
    start_epoch = int(pd.Timestamp(start_date).timestamp())
    end_epoch = int(pd.Timestamp(end_date).timestamp())

    # payloads
    dict_payloads: list[dict[str]] = []

    # inputs
    data_path = f'{api_root}/products/{symbol}/candles'
    params = {
        'start': None,
        'end': None,
        'granularity': granularity,
        'limit': request_size,
    }

    # walk the range oldest -> newest, one window (<= request_size bars) per request
    window_start = start_epoch
    while window_start <= end_epoch:
        window_end = min(window_start + step, end_epoch)
        params['start'] = window_start
        params['end'] = window_end

        retries = 0
        while True:
            try:
                process_requests(data_path, params, dict_payloads)
                break # successfull pull, break loop
            except RequestException as e:
                status = e.response.status_code if e.response is not None else None # None for a timeout/connection error (no response received)
                if (status is None or status == 429 or status >= 500) and retries < max_retries:
                    retries += 1
                    backoff = request_delay * 2 ** retries
                    print(f'{symbol} {"rate limited" if status == 429 else f"request failed ({status})"}, retrying ({retries}/{max_retries}) after {backoff}s')
                    time.sleep(backoff)
                    continue
                detail = e.response.text if e.response is not None else str(e)
                raise RuntimeError(f'{symbol} candles request failed for window {window_start}-{window_end}: {detail}') from e

        window_start = window_end + bar # both ends are inclusive, so the next window starts on the following bar
        time.sleep(request_delay)

    data = data_standerdise(dict_payloads, symbol, start_epoch, end_epoch, granularity)

    # save data
    if data_download_path is not None:
        os.makedirs(data_download_path, exist_ok = True)
        data.to_csv(file_path)

    return None

# run
if __name__ == '__main__':
    # data path
    path = os.environ.get('WFA_DOWNLOAD_PATH')

    # defaults
    symbols = ['BTC-GBP',
               'ETH-GBP',
               'SOL-GBP',
               'LINK-GBP',
               'USDT-GBP'] # XRP-GBP is not a coinbase product (only XRP-EUR/USD/USDC/USDT are quoted)
    start = '2023-01-01'
    end = '2026-01-01'

    # validate up front: an unlisted pair is a hard 400 from the candles endpoint
    listed = coinbase_products()

    # tickers
    file_type = '.csv'
    for s in symbols:
        if s not in listed:
            print(f"skipping {s}: not a listed coinbase product")
            continue
        file_name = f'{s}_{start}_{end}' + file_type
        get_coinbase_candles(data_download_path = path,
                             file_name = file_name,
                             symbol = s,
                             start_date = start,
                             end_date = end,
                             granularity = 'FIVE_MINUTE')
        print(f"successfully downloaded {s} data to '{path}'")
