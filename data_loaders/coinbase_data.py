from dotenv import load_dotenv
load_dotenv()
import os
import requests
from requests.exceptions import HTTPError
import pandas as pd
import time

# default inputs
data_frequencies = ['UNKNOWN_GRANULARITY', 
                    'ONE_MINUTE', 
                    'FIVE_MINUTE', 
                    'FIFTEEN_MINUTE', 
                    'THIRTY_MINUTE', 
                    'ONE_HOUR', 
                    'TWO_HOUR', 
                    'FOUR_HOUR', 
                    'SIX_HOUR', 
                    'ONE_DAY']

candle_limit = 350

request_limit = 10000

# seconds per candle, keyed by granularity
granularity_seconds = {
    'ONE_MINUTE': 60,
    'FIVE_MINUTE': 300,
    'FIFTEEN_MINUTE': 900,
    'THIRTY_MINUTE': 1800,
    'ONE_HOUR': 3600,
    'TWO_HOUR': 7200,
    'FOUR_HOUR': 14400,
    'SIX_HOUR': 21600,
    'ONE_DAY': 86400,
}


def process_requests(request_path: str,
                     requests_input: dict[str],
                     payloads: list):
    """"""
    resp = requests.get(request_path, params = requests_input)
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

    if granularity not in data_frequencies:
        raise ValueError(f'{granularity} not valid frequency, must be one of {data_frequencies}')
    if not (0 < request_size <= candle_limit):
        raise ValueError(f'request_size must be in (0, {candle_limit}]')

    # timeframe in max-request seconds
    step = granularity_seconds[granularity] * request_size

    # candles API uses unix seconds, not milliseconds
    start_epoch = int(pd.Timestamp(start_date).timestamp())
    end_epoch = int(pd.Timestamp(end_date).timestamp())

    # payloads
    dict_payloads: list[dict[str]] = []

    # inputs
    data_path = f'https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles'
    params = {
        'start': None,
        'end': None,
        'granularity': granularity,
        'limit': request_size,
    }

    # walk the range oldest -> newest, one window (<= request_size bars) per request
    window_start = start_epoch
    while window_start < end_epoch:
        window_end = min(window_start + step, end_epoch)
        params['start'] = window_start
        params['end'] = window_end

        retries = 0
        while True:
            try:
                process_requests(data_path, params, dict_payloads)
                break # successfull pull, break loop
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429 and retries < max_retries:
                    retries += 1
                    print(f'rate limited, retrying ({retries}/{max_retries}) after {request_delay}s')
                    time.sleep(request_delay)
                    continue
                raise

        window_start = window_end
        time.sleep(request_delay)

    # filter data
    data = pd.DataFrame(dict_payloads)
    data['start'] = pd.to_datetime(data['start'].astype(int), unit = 's', utc = True)
    data.set_index('start', inplace = True)
    data = data[~data.index.duplicated(keep = 'first')]
    data.sort_index(inplace = True) # guard against any residual out-of-order rows across window boundaries

    # reindex onto the full expected grid so every symbol returns the same number of rows
    expected_index = pd.date_range(start = pd.Timestamp(start_epoch, unit = 's', tz = 'UTC'),
                                    end = pd.Timestamp(end_epoch, unit = 's', tz = 'UTC'),
                                    freq = pd.Timedelta(seconds = granularity_seconds[granularity]))
    if data.index.min() > expected_index[0]:
        print(f"warning: {symbol} has no data before {data.index.min()} "
              f"(likely not listed/traded yet at {expected_index[0]}); ")
    data = data.reindex(expected_index)
    data = data.ffill() # fill genuine no-trade gaps with the last best know price

    # save data
    if data_download_path is not None:
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
               'USDT-GBP',  
               'XRP-GBP']
    start = '2023-01-01'
    end = '2026-01-01'

    # tickers
    file_type = '.csv'
    for s in symbols: 
        file_name = f'{s}_{start}_{end}' + file_type
        get_coinbase_candles(data_download_path = path, 
                             file_name = file_name, 
                             symbol = s, 
                             start_date = start, 
                             end_date = end, 
                             granularity = 'FIVE_MINUTE')
        print(f"successfully downloaded {s} data to '{path}'")