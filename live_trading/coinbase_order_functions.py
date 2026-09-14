"""
coinbase API order logic, via coinbase-advanced python SDK

Coinbase Advanced API, API Reference 
https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/introduction

Coinbase Advanced API Python SDK docs
https://github.com/coinbase/coinbase-advanced-py/blob/master/docs/coinbase.rest.rst

note difference between RESTClient and Websocket 
- RESTClient (Rest) = request (get/post) endpoints 
- WSClient (Web Socket) = request (get/post) live-market data endpoints
"""
from dotenv import load_dotenv; load_dotenv()
import time
from coinbase.rest import RESTClient # Install Coinbase Advanced API Python SDK
from requests.exceptions import RequestException
import uuid
import datetime
from collections import defaultdict
import pandas as pd
from decimal import Decimal
# .py
import live_trading.live_errors as coin_error
from data_loaders.coinbase_data import data_standerdise, candle_limit
from data_loaders.coinbase_data_post_process import coinbase_price_return_data
import data_loaders.utils as utils

class CoinbaseTrader:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = RESTClient(api_key = api_key, api_secret = api_secret, timeout = 30) # SDK default is no timeout - a stalled connection would block forever with no exception raised
        self.authenticated = False
        self.cash = 0
        self._unpriceable_bases = set() # bases with no valid {base}-GBP product (delisted/renamed/fiat/unsupported), cached to avoid repeat failed requests
    
    def login(self):
        """Authentication function"""
        account = self.client.get_accounts() # attempts to retreive account data to authorsie login
        if account is None:
            raise coin_error.CoinLoginError(f'failed to authenticate account cradentials with\napi_key: {self.api_key}\n api_secret: {self.api_secret}')
        self.authenticated = True
        return self.authenticated
    
    def tickers_weight(self, 
                       tickers: list, 
                       weights: list):
        """Creates Ticker-Weight Dict """
        order_message = dict(zip(tickers, weights))
        return order_message
    
    def get_base(self, asset: str):
        """given pair BASE:QUOTE returns BASE"""
        return asset.split("-")[0] 
    
    def asset_base(self, 
                   asset: str, 
                   account_values: dict) -> float: 
        """
        gets BASE currency investment for pair BASE:QUOTE 
        """
        asset = self.get_base(asset)
        return account_values[asset] # asset associted holding, e.g. BTC-USD -> BTC: x  

    def coinbase_data(self,
                      products: list[str],
                      num_bars: int = 1440,
                      granularity: str = 'FIVE_MINUTE',
                      request_delay: float = 0.2,
                      max_retries: int = 5):
        """gets close and return data from exchange, supports 1min"""
        # final dataframe
        dict_payloads = defaultdict(list) # dict of lists of product candle data dicts

        # time-range in seconds
        gran_secs = utils.granularity_seconds[granularity]
        step = gran_secs * (candle_limit - 1) # start/end are both inclusive: exactly candle_limit bar opens per request, one more and the API silently drops the oldest candle
        # time range, start, end time
        now_epoch = int(datetime.datetime.now().timestamp())
        end_epoch = now_epoch - (now_epoch % gran_secs) # floor to get current overlap and subtracked from live tiem to get latest bar open
        start_epoch = end_epoch - num_bars * gran_secs
        self.end_date = pd.Timestamp(end_epoch, unit = 's', tz = 'UTC')
        self.start_date = pd.Timestamp(start_epoch, unit = 's', tz = 'UTC')

        # pagnate over assets
        for product in products:
            window_start = start_epoch
            while window_start <= end_epoch:
                window_end = min(window_start + step, end_epoch) # right pointer

                retries = 0
                while True:
                    try:
                        candles = self.client.get_public_candles(
                            product_id = product,
                            start = window_start,
                            end = window_end,
                            granularity = granularity
                        )
                        break # successful pull, break retry loop
                    except RequestException as e:
                        status = e.response.status_code if e.response is not None else None # None for a timeout/connection error (no response received)
                        if (status is None or status in (429, 500, 502, 503, 504)) and retries < max_retries:
                            retries += 1
                            print(f'{product} rate limited/server error ({status}), retrying ({retries}/{max_retries}) after {request_delay}s')
                            time.sleep(request_delay)
                            continue
                        raise coin_error.CoinDataError(f'{product} failed data pull in range {window_start}-{window_end}: {e}') from e

                if candles is None:
                    raise coin_error.CoinDataError(f'{product} failed data pull in range {window_start}-{window_end}')
                # assumeing error not raised
                window_start = window_end + gran_secs # shift left pointer to the bar after this inclusive window
                # append payload, SDK returns typed Candle objects not raw dicts
                dict_payloads[product].extend([candle.to_dict() for candle in candles['candles']])
                # throttle
                time.sleep(request_delay)

        # dict to dataframe, clean and re-index to time-range
        data_payloads = {}
        for product in products:
            data_payloads[product] = data_standerdise(dict_payloads[product], product, start_epoch, end_epoch, granularity)
        
        # post process for model features 
        data_prices, data_returns = coinbase_price_return_data(data_universe = data_payloads)

        return {
            'prices': data_prices, 
            'returns': data_returns
        }

    def get_user_accounts(self) -> dict[float]:
        """get base value invested for each asset in portfolio"""
        accounts = self.client.get_accounts() # all authenticated user accounts
        account_values = {}
        for account in accounts['accounts']:
            value = account['available_balance']['value']
            account_values[account['currency']] = float(value)
            #print(f"{account['currency']} -> {account['uuid']} | value {value}")
        
        return account_values
        
    def get_asset_changes_price(self, assets: list):
        """price and 24hr price percentage change for each asset in portfolio """
        change_price = {
            "change": [], 
            "price": []
        }
        for asset in assets:
            try:
                product = self.client.get_product(product_id = asset) # product endpoint data
            except Exception as e:
                raise coin_error.CoinDataError(f'{asset} query error: {e}') from e
            percent_change = float((product.price_percentage_change_24h).split()[0]) / 100 # split to drop '%'
            price = float(product.price)
            change_price['change'].append(percent_change)
            change_price['price'].append(price)

        return change_price   

    def get_price(self, asset: str, suffix = 'GBP'):
        """ 
        gets asset price 

        asset: can be in BASE:QUOTE form or BASE form
        suffix: the QUOTE currency if only the BASE is provided
        """
        if len(asset.split('-')) != 2: # if base e.g. 'BTC' add the GBP so we can pull a valid listed pair
            asset = asset + '-' + suffix

        # attempt to pull asset data, otherwise return 0 
        try:
            asset = self.client.get_product(asset)
        except Exception as e:
            raise coin_error.CoinDataError(f'{asset} query error: {e}') from e
        return float(asset.price)
    
    def weighted_value(self, weight, total_pf_val):
        """
        gets the weighted value of the asset in terms of QUOTE (GBP)

        value: this is the quote amount invested in the base term of the base (e.g. GBP-BTC)
        """
        return abs(weight * float(total_pf_val)) # maximum precision is 8 d.p. 

    def to_base_value(self, asset, weighted_value) -> float:
        """
        gets weighted value in terms of BASE currency, e.g. 0.5 BTC 
        """
        return weighted_value / self.get_price(asset) # prices will cancel leaving weight * value_invested_in_base
    
    def base_to_quote(self, accounts: dict[float], asset: str):
        """
        BASE to QUOTE (GBP), e.g. given 0.5 BTC = 40,000 GBP 
        """
        value_base = accounts.get(self.get_base(asset), 0) # if we can not find the asset yield 0
        return value_base * self.get_price(asset) # invested amount in quote (GBP)
    
    def order_type(self, weight: float):
        """
        determines if we are buying or selling
        """
        return 'SELL' if weight < 0 else 'BUY'
    
    def total_portfolio_value(self):
        """
        loops over portfolio and calculates all investments in QUOTE (GBP), e.g. [GBP-BTC: 0.5, GBP-SOL: 2] = [BTC-GBP: 40,000, SOL-GBP: 400]
        """
        accounts = self.get_user_accounts()
        self.cash = accounts.get('GBP', 0) # get GBP else return 0
        invested = 0.0
        for key in accounts.keys():
            if key == 'GBP' or key in self._unpriceable_bases:
                continue
            try:
                invested += self.base_to_quote(accounts, key) # always in terms of BASE, transfer to QUOTE to standerdise
            except coin_error.CoinDataError:
                self._unpriceable_bases.add(key) # no priceable {key}-GBP pair (delisted/renamed/fiat/unsupported) - exclude from valuation going forward
        return float(invested + self.cash)

    def get_real_weights(self, portfolio_tickers) -> list[float]:
        """will get the weight of each specified asset in the portfolio"""
        accounts = self.get_user_accounts()
        pf_tickers_base = [self.get_base(ticker) for ticker in portfolio_tickers] # get base list e.g. [BTC, ETH, SOL, ... ]
        pf_value = self.total_portfolio_value()
        return [max(self.base_to_quote(accounts, key) / pf_value, 0) for key in pf_tickers_base] # base investemnt in quote terms as fraction of total quote e.g. BTC-GBP / Portfolio GBP

    # include a get incrument function, that gets the required precision for each asset, the base_size/quote size must be an mutliple of the base/quote increment
    def order_value_to_increment(self, 
                                 asset: str, 
                                 order_value: float, 
                                 increment_type = 'base_increment') -> float:
        """BASE-incruments vary per asset and order type"""
        if increment_type not in ['base_increment', 'quote_increment']:
            raise KeyError('must be base_incrument or quote_incrument')

        try:
            product = self.client.get_product(asset)
        except Exception as e:
            raise coin_error.CoinDataError(f'{asset} query error: {e}') from e

        increment = Decimal(product[increment_type])
        order_value = Decimal(str(order_value))

        # round to integer and then scale (for QUOTE orders increment is 1 for all assets -> order value only needs to be rounded)
        integer = (order_value // increment) 
        adjusted_order_value = integer * increment 

        # if buy trade get min(quote_size_buy, cash_available), to inforce quote_size_buy <= cash_avialble
        if increment_type == 'quote_increment':
            adjusted_order_value = min(adjusted_order_value, self.cash)

        # check we meet minimum order requirements (post cash-clamp, since clamping can push us back below the minimum)
        min_side_value = 'quote_min_size' if (increment_type == 'quote_increment') else 'base_min_size'
        min_size = Decimal(str(product[min_side_value]))
        if adjusted_order_value < min_size: # if order amount is less than min increment skip order request
            return 0

        return float(adjusted_order_value)

    def market_order_quantity(self, 
                              asset, 
                              weight, 
                              total_portfolio_value, 
                              full_close = False) -> list[str, float]:
        """
        gets order type and order value in BASE:QUOTE 
        
        If investing first time, when looping through this function, base_value will raise a keyError as to_base_value() will not be able to find key in accounts.
        To avoid this we use .get(), if the key is not found we return 0 and base_size is 0 which is valid. 
        """
        accounts = self.get_user_accounts()

        # orders
        if weight is not None:
            order_type = self.order_type(weight)

            w_value = self.weighted_value(weight, total_pf_val = total_portfolio_value)
            base_value = self.to_base_value(asset, w_value) # for SELL, value in BASE (e.g. BTC)

            base_size_sell = self.order_value_to_increment(asset, base_value, increment_type = 'base_increment') # SELL in BASE
            quote_size_buy = self.order_value_to_increment(asset, w_value, increment_type = 'quote_increment') # BUY in QUOTE

            # full close = | SELL using BASE                          , if closing out BUY
            #              | BUY using QUOTE (not supported for spot) , if closing out SELL
            # e.g. bought BTC-GBP using GBP, to close trade must close in BTC (because our position is enumerated in BTC)
            order_value_standard = {
                'base_size': str(base_size_sell) 
                } if (order_type == 'SELL') else {
                    'quote_size': str(quote_size_buy)
                    } 
            
        # full close amount
        asset_value = self.asset_base(asset, account_values = accounts) # for BUY, get quantity invested in asset class in BASE currency (float type)
        if full_close:
            asset_value = self.order_value_to_increment(asset, asset_value, increment_type = 'base_increment') # round down to a valid base increment
        order_value = {'base_size': str(asset_value)} if full_close else order_value_standard
        order_side = 'SELL' if full_close else order_type 

        return order_side, order_value

    def create_asset_order(self, 
                           asset, 
                           account_balance, 
                           weight = None, 
                           full_close = False, 
                           **kwargs):
        """
        Coinbase Advanced API python SDK function already knows what account order is sent to
        - BUY order = QUOTE size
        - SELL order = BASE size

        Places a market order for `asset`. Used both to open a new position (weight)
        and to rebalance/close an existing one (weight as the weight diff, or full_close).
        """
        order_type, order_value = self.market_order_quantity(asset, weight, total_portfolio_value = account_balance, full_close = full_close)
        order = self.client.create_order(
            client_order_id = str(uuid.uuid4()), # must be JSON serialisable, uuid is unique for each opened / closed trade
            product_id = asset,
            side = order_type,
            order_configuration= {
                'market_market_ioc': order_value # order fills at best available market price
            },
            **kwargs
        )
        if not order.success:
            raise coin_error.CoinOrderError(asset, order_type, order.error_response)

        return order

    # function to close out positions
    def modify_asset_order(self, 
                           asset, 
                           total_pf_value, 
                           weight_diff = None, 
                           full_close = False, 
                           **kwargs):
        """
        closes/edit's open positions - thin wrapper around create_asset_order

        Note: the close_position() endpoint canot be used for closing spot market positions. Only valid in future markets.
        """
        return self.create_asset_order(asset, account_balance = total_pf_value, weight = weight_diff, full_close = full_close, **kwargs)
    
    def multi_asset_close(self, 
                          portfolio_tickers: dict, 
                          full_close: bool):
        """
        exist all open trades, returing investemnts to base account
        """
        orders = []
        if full_close:
            for key in portfolio_tickers:
                order = self.modify_asset_order(
                    asset = key, 
                    total_pf_value = None, # weights None, wo will not be passed
                    full_close = True
                )
                orders.append(order)
        
        return orders
        
    def multi_asset_invest(self,
                           portfolio_ticker_weights: dict[str, float],
                           account_base: str = "GBP") -> dict:
        """portfolio rebalancing function (the main signal to trade execuation function)"""
        accounts = self.get_user_accounts()
        equity = accounts[account_base]
        gbp_balance = accounts.pop(account_base) # get only invested amount / remove BASE account
        pf_val = self.total_portfolio_value() # initialize only once

        orders, weight_diffs = [], []
        res = {'orders': orders, 'weight_diffs': weight_diffs, 'account_balance': gbp_balance, 'total_spent': pf_val}

        que = {}
        # liquidate holdings outside the current asset universe first (priority to sells),
        # otherwise their value is never converted to cash and BUY orders below can starve for funds
        strategy_bases = {self.get_base(ticker) for ticker in portfolio_ticker_weights}
        for base, qty in accounts.items():
            if base not in strategy_bases and qty > 0 and base not in self._unpriceable_bases:
                asset = f"{base}-{account_base}"
                try:
                    orders.append(
                        self.modify_asset_order(asset = asset, total_pf_value = None, full_close = True)
                    )
                except (coin_error.CoinDataError, coin_error.CoinOrderError):
                    # holding has no tradeable {base}-{account_base} pair (delisted/renamed/unsupported)
                    # or the close order was rejected - skip it rather than aborting the whole rebalance
                    self._unpriceable_bases.add(base) # add to cached unpricable bases
                    continue

        for key, new_weight in portfolio_ticker_weights.items():
            # check if not invested. If invested (initilise portfolio), otherwise rebalance portfolio
            invested_qty = pf_val - self.cash
            if invested_qty <= 1 or len(accounts) == 0: # possible to have currency left in each asset after closing out positions (unlikley to be alot)
                orders.append(
                    self.create_asset_order(asset = key, account_balance = equity, weight = new_weight) # invest weighted amount from initial balance
                    ) # we invest once and then rebalance
            else:
                # now modify portfolio buy taking opposite / same position in asset
                current_weight = max(self.base_to_quote(accounts, key) / pf_val, 0) # value invested in asset as fraction of total portfolio value in GBP, take max to avoid divide by 0 error
                weight_diff = new_weight - current_weight # new - old
                weight_diffs.append(weight_diff)
                side = self.order_type(weight_diff)

                # queue buys and exacute sell type orders
                # we have to give priority to sell orders, otherwise my may try add to a position and get an INSUFFICIENT_FUND error
                if side == 'SELL':
                    orders.append(
                        self.modify_asset_order(asset = key,
                                                total_pf_value = pf_val,
                                                weight_diff = weight_diff)
                                                ) # for each asset/base divest if full close
                # queue BUY side orders
                else:
                    # query order json
                    que[key] = {
                        'asset': key,
                        'weight_diff': weight_diff
                    }

        # now we have enough fiat in GBP to complete the BUY trades, the sum to 1 constarint ensures this is possible
        if len(que) != 0:
            accounts = self.get_user_accounts()
            pf_val = self.total_portfolio_value() # re-calculate total pf-value to factor in fees
            for key, inputs in que.items():
                orders.append(
                    self.modify_asset_order(total_pf_value = pf_val, **inputs)
                )

        return res