# coinbase App API (via python SDK) allows us to  programmaticaly manage our coinbase accounts
# we can get data, set orders on accounts and / or transfer funds between accounts 
# coinbase python SDK provides us with libraries and tools required to use coinbase API features in python 
# Python SDK docs
# https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/introduction
# Python SDK HTML  
# https://coinbase.github.io/coinbase-advanced-py/coinbase.rest.html
# note difference between RESTClient and Websocket 
# - RESTClient = synchronous pull of data & account actions, -> we can querey endpoints using rest 
# - WSClient = asynchronous push of live market data.

# Get imports
# =============================================
# for trading 
from coinbase.rest import RESTClient # Install Coinbase Advanced API Python SDK  
import uuid
import json
# for data
from sklearn.model_selection import train_test_split
import datetime 
import pandas as pd
import numpy as np 
from decimal import Decimal

class CoinbaseTrader:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.authenticated = False
    
    def login(self):
        """Authentication function:
        """
        try:
            account = self.client.get_accounts() # attempts to retreive account data to authorsie login 
            if account is not None:
                self.authenticated = True
        except Exception as e:
            print(e)
        return self.authenticated
    
    def tickers_weight(self, tickers: list, weights: list):
        """Creates Ticker-Weight Dict 
        """
        order_message = dict(zip(tickers, weights))
        return order_message
    
    def get_base(self, asset: str):
        return asset.split("-")[0] 
    
    def asset_base(self, asset: str, account_values: dict) -> float: 
        """Base Investment:
        """
        asset = self.get_base(asset)
        return account_values[asset] # asset associted holding, e.g. BTC-USD -> BTC: x  

    def coinbase_data(self, products, time_frame_candle, get_returns = True, freq = '1h', **kwargs): 
        """Get Close / Price Data
        **kwargs: datetime.timedelta() input 

        supports 1min -> 1 Day data requests, limited to 1 Day 
        """
        # final dataframe
        data = pd.DataFrame()

        # time (range from past -> present)
        end_time = datetime.datetime.today()
        start_time = end_time - datetime.timedelta(**kwargs) # the histroic time 
        date_range = len(pd.date_range(start = start_time, end = end_time, freq = freq))

        start_time = str(int(start_time.timestamp()))
        end_time = str(int(end_time.timestamp()))

        # get data 
        for product in products:
            candles = self.client.get_public_candles(
                product_id = product, 
                start = start_time, 
                end = end_time, 
                granularity = time_frame_candle, 
            )        

            # filter data 
            closes = [float(candle['close']) for candle in candles['candles']][::-1] # turn string data to float, and invert list to chronological order
            try:
                data[product] = closes
            except Exception as e:
                try:
                    data[product] = np.pad(closes, pad_width = (0, date_range - len(closes)), mode = 'edge').tolist()[1:] # edge forward/backward fills with last value
                except Exception as e:
                    print(f'failed {product} download')
            pass

        if get_returns: 
            data = data.pct_change().dropna(how = 'any') # will drop first row 
        else:
            data = data.iloc[1:, :] # makre sure datasets are equal length
        
        return data  

    def get_user_accounts(self) -> dict[float]:
        """Account Information Function
        """
        accounts = self.client.get_accounts() # all authenticated user accounts
        account_values = {}
        for account in accounts['accounts']:
            value = account['available_balance']['value']
            account_values[account['currency']] = float(value)
            #print(f"{account['currency']} -> {account['uuid']} | value {value}")
        
        return account_values
        
    def get_asset_changes_price(self, assets: list):
        change_price = {"change": [], "price": []}
        for asset in assets:
            product = self.client.get_product(product_id = asset) # product endpoint data  
            percent_change = float((product.price_percentage_change_24h).split()[0]) / 100 # split to drop '%'
            price = float(product.price)
            change_price['change'].append(percent_change)
            change_price['price'].append(price)

        return change_price   

    def get_price(self, asset: str, suffix = 'GBP'):
        """ Gets the asset price
        asset: can be in base:quote form or base form
        suffix: the quote currency if only the base is provided
        """
        if len(asset.split('-')) != 2: # if base e.g. 'BTC' add the GBP so we can pull a valid listed pair
            asset = asset + '-' + suffix

        # attempt to pull asset data, otherwise return 0 
        try:
            asset = self.client.get_product(asset)
        except Exception as e:
            return 0
        return float(asset.price)
    
    def weighted_value(self, weight, total_pf_val, asset):
        """Gets the weighted value of the asset in terms of quote (GBP)
        value: this is the quote amount invested in the base term of the base (e.g. GBP-BTC)
        """
        return abs(weight * float(total_pf_val)) # maximum precision is 8 d.p. 

    def to_base_value(self, asset, weighted_value) -> float:
        """Gets weighted value in terms of asset
        """
        return weighted_value / self.get_price(asset) # prices will cancel leaving weight * value_invested_in_base
    
    def base_to_quote(self, accounts: dict[float], asset: str):
        """Value invested in terms of quote (GBP)
        """
        value_base = accounts.get(self.get_base(asset), 0) # if we can not find the asset yield 0
        return value_base * self.get_price(asset) # invested amount in quote (GBP)
    
    def order_type(self, weight: float):
        '''Determines if we are buying or selling
        '''
        return 'SELL' if weight < 0 else 'BUY'
    
    def total_portfolio_value(self):
        '''Will get the total invested portfolio value in quote (GBP)
        '''
        accounts = self.get_user_accounts()
        return float(sum([self.base_to_quote(accounts, key) for key in accounts.keys() if key != 'GBP'])) # always in terms of base, transfer to quote to standerdise

    def get_real_weights(self, portfolio_tickers) -> list[float]:
        '''Will get the weight of each asset in the portfolio
        '''
        accounts = self.get_user_accounts()
        pf_tickers_base = [self.get_base(ticker) for ticker in portfolio_tickers]
        pf_value = self.total_portfolio_value()
        return [max(self.base_to_quote(accounts, key) / pf_value, 0) for key in pf_tickers_base]

    # include a get incrument function, that gets the required precision for each asset, the base_size/quote size must be an mutliple of the base/quote increment
    def order_value_to_increment(self, asset: str, order_value: float, increment_type = 'base_increment') -> float:
        '''Base-incruments vary per asset and IOC order type
        '''
        if increment_type not in ['base_increment', 'quote_increment']:
            raise ValueError('increment type must be in list')

        product = self.client.get_product(asset)
        increment = Decimal(product[increment_type])
        order_value = Decimal(str(order_value))

        # round to integer and then scale (for quote orders increment is 1 for all assets -> order value only needs to be rounded)
        integer = (order_value // increment) 
        adjusted_order_value = integer * increment 

        # check we meet minimum order requirements
        min_side_value = 'quote_min_size' if increment_type == 'quote_increment' else 'base_min_size' # we do not need to deal with max value conditions unless you can buy 200BTC :)
        min_size = Decimal(str(product[min_side_value]))
        if adjusted_order_value < min_size: # if order amount is less than min increment skip order request
            return 0 
        
        return float(adjusted_order_value)

    def market_order_quantity(self, asset, weight, total_portfolio_value, full_close = False) -> list[str, float]:
        """Gets order type and order value in base/quote
        note if investing first time, when looping thriugh this function, base_value will raise a keyError as to_base_value() will not be able to find key in accounts.
        To avoid this we use .get(), if the key is not found we return 0 and base_size is 0 which is valid. 
        """
        accounts = self.get_user_accounts()

        # orders
        if weight is not None:
            order_type = self.order_type(weight)

            w_value = self.weighted_value(weight, total_pf_val = total_portfolio_value, asset = asset)
            base_value = self.to_base_value(asset, w_value) # for SELL, value in base (e.g. BTC)

            base_size_sell = self.order_value_to_increment(asset, base_value, increment_type = 'base_increment') # SELL in base
            quote_size_buy = self.order_value_to_increment(asset, w_value, increment_type = 'quote_increment') # BUY in quote

            # if full close, to close out BUY, SELL using base
            # if full close, to close out SELL, BUY using quote (not supported for spot)
            order_value_standard = {
                'base_size': str(base_size_sell) 
                } if (order_type == 'SELL') else {
                    'quote_size': str(quote_size_buy)
                    } 
            
        # full close amount 
        asset_value = self.asset_base(asset, account_values = accounts) # for BUY, get quantity invested in asset class (float type)   
        order_value = {'base_size': str(asset_value)} if full_close else order_value_standard
        order_side = 'SELL' if full_close else order_type 

        return order_side, order_value

    def create_asset_order(self, asset, account_balance, weight, **kwargs): 
        """Order Function:
        Coinbase Advanced API python sdk function already knows what account order is sent to
        - BUY order = quote size 
        - SELL order = base size 
        """
        order_type, order_value = self.market_order_quantity(asset, weight, total_portfolio_value = account_balance)
        try: 
            order = self.client.create_order(
                client_order_id = str(uuid.uuid4()), # must be JSON serialisable, uuid is unique for each opened / closed trade
                product_id = asset, 
                side = order_type, 
                order_configuration= {
                    'market_market_ioc': order_value
                },
                **kwargs
            )
        except Exception as e: 
            print(e)

        return order  
    
    # function to close out positions
    def modify_asset_order(self, asset, total_pf_value, weight_diff = None, full_close = False, **kwargs):
        """Close/Edit Open Positions:
        Note: the close_position() endpoint canot be used for closing spot market positions. Only valid in future markets.
        """
        order = None
        order_modify_type, order_value = self.market_order_quantity(asset, weight_diff, total_portfolio_value = total_pf_value, full_close = full_close)
        try: 
            order = self.client.create_order(
                client_order_id = str(uuid.uuid4()), # ceate unique order id 
                product_id = asset,
                side = order_modify_type, # 'SELL' all holdings if full close (manual overide)
                order_configuration = {
                    'market_market_ioc': order_value 
                }, 
                **kwargs
            )
        except Exception as e: 
            print(e)
        
        return order
    
    def multi_asset_close(self, portfolio_tickers: dict, full_close):
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
        
    def multi_asset_invest(self, portfolio_ticker_weights: dict, account_base = "GBP", shut_down = False):
        """Portfolio Rebalancing Function:
        """
        accounts = self.get_user_accounts()
        equity = accounts[account_base]
        gbp_balance = accounts.pop(account_base) # get only invested amount / remove base account 
        total_portfolio_value = self.total_portfolio_value() # initialize only once 
        
        orders, weight_diffs = [], []
        
        que = {}
        if shut_down:
            return 
        else: # can be Any type 
            for key, new_weight in portfolio_ticker_weights.items():
                # check if not invested, if so invest (initilise portfolio), otherwise rebalance portfolio
                if total_portfolio_value <= 1 or len(accounts) == 0: # possible to have currency left in each asset after closing out positions (unlikley to be alot)
                    que = {}
                    try: 
                        orders.append(
                            self.create_asset_order(asset = key, account_balance = equity, weight = new_weight) # invest weighted amount from initial balance
                            ) # we invest once and then rebalance
                    except Exception as e:
                        print(e)
                else: 
                    try:
                        # now modify portfolio buy taking opposite / same position in asset
                        current_weight = max(self.base_to_quote(accounts, key) / total_portfolio_value, 0) # value invested in asset as fraction of total portfolio value in GBP, tak max to avoid divide by 0 error
                        weight_diff = new_weight - current_weight # new - old
                        weight_diffs.append(weight_diff)
                        side = self.order_type(weight_diff)

                        # we have to give priority to sell orders, otherwise my may try add to a position and get an INSUFFICIENT_FUND error 
                        if side == 'SELL': 
                            orders.append(
                                self.modify_asset_order(asset = key, 
                                                        total_pf_value = total_portfolio_value,
                                                        weight_diff = weight_diff)
                                                        ) # for each asset/base divest if full close 
                        else: 
                            que[key] = {
                                'asset': key, 
                                'total_pf_value': total_portfolio_value,
                                'weight_diff': weight_diff
                            }
                    except Exception as e:
                        print(e)

            # now we have enough fiat in GBP to complete the BUY trades, the sum to 1 constarint ensures this is possible 
            if len(que) != 0:
                for key, inputs in que.items():
                    try:
                        orders.append(
                            self.modify_asset_order(**inputs)
                        )
                    except Exception as e:
                        print(e)

        return {'orders': orders, 
                'weight_diffs': weight_diffs, 
                'account_balance': gbp_balance, 
                'total_spent': total_portfolio_value, 
                'strategy_live': not shut_down} # orders will be a list of dicts 
    