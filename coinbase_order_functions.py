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

class CoinbaseTrader:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.authenticated = False
    
    def login(self):
        """Authentication function:
        """
        try:
            _ = self.client.get_accounts()
            self.authenticated = True
            return self.authenticated
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
    
    def asset_base(self, asset: str, account_values: dict): 
        """Base Investment:
        """
        asset = self.get_base(asset)
        return account_values[asset] # asset associted holding, e.g. BTC-USD -> BTC    

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

    def get_user_accounts(self):
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
    
    def weighted_value(self, weight, value):
        return abs(weight * float(value)) # maximum precision is 8 d.p. 
    
    def order_type(self, weight):
        return 'SELL' if weight <= 0 else 'BUY'
    
    def to_base_value(self, asset, weighted_value):
        base_asset = self.client.get_product(asset)
        return weighted_value / float(base_asset.price)
    
    def base_to_quote(self, value_base):
        return (1 / value_base)

    def market_order_quantity(self, asset, weight, value, full_close = False):
        """Get Order Quantity in Base / Quote:
        """
        order_value = None
        order_type = self.order_type(weight) # overide if full close 
        w_value = self.weighted_value(weight, value)

        base_value = self.to_base_value(asset, w_value)

        asset_value = self.asset_base(asset, self.get_user_accounts()) # get quantity invested in asset class

        base_size_sell = round(base_value, 6) # up to 8 d.p. supported for base 
        quote_size_buy = round(w_value, 2) # up to 2 d.p. for fiate 

        # if full close, to close out BUY, SELL using base
        # if full close, to close out SELL, BUY using quote (not supported for spot)
        if not full_close:  
            order_value = {
                'base_size': str(base_size_sell) 
                } if (order_type == 'SELL') else {
                    'quote_size': str(quote_size_buy)
                    } 
        else:
            base_size_close = asset_value 
            order_value = {
                'base_size': str(base_size_close)
            }
            
        return order_type, order_value

    def create_asset_order(self, asset, weight, value, **kwargs): 
        """Order Function:
        Coinbase Advanced API python sdk function already knows what account order is sent to
        * BUY order = quote size 
        * SELL order = base size 
        """
        order_type, order_value = self.market_order_quantity(asset, weight, value)
        try: 
            order = self.client.create_order(
                client_order_id = str(uuid.uuid4()), # must be JSON serialisable, uuid is uniqe for each opend / closed trade
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
    def modify_asset_order(self, asset, account_values: dict, side_close: str, full_close = False, weight_diff = None, **kwargs):
        """Close/Edit Open Positions:
        Note: the close_position() endpoint canot be used for closing sport positions 
        """
        order = None
        asset_holding = self.asset_base(asset=asset, account_values = account_values) # get quote value invested in asset
        order_modify_type, order_value = self.market_order_quantity(asset, weight_diff, asset_holding, full_close = full_close)
        try: 
            order = self.client.create_order(
                client_order_id = str(uuid.uuid4()), # ceate unique order id 
                product_id = asset,
                side = 'SELL' if full_close else side_close, # 'SELL' all holdings if full close 
                order_configuration = {
                    'market_market_ioc': order_value 
                }, 
                **kwargs
            )
        except Exception as e: 
            print(e)
        
        return order
    
    def multi_asset_invest(self, portfolio_ticker_weights: dict, account_base = "GBP", full_close = False):
        """Portfolio Rebalancing Function:
        """
        accounts = self.get_user_accounts()
        equity = accounts[account_base]
        portfolio_value = accounts.pop(account_base) # get only invested amount / remove base account 
        total_portfolio_value = float(sum(accounts.values()))
        orders, weight_diffs = [], []
        
        for key, new_weight in portfolio_ticker_weights.items():
            # check if not invested, if so invest (initilise portfolio), otherwise rebalance portfolio
            if total_portfolio_value == 0 and not full_close: # avoid issue of sending BUY orders when full_close is true
                try: 
                    orders.append(
                        self.create_asset_order(asset = key, weight = new_weight, value = equity)
                        ) # we invest once and then rebalance
                except Exception as e:
                    print(e)
            else: 
                try:
                    base_value = accounts[self.get_base(key)]
                    current_weight = self.base_to_quote(base_value) / total_portfolio_value # value invested in asset as fraction of total portfolio value in GBP
                    weight_diff = new_weight - current_weight # new - old
                    weight_diffs.append(weight_diff)
                    side = self.order_type(weight_diff)

                    # now modify portfolio buy taking opposite / same position in asset
                    orders.append(
                        self.modify_asset_order(asset = key, 
                                                account_values = accounts, # currency value invested in asset
                                                side_close = side, 
                                                full_close = full_close, 
                                                weight_diff = weight_diff)
                                                ) # for each asset / divest base if full close 
                except Exception as e:
                    print(e)

        return {'orders': orders, 'weight_diffs': weight_diffs}
    