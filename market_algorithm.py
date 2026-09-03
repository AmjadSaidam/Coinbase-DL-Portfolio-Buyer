# imports (only read by python process at import time)
import pandas as pd 
import numpy as np
import models.lstm_trading as lstm 
import trading_logic.coinbase_order_functions as cb_trade 
import datetime as dt 
import schedule 
import time
from scipy.stats import skewnorm
import torch
import torch.utils.data as data
# get strategy status from telegram bot (runs in same interpreter)
from telegram_bot import shut_down_state_file_name, load_json # need file name so we can load written json data
# store results in a sql database 
import data_loaders.data_to_sql as data_to_sql 

# pull data and train model 
def account_path(path_cdp: str): 
    """
    """
    cdp_api_keys = load_json(path_cdp)
    
    # authenticate login 
    cb_acc = cb_trade.CoinbaseTrader(api_key = cdp_api_keys['name'], api_secret = cdp_api_keys['privateKey'])
    cb_acc.login()
    
    return cb_acc if cb_acc.authenticated else print('failed account link') 
        
# get data 
def get_feature_data(coinbase_account, tickers: list, cb_timeframe, frequency, **kwargs):
    """
    """

    return_data = coinbase_account.coinbase_data(
        products = tickers, 
        time_frame_candle = cb_timeframe,
        freq = frequency, 
        **kwargs
    )
    price_data = coinbase_account.coinbase_data(
        products = tickers, 
        time_frame_candle = cb_timeframe,
        freq = frequency, 
        get_returns = False, 
        **kwargs
    )

    return {'return_data': return_data, 'price_data': price_data}

# target return data 
def target_dist_sampler(n_samples, **kwargs): 
    """
    """
    # **kwargs = a, loc, scale
    return skewnorm.rvs(**kwargs, size = n_samples)

# get data for model 
def model_batch_data(return_data, price_data, target_returns, batching_lookback, batch_size) -> data.DataLoader: 
    """
    """
    x_batch_data, y_batch_data, x_inv_batch = lstm.prepare_features(asset_returns = return_data, 
                                                         asset_prices = price_data, 
                                                         target_returns = target_returns,
                                                         lookback = batching_lookback)
    loader = data.DataLoader(data.TensorDataset(x_batch_data, y_batch_data, x_inv_batch), batch_size = batch_size, shuffle = False, drop_last = False)
    
    return loader

# runs all models
def run_process(process, every: int, time_unit: str, **kwargs):
    """ 
    Runs all Process
    process: the job we want to exacute  
    every: multiple of time, used as key in schedule module 
    time: the timeframe 
    """
    return getattr(schedule.every(interval = every), time_unit).do(process, **kwargs) # if process has arguments call in .do() so process is callable

def run_schedule(): 
    while True: 
        schedule.run_pending()

# tester function 
def f():
    return print('idling | time:', dt.datetime.today().isoformat())

# out tickers
portfolio_assets = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP', 'ADA-GBP'] # can include more assets for large portfolio
n_assets = len(portfolio_assets)

# run all 
if __name__ == '__main__': 
    # connect to coinbase advanced account 
    coinbase_account = account_path('cdp_activate_api_key.json') # follow README.md instructions to get your key (then jsut copy path name into account_path()
    # for data functions
    data_frequency = 'ONE_HOUR'
    freq = '1h'
    data_max = {'hours': 350}
    # schedule data/model-training/model-predictions/orders and database upload
    schedule_every = 1
    schedule_tf = 'hours'
    # min invest quantity
    min_weight = 0.1 # min % of portfolio value to invest (min asset weight)
    # instantiate database for real time data upload 
    model_db = data_to_sql.CreateSQLiteDatabase(tickers = portfolio_assets, name_database = 'model_output.db', name_data = 'model_output')
    model_db.create_access_database_file()
    model_db.create_data()
    real_weight_db = data_to_sql.CreateSQLiteDatabase(tickers = portfolio_assets, name_database = 'real_invested_weights.db', name_data = 'real_invested_weights')
    real_weight_db.create_access_database_file()
    real_weight_db.create_data()
    # instentiate lstm model 
    model = lstm.lstm(input_dim = n_assets*2, output_dim = n_assets, hidden_dim = 225, weight_constraint = True, min_weight = min_weight)
    model.lkb = 10

    # we only want to pull new data every 4H
    scheduled_output = None
    def scheduled_data():
        """
        """
        global scheduled_output # otherwise local can not-overite this value 
        # get initial data and model predictions 
        initial_data = get_feature_data(coinbase_account = coinbase_account, 
                                        tickers = portfolio_assets, 
                                        cb_timeframe = data_frequency, 
                                        frequency = freq, 
                                        **data_max)
        
        if initial_data:
            scheduled_output = initial_data

        print(f'return data: \n{scheduled_output['return_data']}')
        print(f'price data: \n{scheduled_output['price_data']}')
        print('-------------------------------------------------------') 

    # model training function
    target_dist_param = {"a": 3, 'loc': 0.01, 'scale': 0.04}
    returns_target = None 
    def train_dmlstm():
        """
        """
        global returns_target
        # train model on all but last value 
        rt = np.array(scheduled_output['return_data'])
        pr = np.array(scheduled_output['price_data'])
        returns_target = target_dist_sampler(n_samples = len(rt), **target_dist_param)
        rt_t = np.array(returns_target) # target portfolio return distribution
        # build loader 
        tr_loader = model_batch_data(rt[:-1], pr[:-1], rt_t[:-1], batch_size = 64, batching_lookback = model.lkb)
        # train model
        model.lstm_train(tr_loader)
        
        print(f'Trained model loss = {model.tr_loss}')
        print('-------------------------------------------------------') 
    
    # model predictions 
    predicted_portfolio_weights = None
    def model_prediction():
        """
        """
        global predicted_portfolio_weights
        # test set 
        rt = np.array(scheduled_output['return_data'])[-1]
        pr = np.array(scheduled_output['price_data'])[-1]

        # evaluate model for one single feature observation
        model.model.eval()
        features = np.concat([rt, pr])
        feature_obv = torch.as_tensor(features, dtype = torch.float32).reshape(1, 1, n_assets*2) # (batch, sequence_length, input_dim)
        with torch.no_grad():
            yhat = model.model(feature_obv, use_weight_constraint = model.w_constraint, weight_constraint = model.w_min)
            predicted_portfolio_weights = yhat.detach().numpy().tolist()[0]
        
        # attempty to get real weights
        try:
            pre_rebalance_weights = coinbase_account.get_real_weights(portfolio_assets)
        except Exception as e:
            pre_rebalance_weights = None 
    
        print(f'pre-train/predict portfolio weights = {pre_rebalance_weights}')
        print('-------------------------------------------------------')
        print(f'predicted portfolio weights = {predicted_portfolio_weights}') # for debug 
        print('-------------------------------------------------------')

    # send market orders by instantiating portfolio or rebalancing
    order_to_market = None 
    def order_to_exchange():
        """
        """
        global order_to_market, predicted_portfolio_weights
        # check if user has prompted telegram bot to stop strategy (with effect in next scheduled call)
        try:
            strategy_shut_command = load_json(file = shut_down_state_file_name)['status'] # filter for required key
            print(f'strategy shut down = {strategy_shut_command}')
            print('-------------------------------------------------------') 

            # ticker/weight model prediction 
            order_to_exchange_input_dict = coinbase_account.tickers_weight(portfolio_assets, predicted_portfolio_weights)
            
            # model prediction to order message 
            order_to_market = coinbase_account.multi_asset_invest(
                portfolio_ticker_weights = order_to_exchange_input_dict,
                shut_down = bool(strategy_shut_command) # check is user has set shut down to True, if so hault strategy 
            )
        except Exception as e:
            print(e) 

        print(f'market order message: \n{order_to_market}')
        print('-------------------------------------------------------')
    
    # model output weights to database 
    def weights_to_database(): 
        """
        """
        model_db.list_to_data(weights = predicted_portfolio_weights)
    
    # real weights to database
    def get_real_weights():
        """
        """
        # attempt to get weights. If not invested, to avoid division be zero error, run try except block
        try:
            real_weights = coinbase_account.get_real_weights(portfolio_tickers = portfolio_assets)
        except Exception as e:
            real_weights = [0 for _ in portfolio_assets] # if closing out our real weights will be 0 vectors
        real_weight_db.list_to_data(weights = real_weights)

        print(f'post-rebalance portfolio weights = {real_weights}')
        print('-------------------------------------------------------')

    # Build pipeline to inforce scheduling dependence: order is initial data pull -> model train -> model predict -> send orders -> get metadata -> loop 
    def pipeline():
        # 1) pull data
        scheduled_data()
        # 2) train model 
        train_dmlstm()
        # 3) model predictions 
        model_prediction()
        # 4) create signals and send orders 
        order_to_exchange()
        # 5) write SQL data:
        weights_to_database()
        time.sleep(15) # sleep and wait for orders to exacuate and reflect in account 
        get_real_weights()

    # call in synch pipeline 
    #schedule.every(schedule_every).minutes.do(pipeline)
    run_process(process = pipeline, every = schedule_every, time_unit = 'minutes')

    # Re-train model (dubug code) every set time 
    #run_process(process=f, every = 1, time_unit = 'seconds')
    
    # run process
    print('Press Ctrl+C to hault process')
    run_schedule()
