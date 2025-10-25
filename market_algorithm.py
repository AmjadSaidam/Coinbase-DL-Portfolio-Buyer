# imports (only read by python process at import time)
import pandas as pd 
import numpy as np
import dmlstm_trading as dmlstm 
import coinbase_order_functions as cb_trade 
import datetime as dt 
import schedule 
import json
from scipy.stats import skewnorm
# get strategy status from telegram bot (runs in same interpreter)
from telegram_bot import shut_down_state_file_name, load_json # need file name do we can load written josn data
# store results in a sql database 
import data_to_sql 

# pull data and train model 
def account_path(path_cdp: str): 
    cdp_api_keys = load_json(path_cdp)
    
    # authenticate login 
    cb_acc = cb_trade.CoinbaseTrader(api_key = cdp_api_keys['name'], api_secret = cdp_api_keys['privateKey'])
    cb_acc.login()
    
    return cb_acc if cb_acc.authenticated else print('failed account link') 
        
# get data 
def get_feature_data(coinbase_account, tickers: list, cb_timeframe, frequency, **kwargs):

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
    # **kwargs = a, loc, scale
    return skewnorm.rvs(**kwargs, size = n_samples)

# get data for model 
def model_batch_data(return_data, price_data, batching_lookback, **kwargs): 
    x_batch_data, y_batch_data = dmlstm.prepare_features(asset_returns = np.array(return_data), 
                                                         asset_prices = np.array(price_data), 
                                                         target_returns = target_dist_sampler(n_samples = len(return_data),**kwargs),
                                                         lookback = batching_lookback)
    
    return {'dmlstm_features': x_batch_data, 'dmlstm_labels': y_batch_data}

# runs all models
def run_process(process, every: int, time_unit: str, **kwargs):
    """ Runs all Process
    process: the job we want to exacute  
    every: multiple of time, we append this string as a key to the schedule module 
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
portfolio_assets = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP'] # can include more assets for large portfolio

# run all 
if __name__ == '__main__': 
    # connect to coinbase advanced account 
    coinbase_account = account_path('cdp_activate_api_key.json')
    # for data functions
    data_frequency = 'ONE_HOUR'
    freq = '1h'
    data_max = {'hours': 350}
    # schedule data, predict, order times
    schedule_every = 1
    schedule_tf = 'hours'
    # schedule training times 
    schedule_training_every = 1
    schedule_training_tf = 'D' # to replicate test set in backtest do 350*(1 - train_fraction) * tf
    # time for real weights upload (their is lag to account for, as order although sent instantely do no exacute on the exchange instantelly)
    schedule_real_weights_every = 40 # could choosen 1min although we would then have 60 updates per minute
    schedule_real_weights_tf = 'minutes'
    # min invest quantity
    min_weight = 0.2 # min % of portfolio value investment
    # instantiate database for real time data upload 
    model_db = data_to_sql.CreateSQLiteDatabase(tickers = portfolio_assets, name_database = 'model_output.db', name_data = 'model_output')
    model_db.create_access_database_file()
    model_db.create_data()
    real_weight_db = data_to_sql.CreateSQLiteDatabase(tickers = portfolio_assets, name_database = 'real_invested_weights.db', name_data = 'real_invested_weights')
    real_weight_db.create_access_database_file()
    real_weight_db.create_data()

    # we only want to pull new data every 4H
    scheduled_output = None
    def scheduled_data():
        global scheduled_output # otherwise local can not-overite this value 
        # get initial data and model predictions 
        initial_data = get_feature_data(coinbase_account = coinbase_account, 
                                        tickers = portfolio_assets, 
                                        cb_timeframe = data_frequency, 
                                        frequency = freq, 
                                        **data_max)
        
        if initial_data:
            scheduled_output = initial_data

        return print('data', scheduled_output['return_data'], scheduled_output['price_data']) # for debug

    # model training function
    model = None 
    hyper_hidden_dims = 225
    hyper_lookback = 9
    target_dist_param = {"a": 3, 'loc': 0.01, 'scale': 0.04}
    def train_dmlstm():
        global model 
        returns_all = np.array(scheduled_output['return_data']) # possibly abstract away into run_process using **kwargs
        prices_all = np.array(scheduled_output['price_data'])
        target_returns_all = np.array(target_dist_sampler(n_samples = len(returns_all), **target_dist_param))
        model = dmlstm.backprop_rnn(asset_returns = returns_all, 
                                    asset_prices = prices_all, 
                                    target_returns = target_returns_all, 
                                    lookback = hyper_lookback, 
                                    batch_size = 64, 
                                    hidden_dim = hyper_hidden_dims, 
                                    num_layers = 2,
                                    n_epochs = 100, 
                                    learning_rate = 0.001,
                                    min_weight_constraint = True, 
                                    min_weight_value = min_weight,  
                                    show_progress = True)
        
        return print('trained model', model) # for debug 
    
    # model predictions 
    predicted_portfolio_weights = None
    def model_prediction():
        global predicted_portfolio_weights
        dmlstm_model = model['trained_model']
        # since train_test_split_time_series() is not used data needs to be converted to numpy array
        feature_labels = model_batch_data(return_data = scheduled_output['return_data'], 
                                        price_data = scheduled_output['price_data'], 
                                        batching_lookback = hyper_lookback, 
                                        **target_dist_param)
        all_predicted_weights = dmlstm_model(feature_labels['dmlstm_features'], use_weight_constraint = True, weight_constraint = min_weight) # pass all batches then subset for final weight as LSTM learnes patterns from earlier batches 
        predicted_portfolio_weights = all_predicted_weights.detach().numpy()[-1].tolist() # get most recent prediction

        # attempty to get real weights
        try:
            pre_rebalance_weights = coinbase_account.get_real_weights(portfolio_assets)
        except Exception as e:
            pass
    
        return print('pre-rebalance portfolio weights ------->', pre_rebalance_weights, 
                     '\nmodel prediction ------->', predicted_portfolio_weights) # for debug 

    # send market orders by instantiating portfolio or rebalancing
    order_to_market = None 
    def order_to_exchange(**kwargs):
        global order_to_market
        # check if user has prompted telegram bot to stop strategy (with effect in next scheduled call)
        try:
            strategy_shut_command = load_json(file = shut_down_state_file_name)['status'] # filter for required key
            print('strategy shut down ------->', strategy_shut_command)

            # send order
            order_to_market = coinbase_account.multi_asset_invest(
                portfolio_ticker_weights = coinbase_account.tickers_weight(**kwargs),
                shut_down = bool(strategy_shut_command) # check is user has set shut down to True, if so hault strategy 
            )
        except Exception as e:
            print(e) 

        return print(order_to_market)
    
    # model output weights to database 
    def weights_to_database(): 
        model_db.list_to_data(weights = predicted_portfolio_weights)
    
    # real weights to database
    def get_real_weights():
        # attempt to get weights. If not invested, to avoid division be zero error, run try except block
        try:
            real_weights = coinbase_account.get_real_weights(portfolio_tickers = portfolio_assets)
        except Exception as e:
            real_weights = [0 for _ in portfolio_assets] # if closing out our real weights will be 0 vectors
        real_weight_db.list_to_data(weights = real_weights)

        print('post-rebalance portfolio weights ------->', real_weights)

    # initial data pull -> model train -> model predict
    scheduled_data()
    train_dmlstm()
    model_prediction()
    order_to_exchange_input_dict = {'tickers': portfolio_assets, 'weights': predicted_portfolio_weights} # after the above functions run the global variables are updated, avoiding a NoneType error
    order_to_exchange(**order_to_exchange_input_dict)

    # save model output -> save the total invested in each asset as fraction of total invested (should be very close to model output)
    weights_to_database()

    # schedule future updates in order of initial run 
    # pull data (every data_frequency e.g. 4H)
    run_process(process = scheduled_data, every = schedule_every, time_unit = schedule_tf) # adjust get_feature_data to include global callback 

    # train model (every retrain period e.g. D)
    run_process(process = train_dmlstm, every = schedule_training_every, time_unit = schedule_training_tf)

    # use model to predict (every data_frequency e.g. 4H)
    run_process(process = model_prediction, every = schedule_every, time_unit = schedule_every) # training many models on the same data, outputs similar results

    # create portfolio using model predictions (every data_frequency e.g 4H)
    run_process(process = order_to_exchange, every = schedule_every, time_unit = schedule_tf, **order_to_exchange_input_dict)

    # upload model weights and real weights to SQLight database 
    run_process(process = weights_to_database, every = schedule_every, time_unit = schedule_tf)

    # Check if we invested the model predicted weights, time must be less than others
    run_process(process = get_real_weights, every = schedule_real_weights_every, time_unit = schedule_real_weights_tf) # takes some time for transcations to show on accounts

    # Re-train model every set time - just for debug
    run_process(process=f, every = 1, time_unit = 'seconds')
    
    # run process
    print('Press Ctrl+C to hault process')
    run_schedule()
