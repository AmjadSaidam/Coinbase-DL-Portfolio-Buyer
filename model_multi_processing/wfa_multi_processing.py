import numpy as np
from collections import defaultdict
from data.kaggle_data import kaggle_price_return_data
from data.features_and_labels import rolling_volatility, target_returns_label
import multiprocessing as mp 
import torch
import torch.utils.data as data 
import models.lstm_trading as lstm
import models.nn_model as nn_model
import pickle

def model_set_attributes(model: lstm, params: dict):
    """
    set model attributes for class
    """
    [setattr(model, k, v) for (k, v) in params.items()]
    pass

def lstm_pipeline(dim, train_loader, eval_loader, test_loader, loss_sharpe, target_vol): 
    """
    lstm train/eval/predict pipeline
    """
    model = lstm.lstm(dim*2, dim, weight_constraint = False, sharpe_loss = loss_sharpe)
    model_vs = lstm.lstm(dim*2, dim, shorts = True, weight_constraint = False, vol_scaling = True, sharpe_loss = loss_sharpe)

    params = {'p': 1, 'vol_trg': target_vol, 'vol_scale_lkb': 10}
    model_set_attributes(model, params)
    model_set_attributes(model_vs, params)

    # train, evaluate
    epochs = 10
    model.lstm_train(train_loader, eval_loader, n_epochs = epochs)
    model_vs.lstm_train(train_loader, eval_loader, n_epochs = epochs)

    # predict 
    res_model = model.lstm_evaluate(test_loader)
    res_model_vs = model_vs.lstm_evaluate(test_loader)

    return res_model, res_model_vs

def nn_pipeline(dim, train_loader, eval_loader, test_loader):
    """
    base neural network train/eval/predict pipleline
    """
    model = nn_model.simple_neural_net(dim, dim)
    # train 
    model.simple_neural_net_train(train_loader, eval_loader, epochs = 10)
    # predict 
    res_model = model.simple_neural_net_evaluate(test_loader)

    return res_model

def backtest(config) -> dict[str, np.ndarray]:
    """
    using prior returns/prices and current target returns conducts single train/eval/test 
    """
    price_data = config['price_data']
    return_data = config['return_data']
    target_data = config['target_data']
    current_returns = config['current_returns']   
    target_vol = config['target_vol']
    data_lookback = config['data_lookback']
    tr_split = config['tr_split']
    dls = config['dls']
    model_type = config['model_type']
    cutoff = config['cutoff']            

    # dims 
    dim = return_data.shape[1]
    # data
    bs = 64
    if model_type == 'lstm':
        # sequence data 
        rt_tr, rt_rest = lstm.train_test_split_time_series(return_data, tr_split)
        pr_tr, pr_rest = lstm.train_test_split_time_series(price_data, tr_split)
        y_tr, y_rest = lstm.train_test_split_time_series(target_data, tr_split)

        rt_eval, rt_ts = rt_rest[:-1, :], rt_rest[-data_lookback:, : ] 
        pr_eval, pr_ts = pr_rest[:-1, :], pr_rest[-data_lookback:, : ]
        y_eval, y_ts = y_rest[:-1], y_rest[-data_lookback: ]
        test_features = torch.concat([rt_ts, pr_ts], axis = 1)
        # loaders
        tr_loader = lstm.data_pre_process(rt_tr, pr_tr, y_tr, data_lookback, batches = bs)
        eval_loader = lstm.data_pre_process(rt_eval, pr_eval, y_eval, data_lookback, batches = bs)
        # custom test loader 
        x_ts = test_features.unsqueeze(0)
        rt_test = current_returns.unsqueeze(0)
        x_inv_ts = rt_ts.unsqueeze(0)
        y_ts = y_ts.unsqueeze(0)
        ts_loader = data.DataLoader(data.TensorDataset(x_ts, y_ts, rt_test, x_inv_ts), shuffle = False, drop_last = False, batch_size = 1)
        # prediction
        res_model, res_model_vs = lstm_pipeline(dim, tr_loader, eval_loader, ts_loader, dls, target_vol)

        payload =  {'no_vs': res_model, 'vs': res_model_vs}

    elif model_type == 'nn': 
        future_returns = return_data.roll(shifts = -1, dims = 0) # get current returns today 
        future_returns[-1, :] = current_returns 
        # split data 
        x_train, x_rest, y_train, y_rest = lstm.train_test_split(return_data, target_data, train_size = 0.6)
        n, m = x_train.shape
        rt_train, rt_rest = future_returns[:n, :], future_returns[n:, :]

        x_eval, x_test = x_rest[:-1 , :], x_rest[-1, :].unsqueeze(0)
        y_eval, y_test = y_rest[:-1], y_rest[-1].unsqueeze(0)
        rt_eval, rt_test = rt_rest[:-1, :], rt_rest[-1, :].unsqueeze(0)
        # loaders
        tr_loader = nn_model.data_pre_process(x_train, rt_train, y_train, batches = bs)
        eval_loader = nn_model.data_pre_process(x_eval, rt_eval, y_eval, batches = bs)
        ts_loader = nn_model.data_pre_process(x_test, rt_test, y_test, batches = 1)
        # prediction
        res_model = nn_pipeline(dim, tr_loader, eval_loader, ts_loader)
    
        payload = {'no_vs': res_model}
    
    return {
        'cutoff': cutoff, 
        'dls': dls, 
        'model_type': model_type,
        'model_predictions': payload
    }

def walk_forward_analysis(return_data, price_data, target_returns, lookback: int = 10, initial_split = 0.9995) -> dict[str, dict[str: list]]: 

    backtest_configs = []

    n, m = return_data.shape
    split = int(np.floor(n*initial_split))
    for dls in [False, True]: 
        for model_type in ['lstm', 'nn']: 
            if dls and (model_type == 'nn'):
                continue
            for cutoff in range(split, n):
                # wfa features 
                wfa_returns = return_data[cutoff - split:cutoff, :]
                wfa_prices = price_data[cutoff - split:cutoff, :] if (price_data is not None) else 0
                wfa_current_returns = return_data[cutoff, :] 
                # wfa label
                label = target_returns[cutoff - split: cutoff]
                # backtest
                backtest_configs.append(
                    {
                        'price_data': wfa_prices, 
                        'return_data': wfa_returns, 
                        'current_returns': wfa_current_returns,
                        'target_data': label, 
                        'target_vol': 0.01,
                        'data_lookback': lookback, 
                        'dls': dls,
                        'model_type': model_type,
                        'tr_split': 0.7, 
                        'cutoff': cutoff
                    }
                )
    return backtest_configs

def aggregate_results(mp_results: list[dict]): 
    """
    returns 
    (lstm)
    (lstm, dls)
    (nn)
    """
    buckets: dict[tuple, list] = defaultdict(list) 
    for r in mp_results:
        keys = (r['model_type'], r['dls'])
        buckets[keys].append(r)
    
    # stack by bucket 
    aggregated = {}
    for keys, records in buckets.items():
        records.sort(key = lambda x: x['cutoff'])
        stacked = defaultdict(list)  
        for r in records: 
            for sub_key, arr in r['model_predictions'].items(): # loop over payload outputs
                stacked[sub_key].append(arr)
        aggregated[keys] = {k: np.array(v) for k, v in stacked.items()}

    return aggregated 

if __name__ == '__main__': 
    # pull data 
    df_prices, df_returns = kaggle_price_return_data('rprkh15/sp500-stock-prices')
    # features and labels
    rolling_vol = rolling_volatility(df_prices)    
    target_returns = target_returns_label(df_returns) 
    # to tensor
    x1 = torch.tensor(df_returns.to_numpy(), dtype = torch.float32)
    x2 = torch.tensor(rolling_vol, dtype = torch.float32)
    df_label = torch.tensor(target_returns, dtype = torch.float32)
    # wfa congifs 
    configs = walk_forward_analysis(x1, x2, df_label)
    # GPU multiprocessing 
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=10) as pool:
        results = pool.map(backtest, configs)

    # outputs 
    aggregated = aggregate_results(results)
    
    # save to disk 
    with open('multi-processing/wfa.pkl', mode = 'wb') as f: 
        pickle.dump(obj = aggregated, file = f)

