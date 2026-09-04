"""
temrinal run command: python -m walk_forward_analysis.run_wfa
"""
from collections import defaultdict
from pathlib import Path
import torch
import multiprocessing as mp
import pickle
# .py
from data_loaders.coinbase_data_post_process import coinbase_price_return_data
from models import data_preprocessing as data_prep
import models.lstm_trading as lstm


def model_set_attributes(model, params: dict):
    """
    set model attributes for class
    """
    [setattr(model, k, v) for (k, v) in params.items()]
    pass


def lstm_pipeline(dim, 
                  train_loader, 
                  eval_loader, 
                  test_loader, 
                  target_vol: float = 0.1, 
                  vol_scale_lkb: int | None = None, 
                  loss_sharpe = True): 
    """lstm train/eval/predict pipeline""" 
    model_l = lstm.lstm(dim * 2, 
                        dim, 
                        hidden_dim = 128, 
                        sharpe_loss = loss_sharpe, 
                        volatility_lookback = vol_scale_lkb)

    params = {
        'vol_trg': target_vol, 
    }
    if vol_scale_lkb: # true for any non-zero number
        model_set_attributes(model_l, params)

    # train, evaluate
    epochs = 100
    model_l.lstm_train(train_loader, eval_loader, n_epochs = epochs)

    # predict 
    res_model_l = model_l.lstm_evaluate(test_loader)

    return {
        'model': model_l.model.to('cpu').state_dict(), 
        'model_train_loss_container': model_l.tr_loss_container, 
        'model_eval_loss_container': model_l.eval_loss_container, 
        'res': res_model_l, 
    }

def walk_forward_analysis(return_data, 
                          price_data,
                          lookback: int = 21, 
                          split_len = 0.05) -> dict[str, dict[str: list]]: 
    """"""
    backtest_configs = []

    n = return_data.shape[0]
    split_len = int(split_len)
    oos_len = int(split_len / 2) # wfa test set is half the in-sample window

    for cutoff in range(split_len, n - oos_len, oos_len): # step forward by the oos window for non-overlapping folds
        # wfa in-sample features 
        is_returns = return_data[cutoff - split_len: cutoff, :]
        is_prices = price_data[cutoff - split_len: cutoff, :] 
        # wfa out-of-sample features
        oos_returns = return_data[cutoff: cutoff + oos_len, :]
        oos_prices = price_data[cutoff: cutoff + oos_len, :]

        # backtest
        backtest_configs.append(
            {
                'wfa_in_sample_returns': is_returns, 
                'wfa_in_sample_prices': is_prices, 
                'wfa_out_of_sample_returns': oos_returns, 
                'wfa_out_of_sample_prices': oos_prices,
                'data_lookback': lookback, 
                'tr_split': 0.9, # paper states test set 10% split
                'cutoff': cutoff # save for data spliting
            }
        )
    return backtest_configs


def backtest(config: dict[str]):
    """using prior features (returns/prices) runs single train/eval/test"""
    # data
    is_returns = config['wfa_in_sample_returns']
    is_prices = config['wfa_in_sample_prices']
    oos_returns = config['wfa_out_of_sample_returns']
    oos_prices = config['wfa_out_of_sample_prices']
    
    # train defaults
    tr_split = config['tr_split']

    # train and eval sets from in sample data 
    tr_returns, eval_returns = data_prep.train_test_split_time_series(is_returns, tr_split)
    tr_prices, eval_prices = data_prep.train_test_split_time_series(is_prices, tr_split)

    # build loaders for wfa data
    train_set = {'returns': tr_returns, 'prices': tr_prices}
    eval_set = {'returns': eval_returns, 'prices': eval_prices}
    test_set = {'returns': oos_returns, 'prices': oos_prices}
    loaders = data_prep.train_eval_test_loaders({
        'train': train_set, 
        'eval': eval_set, 
        'test': test_set, 
    })

    # prediction
    model_pipe = lstm_pipeline(dim = is_returns.shape[1], 
                               train_loader = loaders['train_loader'], 
                               eval_loader = loaders['eval_loader'], 
                               test_loader = loaders['test_loader'])
    
    return {
        'model': model_pipe['model'],
        'model_predictions': model_pipe['res']
    }


def aggregate_results(wfa_results: list[dict]): 
    """groups aggregated backtest data"""    
    # stack by bucket 
    stacked = defaultdict(list) # wights/returns/vol_scale 
    for r in wfa_results: 
        # model
        stacked['model'].append(r['model'])
        stacked['model_train_loss_container'].append(r['model_train_loss_container'])
        stacked['model_eval_loss_container'].append(r['model_eval_loss_container'])
        # model output
        for sub_key, arr in r['model_predictions'].items(): # loop over payload outputs
            stacked[sub_key].append(arr) # for each stack, cutoff will contain weights/returns/vol_scale
    
    return stacked


if __name__ == '__main__':
    # pull data
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    df_prices, df_returns = coinbase_price_return_data(data_dir)

    # to tensor
    x1 = torch.tensor(df_returns.to_numpy(), dtype = torch.float32)
    x2 = torch.tensor(df_prices.to_numpy(), dtype = torch.float32)

    # wfa congifs 
    dls_in_sample_split = 60 / 5 * 24 * 50 # number of 5min brs in 50 days 
    configs = walk_forward_analysis(x1, x2, split_len = dls_in_sample_split)

    # GPU multiprocessing 
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes = 10) as pool:
        wfa_results = pool.map(backtest, configs)

    # outputs 
    aggregated = aggregate_results(wfa_results)
    
    # save to disk
    with open(Path(__file__).resolve().parent / 'wfa.pkl', mode = 'wb') as f:
        pickle.dump(obj = aggregated, file = f)