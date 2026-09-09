"""
to run on vast-ai run: 
git clone https://github.com/AmjadSaidam/Coinbase-DL-Portfolio-Buyer.git \Coinbase-DL-Portfolio-Buyer
uv sync
uv run -m walk_forward_analysis.run_wfa
scp vast-gpu:/workspace/Coinbase-DL-Portfolio-Buyer/walk_forward_analysis/wfa.pkl "/Users/amjadsaidam/Desktop/Quant stuff /Algorithmic Strategies /coinbase_trading_bot/walk_forward_analysis"
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import pickle
# .py
from data_loaders.coinbase_data_post_process import coinbase_price_return_data
from models import data_preprocessing as data_prep
import models.lstm_trading as lstm

# regime gate constants - fixed (not grid-searched), mirroring notebooks/backtest_dls.ipynb defaults
REGIME_CORRELATION_THRESHOLD = 0.8
REGIME_FRACTION_NEGATIVE = 0.7


def model_set_attributes(model: lstm, 
                         params: dict):
    """
    set model attributes for class
    """
    [setattr(model, k, v) for (k, v) in params.items()]
    pass


# ---------------------------------------------
# Regime gate (crash risk-off / de-invest signal), mirrors notebooks/backtest_dls.ipynb
# ---------------------------------------------
def rolling_corr(bench_returns: np.ndarray, 
                 asset_returns: np.ndarray, 
                 window: int) -> np.ndarray:
    """rolling correlation between a benchmark and an asset return series"""
    bench_rt = pd.Series(bench_returns)
    asset_rt = pd.Series(asset_returns)
    cov = (bench_rt * asset_rt).rolling(window).mean() - bench_rt.rolling(window).mean() * asset_rt.rolling(window).mean()
    std = bench_rt.rolling(window).std() * asset_rt.rolling(window).std()
    corr = np.where(std > 0, cov / std, 0.0)
    return np.nan_to_num(corr, nan = 0.0)


def rolling_expected_pos_returns(asset_returns: np.ndarray, 
                                 window: int, 
                                 fraction_negative: float) -> np.ndarray:
    """True where the rolling mean return of >= fraction_negative of assets is negative"""
    rolling_mean = pd.DataFrame(asset_returns).rolling(window).mean().fillna(0.0).to_numpy()
    n_neg_required = int(np.ceil(fraction_negative * asset_returns.shape[1]))
    n_assets_negative = (rolling_mean < 0).sum(axis = 1)
    return n_assets_negative >= n_neg_required


def rolling_sharpe(returns: np.ndarray, 
                   window: int) -> np.ndarray:
    """rolling (non-annualised) sharpe ratio - used only as the grid-search objective"""
    s = pd.Series(returns)
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    sharpe = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(sharpe, nan = 0.0)


def regime_gate(asset_returns: np.ndarray,
                bench_idx: int,
                window_corr: int,
                window_exp: int,
                correlation_threshold: float = REGIME_CORRELATION_THRESHOLD,
                fraction_negative: float = REGIME_FRACTION_NEGATIVE) -> np.ndarray:
    """crash = every non-bench asset highly correlated with the bench AND broad negative breadth"""
    n_assets = asset_returns.shape[1]
    bench_returns = asset_returns[:, bench_idx]
    non_bench_idx = [i for i in range(n_assets) if i != bench_idx]

    corr_flags = np.stack([
        rolling_corr(bench_returns, asset_returns[:, i], window_corr) > correlation_threshold
        for i in non_bench_idx
    ], axis = 1)
    crash = corr_flags.sum(axis = 1) == (n_assets - 1) # all pair-wise correlations to base exceed threshold

    breadth = rolling_expected_pos_returns(asset_returns, window_exp, fraction_negative)
    return crash & breadth


def select_regime_windows(eval_port_returns: np.ndarray,
                          eval_asset_returns: np.ndarray,
                          bench_idx: int,
                          score_window: int = 100,
                          window_corr_grid = (100, 150, 200),
                          window_exp_grid = (100, 150, 200)) -> dict:
    """
    small grid search over regime rolling-window lengths, scored by sharpe ratio on
    in-sample eval predictions only - never touches oos data, so the fold this selects
    for stays unbiased.
    """
    best_score = -np.inf
    best = {
        'window_corr': window_corr_grid[0], 
        'window_exp': window_exp_grid[0]
    }
    for window_corr in window_corr_grid:
        for window_exp in window_exp_grid:
            gate = regime_gate(eval_asset_returns, bench_idx, window_corr, window_exp)
            gated_rt = np.where(gate, 0.0, eval_port_returns)
            score = rolling_sharpe(gated_rt, score_window).mean()
            if score > best_score:
                best_score = score
                best = {
                    'window_corr': window_corr, 
                    'window_exp': window_exp
                }
    return best


def lstm_pipeline(dim,
                  train_loader,
                  eval_loader,
                  test_loader,
                  eval_returns: np.ndarray,
                  lookback: int,
                  bench_idx: int,
                  target_vol: float = 0.1,
                  vol_scale_lkb: int | None = 24, # 2 hours volatility estimate
                  loss_sharpe = True,
                  cost: float = 0.0):
    """lstm train/eval/predict pipeline"""
    model_l = lstm.lstm(dim * 2,
                        dim,
                        hidden_dim = 128,
                        sharpe_loss = loss_sharpe,
                        volatility_lookback = vol_scale_lkb,
                        cost = cost)

    params = {
        'vol_trg': target_vol,
    }
    if vol_scale_lkb: # true for any non-zero number
        model_set_attributes(model_l, params)

    # IS
    # train, evaluate
    epochs = 100
    model_l.lstm_train(train_loader, eval_loader, n_epochs = epochs)

    # regime windows selected from the best-epoch in-sample eval predictions only (never oos) -
    # opt_res is already computed as a byproduct of training, so this adds no extra model passes
    eval_port_rt = model_l.opt_res['returns']
    eval_asset_rt = np.asarray(eval_returns)[lookback:][: len(eval_port_rt)]
    best_windows = select_regime_windows(eval_port_rt, eval_asset_rt, bench_idx)

    # OOS
    # predict - fee-free, so equity() can sweep fee tiers post-hoc without re-running the WFA
    res_model_l = model_l.lstm_evaluate(test_loader, apply_cost = False)

    return {
        'model': model_l.model.to('cpu').state_dict(),
        'model_train_loss_container': model_l.tr_loss_container,
        'model_eval_loss_container': model_l.eval_loss_container,
        'res': res_model_l,
        'regime_windows': best_windows,
    }

def walk_forward_analysis(return_data,
                          price_data,
                          bench_idx: int,
                          lookback: int = 288,
                          batch_size: int = 288,
                          split_len = 0.05,
                          cost: float = 0.0) -> dict[str, dict[str: list]]:
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
                'batch_size': batch_size,
                'tr_split': 0.9, # paper states test set 10% split
                'cutoff': cutoff, # save for data spliting
                'cost': cost,
                'bench_idx': bench_idx,
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
    lookback = config['data_lookback']
    batch = config['batch_size']
    cost = config['cost']
    bench_idx = config['bench_idx']

    # train and eval sets from in sample data 
    tr_returns, eval_returns = data_prep.train_test_split_time_series(is_returns, tr_split)
    tr_prices, eval_prices = data_prep.train_test_split_time_series(is_prices, tr_split)

    # data_pre_process() defaults
    train_set = {'returns': tr_returns, 'prices': tr_prices, 'lookback': lookback, 'mini_batches': batch}
    eval_set = {'returns': eval_returns, 'prices': eval_prices, 'lookback': lookback, 'mini_batches': batch}
    test_set = {'returns': oos_returns, 'prices': oos_prices, 'lookback': lookback, 'mini_batches': batch}
    # build loaders 
    loaders = data_prep.train_eval_test_loaders({
        'train': train_set, 
        'eval': eval_set, 
        'test': test_set, 
    })

    # prediction - regime windows are selected inside lstm_pipeline from in-sample eval data only
    model_pipe = lstm_pipeline(dim = is_returns.shape[1],
                               train_loader = loaders['train_loader'],
                               eval_loader = loaders['eval_loader'],
                               test_loader = loaders['test_loader'],
                               eval_returns = eval_returns,
                               lookback = lookback,
                               bench_idx = bench_idx,
                               cost = cost)

    # apply the (already fixed, not re-tuned) in-sample-selected regime windows to oos predictions
    oos_port_rt = model_pipe['res']['returns']
    oos_asset_rt = np.asarray(oos_returns)[lookback:][: len(oos_port_rt)]
    regime_mask = regime_gate(oos_asset_rt, bench_idx,
                              model_pipe['regime_windows']['window_corr'],
                              model_pipe['regime_windows']['window_exp'])
    model_pipe['res']['regime_mask'] = regime_mask # truth array, True when confluence of regimes, False otherwise
    model_pipe['res']['gated_returns'] = np.where(regime_mask, 0.0, oos_port_rt)

    return model_pipe


def aggregate_results(wfa_results: list[dict]): 
    """groups aggregated backtest() data"""    
    # stack by bucket 
    stacked = defaultdict(list) 
    # combine results of model_pipe
    for r in wfa_results: 
        # model
        stacked['model'].append(r['model'])
        stacked['model_train_loss_container'].append(r['model_train_loss_container'])
        stacked['model_eval_loss_container'].append(r['model_eval_loss_container'])
        stacked['regime_windows'].append(r['regime_windows'])
        # model output
        for sub_key, arr in r['res'].items(): # loop over payload outputs
            stacked[sub_key].append(arr) # for each stack, cutoff will contain weights/returns/vol_scale + regime_mask/gated_returns
    
    return stacked


if __name__ == '__main__':
    # pull data
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    df_prices, df_returns = coinbase_price_return_data(data_dir)

    # to tensor
    x1 = torch.tensor(df_returns.to_numpy(), dtype = torch.float32)
    x2 = torch.tensor(df_prices.to_numpy(), dtype = torch.float32)
    bench_idx = df_prices.columns.get_loc('BTC-GBP') # regime gate benchmark column

    # wfa congifs
    dls_in_sample_split = 60 / 5 * 24 * 50 # number of 5min brs in 50 days
    configs = walk_forward_analysis(x1, x2, bench_idx, split_len = dls_in_sample_split, cost = 0.0016) # advanced_4 tier, applied during training only - test predictions stay fee-free

    # GPU multiprocessing 
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes = 2) as pool:
        wfa_results = pool.map(backtest, configs)

    # outputs 
    aggregated = aggregate_results(wfa_results)
    
    # save to disk
    with open(Path(__file__).resolve().parent / 'wfa.pkl', mode = 'wb') as f:
        pickle.dump(obj = aggregated, file = f)