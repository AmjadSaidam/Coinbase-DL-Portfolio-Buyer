"""
live script executbale file, includes full model pipeline

run shell command (adds cwd to sys.path[0]): uv run -m live_trading.live_loop
"""
import os
import logging
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime
import numpy as np
import pandas as pd
import time
import json
import pickle
import torch
# .py
from models.lstm_trading import lstm, vol_scale
import models.data_preprocessing as data_prep
from walk_forward_analysis.run_wfa import regime_gate, select_regime_windows
from live_trading.coinbase_order_functions import CoinbaseTrader
import live_trading.live_errors as coin_error
import live_trading.live_logging as live_logging
from telegram_bot import shut_down_state_file_name, load_json
from data_loaders import data_to_sql, utils

# --- PATHS ---
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'logs')
LOGS_LIVE = os.path.join(LOGS, 'live_state.json')
LOGS_DATA = os.path.join(LOGS, 'data_cache')
LOGS_MODEL = os.path.join(LOGS, 'live_model.pt')

# --- LIVE LOOP ---
def run_live_loop(asset_universe: list[str],
                  benchmark: str = 'BTC-GBP'):
    """live signal to orders function"""
    # exchange env vars
    api_key                  = os.environ.get('LIVE_API_KEY')
    api_secret               = os.environ.get('LIVE_API_SECRET')
    # wfa env vars
    timeframe                = os.environ.get('LIVE_TIMEFRAME') 
    time_incrument           = int(utils.granularity_seconds[timeframe] / 60) # time frame increment in minutes
    is_fold_days             = int(os.environ.get('LIVE_WFA_IS_DAYS')) # IS len in days 
    oos_time_days            = int(is_fold_days // 2) # OOS len in days
    is_fold_bars             = int(60 / time_incrument * 24 * is_fold_days) # IS len in timeframe bars 
    # lstm env vars
    lstm_hidden_dim          = int(os.environ.get('LIVE_LSTM_HIDDEN_DIM'))
    lstm_lookback            = int(os.environ.get('LIVE_LSTM_LOOKBACK'))
    lstm_vol_scale_lookback  = int(os.environ.get('LIVE_LSTM_VOL_SCALE_WINDOW'))
    lstm_batch               = int(os.environ.get('LIVE_LSTM_BATCH_SIZE'))
    lstm_train_frac          = float(os.environ.get('LIVE_LSTM_TRAIN_SPLIT'))
    lstm_pct_volume_fees     = float(os.environ.get('LIVE_LSTM_TRAINING_FEES'))
    lstm_epochs              = int(os.environ.get('LIVE_LSTM_EPOCHS'))
    # portfolio env vars
    port_min_turnover        = float(os.environ.get('LIVE_PORTFOLIO_MIN_TUROVER'))
    # post-hoc regime env vars
    bench_window_days        = int(os.environ.get('LIVE_BENCHMARK_WINDOW_DAYS')) # benchmark sma len in days
    bench_sma_bars           = int(60 / time_incrument * 24 * bench_window_days) # benchmark sma len in timeframe bars, mirrors notebooks/backtest_dls.ipynb sma_window

    # data cache
    data_cache = None

    # other model vars
    dim = len(asset_universe)

    # instentiate logging 
    live_trading: logging = live_logging.logger_setup(LOGS, 'live_trading', formatting = True)
    trades: logging = live_logging.logger_setup(LOGS, 'trades')

    # default bookkeeping variables
    time_last_wfa = None
    is_optimal_model = lstm(input_dim = dim * 2,
                            output_dim = dim,
                            hidden_dim = lstm_hidden_dim,
                            volatility_lookback = lstm_vol_scale_lookback,
                            cost = lstm_pct_volume_fees) # placeholder, restored from disk below or trained on first wfa pass
    is_optimal_corr_window = None
    is_optimal_exp_window = None
    is_target_volatility = None

    # load states
    # run once on restart/crash
    saved_state: dict = _load_state()
    if saved_state is not None:
        time_last_wfa = datetime.fromisoformat(saved_state['time_last_wfa']) if saved_state.get('time_last_wfa') else None # read last wfa time
        is_optimal_corr_window = saved_state.get('is_optimal_corr_window', None)
        is_optimal_exp_window = saved_state.get('is_optimal_exp_window', None)
        is_target_volatility = saved_state.get('is_target_volatility', None)
        if os.path.exists(LOGS_MODEL): # read optimal model weights
            is_optimal_model.model.load_state_dict(torch.load(LOGS_MODEL, map_location = 'cpu'))
        # data cache (pickled, not JSON - a DataFrame with a DatetimeIndex doesn't round-trip through json)
        if os.path.exists(LOGS_DATA):
            with open(LOGS_DATA, mode = 'rb') as f:
                data_cache = pickle.load(f)

    # write changes
    def _write_state():
        os.makedirs(os.path.dirname(LOGS_LIVE), exist_ok = True)
        # save model
        torch.save(is_optimal_model.model.to('cpu').state_dict(), LOGS_MODEL) # model weights are not JSON serialisable, saved separately
        # save state dicts
        _save_state({
            'time_last_wfa': time_last_wfa.isoformat() if (time_last_wfa is not None) else None,
            'is_optimal_corr_window': is_optimal_corr_window,
            'is_optimal_exp_window': is_optimal_exp_window,
            'is_target_volatility': is_target_volatility
        })
        # save data cache
        with open(LOGS_DATA, mode = 'wb') as f:
            pickle.dump(data_cache, f)

    # instentiate database files
    database = 'live_trading_database' # will create in root folder
    pred_db = _sql_database(database_name = database, name_data = 'predicted_weights', asset_universe = asset_universe)
    act_db = _sql_database(database_name = database, name_data = 'actual_weights', asset_universe = asset_universe) # actual invested amount

    # call coinbase account manager
    coin = CoinbaseTrader(api_key, api_secret) # reads api key from .env
    # attempt account connection
    while True:
        try:
            coin.login()
            break
        except coin_error.CoinLoginError as e:
            # log ERROR type to live_trades.log
            live_trading.error(f"failed to connect to coinbase-advanced with error {e}: re-trying in 1s")
            _attempt_reconnect(coin)
    # log INFO type to live_trades.log
    live_trading.info(f"successfully connected to coinbase-advanced, API link established and ready for GET/POST querying")

    # log defaults 
    logged_regime_flag = False 

    # per 5min check 
    latest_bar_time = None

    while True:
        try:
            # time vars
            current_time = datetime.now()
            # wfa
            fold_params_missing = None in (is_optimal_corr_window, is_optimal_exp_window, is_target_volatility) # e.g. a state file from before these were persisted
            run_wfa = time_last_wfa is None or fold_params_missing or (current_time - time_last_wfa).days >= oos_time_days
            if run_wfa:
                # inintial full IS API request on start up, otherwise read from data dict which is persisted per bar
                if data_cache is None:
                    data = coin.coinbase_data(asset_universe,
                                              num_bars = is_fold_bars + 1,
                                              granularity = timeframe)
                    # closed data
                    df_pr, df_rt = _closed_bars(data, is_fold_bars)
                    data_cache = {
                        'prices': df_pr,
                        'returns': df_rt
                    }
                # features - data_cache is rolled forward one bar at a time by the live-logic
                # block below, so after the initial load here no separate fetch is needed
                x_price = torch.tensor(data_cache['prices'].to_numpy(), dtype = torch.float32)
                x_return = torch.tensor(data_cache['returns'].to_numpy(), dtype = torch.float32)
                # wfa in-sample fold
                is_train_pr, is_eval_pr = data_prep.train_test_split_time_series(x_price, lstm_train_frac)
                is_train_rt, is_eval_rt = data_prep.train_test_split_time_series(x_return, lstm_train_frac)
                loader_keys = ['prices', 'returns', 'lookback', 'mini_batches']
                train_set = dict(zip(loader_keys, [is_train_pr, is_train_rt, lstm_lookback, lstm_batch]))
                eval_set = dict(zip(loader_keys, [is_eval_pr, is_eval_rt, lstm_lookback, lstm_batch]))
                loaders = data_prep.train_eval_test_loaders({
                    'train': train_set,
                    'eval': eval_set
                })
                # model training
                vol_proxy = is_train_rt.std(dim = 0).numpy() # per-asset volatility estimate
                target_vol = np.percentile(vol_proxy, 90) # 90% percentile cross asset volatility estimate
                # model
                model = lstm(input_dim = dim * 2,
                             output_dim = dim,
                             hidden_dim = lstm_hidden_dim,
                             volatility_lookback = lstm_vol_scale_lookback,
                             cost = lstm_pct_volume_fees)
                model.vol_trg = target_vol # set target volatility
                # model training
                model.lstm_train(train_loader = loaders['train_loader'],
                                 eval_loader = loaders['eval_loader'],
                                 n_epochs = lstm_epochs)
                is_optimal_model = model # save optimal model
                is_target_volatility = target_vol
                # evaluation set grid search
                eval_port_rt = model.opt_res['returns']
                eval_asset_rt = np.asarray(is_eval_rt)[lstm_lookback: ][: len(eval_port_rt)]
                best_windows = select_regime_windows(eval_port_returns = eval_port_rt,
                                                     eval_asset_returns = eval_asset_rt,
                                                     bench_idx = data_cache['prices'].columns.get_loc(benchmark))
                is_optimal_corr_window = best_windows['window_corr']
                is_optimal_exp_window = best_windows['window_exp']
                # write changes
                time_last_wfa = current_time
                _write_state()
                # logs
                live_trading.info('successfully trained model on most recent in-sample fold: {}-{}'.format(data_cache['prices'].index[0], data_cache['prices'].index[-1]))
                # skip live iteration this bar
                continue 

            # check shut down state
            strategy_off = load_json(file = shut_down_state_file_name)['status']
            if strategy_off:
                # log INFO to live_trading.log
                live_trading.info('strategy shut-down via telegram, ressume live polling via telegram bot')
                time.sleep(1)
                continue

            # live logic
            # throttle per bar open to reduce api query rate
            current_time_posix = int(current_time.timestamp())
            seconds_since_bar_open = current_time_posix % (time_incrument * 60) # time since bar open in seconds
            bar_open_time = current_time_posix - seconds_since_bar_open # bar open time
            if latest_bar_time == bar_open_time: # check if same bar, true for all but new bar updated (-0) (strategy should only run logic on open of new bar)
                time.sleep(1)
                continue # already polled re-run 
            latest_bar_time = bar_open_time
            # data
            oos_data = coin.coinbase_data(products = asset_universe,
                                          num_bars = 2,
                                          granularity = timeframe)
            # closed data
            df_pr, df_rt = _closed_bars(oos_data, 1)
            # cache data - skip if the fetched bar is already the cache's last bar (eg just ran WFA)
            if data_cache and (df_pr.index[-1] > data_cache['prices'].index[-1]): # cached index must be greater than loaded index to append new bar data
                n_new = len(df_pr)
                data_cache['prices'] = pd.concat([data_cache['prices'].iloc[n_new: ], df_pr])
                data_cache['returns'] = pd.concat([data_cache['returns'].iloc[n_new: ], df_rt])
            # save updates
            _write_state()
            # lstm feature window - the last `lookback` closed bars, prepare_features() subset [t - lookback, t)
            oos_closed_pr_data = data_cache['prices']
            oos_closed_rt_data = data_cache['returns']
            oos_pr_feature_data = oos_closed_pr_data.iloc[-lstm_lookback: , :]
            oos_rt_feature_data = oos_closed_rt_data.iloc[-lstm_lookback: , :]
            # oos prediction (inference runs on cpu - single-sample cost is negligible,
            # and this keeps the model's device in sync with the plain cpu feature tensors below)
            is_optimal_model.model.to('cpu').eval()
            closed_pr_t = torch.tensor(oos_pr_feature_data.to_numpy(), dtype = torch.float32)
            closed_rt_t = torch.tensor(oos_rt_feature_data.to_numpy(), dtype = torch.float32)
            oos_features = torch.concat([
                data_prep.tensor_standardise(closed_pr_t),
                data_prep.tensor_standardise(closed_rt_t)
            ], axis = 1).unsqueeze(0) # (1, lookback, n_assets * 2), mirrors prepare_features() ordering
            with torch.no_grad():
                w_pred = is_optimal_model.model(oos_features, is_optimal_model.w_min).detach().numpy()
            # volatility scaling
            vol_scaler = vol_scale(closed_rt_t.unsqueeze(0), is_target_volatility, lstm_vol_scale_lookback)[:, -1, :].numpy()
            w_pred_adj = (w_pred * vol_scaler).flatten()

            # regimes
            # 1) fold crash gate - in-sample grid-searched windows, evaluated on the last closed bar
            bench_idx = df_pr.columns.get_loc(benchmark)
            regime_flag = bool(regime_gate(asset_returns = oos_closed_rt_data.to_numpy(),
                                           bench_idx = bench_idx,
                                           window_corr = is_optimal_corr_window,
                                           window_exp = is_optimal_exp_window)[-1])
            # 2) benchmark trend gate - last close below its rolling sma (notebook backtest() sma_window);
            #    an sma that has not warmed up (nan) never gates, as in the notebook
            bench_px = oos_closed_pr_data.iloc[:, bench_idx]
            bench_sma = bench_px.rolling(window = bench_sma_bars).mean().iloc[-1]
            sma_flag = bool(bench_px.iloc[-1] < bench_sma) if not np.isnan(bench_sma) else False
            full_mask = regime_flag or sma_flag
            # dont trade case - the target is the all-cash portfolio
            target_w = np.zeros(dim) if full_mask else w_pred_adj

            # turnover deadband vs the actual (drifted) held position, mirrors notebook backtest():
            # skip small rebalances, but a regime-forced liquidation must always execute
            w_prev_adj = np.array(coin.get_real_weights(asset_universe))
            turnover = np.sum(np.abs(target_w - w_prev_adj))
            if not ((turnover >= port_min_turnover) or (full_mask and turnover > 0)):
                time.sleep(1)
                continue

            # signal to exchange
            if not full_mask:
                asset_weight_dict = coin.tickers_weight(asset_universe, w_pred_adj) # each asset new portfolio weight
                order_payload = coin.multi_asset_invest(portfolio_ticker_weights = asset_weight_dict) # signal to exchange
                # log to loggers 
                live_trading.info('successfully rebalanced portfolio')
                live_logging.trades_logger('trades', 'portfolio_rebalance', model_port_weights = w_pred_adj, turnover = order_payload['weight_diffs'], account_balance = order_payload['account_balance'])
                # reset flaged trade on 
                logged_regime_flag = False 
            else: 
                # liquidate portfolio 
                coin.multi_asset_close(asset_universe, full_close = True)
                # logg full close 
                live_trading.info('successfully liquidated portfolio position')
                live_logging.trades_logger('trades', 'portfolio_liquidation', turnover = turnover, regime_flag = regime_flag, sma_flag = sma_flag)
                # only write flaged trades on switch 
                if not logged_regime_flag:
                    live_trading.info(f'regime state True (crash gate: {regime_flag}, benchmark below sma: {sma_flag}), liquidated portfolio to cash')
                    logged_regime_flag = True

            # log all weights (flagged/unflagged) to database dataframes 
            pred_db.list_to_data(weights = w_pred_adj)
            act_db.list_to_data(weights = w_prev_adj)

        # error/logging
        # error strings written from /coinbase_order_functions
        except coin_error.CoinLoginError as e:
            live_trading.error(f'login error: {e}')

        except coin_error.CoinDataError as e:
            live_trading.error(f'failed exchange data request: {e}')

        except coin_error.CoinOrderError as e:
            live_trading.error(f'failed exchange order post: {e}')

        except Exception as e:
            live_trading.critical(f'unkown error: {e}', exc_info = True)

        # poll
        time.sleep(1)

# --- HELPERS ---
def _load_state():
    """loads live state file if exists"""
    if not os.path.exists(LOGS_LIVE):
        return None
    with open(LOGS_LIVE) as f:
        return json.load(f)

def _save_state(state: dict[str]):
    """creates/updates state file with dict"""
    # create directory if does not exists
    os.makedirs(os.path.dirname(LOGS_LIVE), exist_ok = True)
    tmp_path = LOGS_LIVE + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj = state, fp = f, default = float)
    os.replace(tmp_path, LOGS_LIVE)

def _attempt_reconnect(coin: CoinbaseTrader):
    """attempts to throttle and reconnect to account"""
    if not coin.authenticated:
        time.sleep(5)
        coin.login()

def _closed_bars(data: dict,
                 n_bars: int):
    """filters for closed bars only"""
    prices = data['prices'].iloc[1: -1, :]
    returns = data['returns'].iloc[1: -1, :]
    if len(returns) != n_bars: # universe alignment (leading nans) or a short api reply shaved rows - fail loud, never gate on fewer bars
        raise coin_error.CoinDataError(f'expected {n_bars} closed bars, received {len(returns)}')
    return prices, returns

def _sql_database(database_name: str,
                  name_data: str,
                  asset_universe: list[str]):
    """"""
    db = data_to_sql.CreateSQLiteDatabase(tickers = asset_universe, name_database = database_name, name_data = name_data)
    db.create_access_database_file()
    db.create_data()
    return db

# --- RUN PIPELINE ---
if __name__ == '__main__':
    asset_universe = [
        'BTC-GBP', 'ETH-GBP', 'SOL-GBP', 'LINK-GBP', 'USDT-GBP'
    ]
    run_live_loop(asset_universe = asset_universe)