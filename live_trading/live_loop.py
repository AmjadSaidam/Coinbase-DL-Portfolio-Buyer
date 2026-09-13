"""
live script executbale file, includes full model pipeline

run shell command (adds cwd to sys.path[0]): uv run -m live_trading.live_loop
"""
import os
import logging
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime
import numpy as np
import time
import json
import torch
# .py
from models.lstm_trading import lstm, vol_scale
import models.data_preprocessing as data_prep
from walk_forward_analysis.run_wfa import regime_gate, select_regime_windows
from live_trading.coinbase_order_functions import CoinbaseTrader
import live_trading.live_errors as coin_error
import live_trading.live_logging as live_logging
from telegram_bot import shut_down_state_file_name, load_json
from data_loaders import data_to_sql

# --- PATHS ---
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'logs')
LOGS_LIVE = os.path.join(LOGS, 'live_state.json')
LOGS_MODEL = os.path.join(LOGS, 'live_model.pt')

# --- LIVE LOOP ---
def run_live_loop(asset_universe: list[str],
                  benchmark: str = 'BTC-GBP'):
    """live signal to orders function"""
    # exchange env vars
    api_key                  = os.environ.get('LIVE_API_KEY')
    api_secret               = os.environ.get('LIVE_API_SECRET')
    # wfa env vars
    is_fold_len              = int(os.environ.get('LIVE_WFA_IS_BARS'))
    oos_fold_len             = int(is_fold_len // 2) # OOS is half len of IS
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

    # write changes
    def _write_state():
        os.makedirs(os.path.dirname(LOGS_LIVE), exist_ok = True)
        torch.save(is_optimal_model.model.to('cpu').state_dict(), LOGS_MODEL) # model weights are not JSON serialisable, saved separately
        _save_state({
            'time_last_wfa': time_last_wfa.isoformat() if (time_last_wfa is not None) else None,
            'is_optimal_corr_window': is_optimal_corr_window,
            'is_optimal_exp_window': is_optimal_exp_window,
            'is_target_volatility': is_target_volatility
        })

    # instentiate database files
    database = 'live_trades'
    pred_db = _sql_database(database_name = database, name_data = 'predicted_weights', asset_universe = asset_universe)
    obv_db = _sql_database(database_name = database, name_data = 'observed_weights', asset_universe = asset_universe) # actual invested amount

    # call coinbase account manager
    coin = CoinbaseTrader(api_key, api_secret)
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
    live_trading.info(f"successfully connected to coinbase-advanced, API link open and read for GET/POST")

    while True:
        try:
            # time vars
            current_time = datetime.now()
            # wfa
            run_wfa = time_last_wfa is None or (current_time - time_last_wfa).days >= 1
            if run_wfa:
                # data
                data = coin.coinbase_data(asset_universe, num_bars = is_fold_len)
                df_pr, df_rt = data['prices'], data['returns']
                # features
                x_price = torch.tensor(df_pr.to_numpy(), dtype = torch.float32)
                x_return = torch.tensor(df_rt.to_numpy(), dtype = torch.float32)
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
                model.vol_trg = target_vol
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
                                                     bench_idx = df_pr.columns.get_loc(benchmark))
                is_optimal_corr_window = best_windows['window_corr']
                is_optimal_exp_window = best_windows['window_exp']
                # write changes
                time_last_wfa = current_time
                _write_state()
                # logs
                live_trading.info('successfully trained model on most recent in-sample fold: {}-{}'.format(coin.start_date, coin.end_date))

            # check shut down state
            strategy_off = load_json(file = shut_down_state_file_name)['status']
            if strategy_off:
                # log INFO to live_trading.log
                live_trading.info('strategy shut-down via telegram, ressume live polling via telegram bot')
                time.sleep(1)
                continue

            # live logic
            # data
            oos_num_bars = max(is_optimal_corr_window, is_optimal_exp_window, lstm_lookback + 1)
            oos_data = coin.coinbase_data(products = asset_universe,
                                          num_bars = oos_num_bars)
            oos_closed_pr_data = oos_data['prices'].iloc[: -1, :]
            oos_closed_rt_data = oos_data['returns'].iloc[: -1, :]
            # data length minimum bound equal to lookback
            closed_pr_data = oos_closed_pr_data.iloc[-lstm_lookback: , :]
            closed_rt_data = oos_closed_rt_data.iloc[-lstm_lookback: , :]

            # oos prediction (inference runs on cpu - single-sample cost is negligible,
            # and this keeps the model's device in sync with the plain cpu feature tensors below)
            is_optimal_model.model.to('cpu').eval()
            closed_pr_t = torch.tensor(closed_pr_data.to_numpy(), dtype = torch.float32)
            closed_rt_t = torch.tensor(closed_rt_data.to_numpy(), dtype = torch.float32)
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
            regime_flag = regime_gate(asset_returns = oos_closed_rt_data.to_numpy(),
                                      bench_idx = oos_closed_rt_data.columns.get_loc(benchmark),
                                      window_corr = is_optimal_corr_window,
                                      window_exp = is_optimal_exp_window)[-1]

            # turnover flag
            w_prev_adj = coin.get_real_weights(asset_universe)
            turnover = w_pred_adj - np.array(w_prev_adj)
            if np.sum(np.abs(turnover)) < port_min_turnover:
                # log INFO to trades_live.log
                time.sleep(1)
                continue

            # signal to exchange
            if regime_flag:
                asset_weight_dict = coin.tickers_weight(asset_universe, w_pred_adj) # each asset new portfolio weight
                order_payload = coin.multi_asset_invest(portfolio_ticker_weights = asset_weight_dict) # signal to exchange
                # log to loggers 
                live_trading.info('successfully rebalanced portfolio')
                live_logging.trades_logger('trades', 'portfolio rebalance', model_port_weights = w_pred_adj, turnover = order_payload['weight_diffs'], account_balance = order_payload['account_balance'])

            # log all weights (flagged/unflagged) to database dataframes 
            pred_db.list_to_data(weights = w_pred_adj)
            obv_db.list_to_data(weights = w_prev_adj)

        # error/logging
        # error strings written from /coinbase_order_functions
        except coin_error.CoinLoginError as e:
            live_trading.error(f'login error: {e}')

        except coin_error.CoinDataError as e:
            live_trading.error(f'failed exchange data request: {e}')

        except coin_error.CoinOrderError as e:
            live_trading.error(f'failed exchange order post: {e}')

        except Exception as e:
            live_trading.critical(f'untracked expected error: {e}')

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
    pass