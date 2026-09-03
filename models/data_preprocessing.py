import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================
# Data Pre-Processing
# =============================================
def pad(returns: np.ndarray, lookback: int) -> np.ndarray:
    """
    zero left pad output function to get equal size vectot as input after laged by lookback
    """
    return np.pad(returns, constant_values = 0, pad_width = (lookback, 0))

def tensor_standardise(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    z-score normalisation (does not brake computation graph)
    standerdise per asset (column) or cross section (row)
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype = torch.float32)
    mean = torch.mean(x, dim = axis, keepdim = True)
    std = torch.std(x, dim = axis, keepdim = True)
    return (x - mean) / std

# standerdise inputs
def standerdise(x: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """
    z-score row normlaisation (brakes computation graph)
    """
    res = None
    scaler = StandardScaler()

    if x.ndim == 1:
        x = x.reshape(-1, 1) # single feature data
        res = scaler.fit_transform(x)
    else:
        res = scaler.fit_transform(x)
    inv = scaler.inverse_transform(res)
    return res, inv

# (NOT USED)
def difference(series: np.ndarray):
    """
    make price series stationary process by differencing
    """
    series = pd.DataFrame(series)
    asset_prices_shift = series.shift(1).bfill() # fill na's with last value
    stationary_returns = series - asset_prices_shift

    return np.array(stationary_returns)

# (NOT USED)
def function_to_lstm_batch(x: torch.Tensor, f):
    """
    applies function to LSTM batch
    """
    batches = [f(x[i]) for i in range(x.size(0))]
    return torch.stack(batches, dim = 0)

# gets training and test data from entire dataset
def train_test_split_time_series(data, 
                                 train_size = 0.7):
    """ Train Test Split Function:
    """
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    train, test = train_test_split(data, train_size = train_size, shuffle = False)

    return train, test

# feature preperation, here we get our batches
def prepare_features(asset_returns: np.ndarray,
                     asset_prices: np.ndarray,
                     target_returns: np.ndarray | None,
                     lookback: int) -> tuple[torch.Tensor]:
    """
    split data using rolling lookback, must be called after train/eval/test splits to avoid lookahead bias \\
    Eg
    X = [t1, t2, t3, t4, t5] \\
    a = [t1, t2, t3] \\
    b = [t2, t3, t4] \\
    if I split, making a = train and b = test, then I introduce lookahead bias
    If I split before then stack all splits will be [) and [), so no look ahead bias
    """
    n_timesteps = asset_returns.shape[0]

    # deal with insufficient data
    if n_timesteps < lookback:
        raise ValueError("lookback must be less than timeseries length")

    # feature and target label arrays
    X, Y = [], []
    x_last = []
    inv_X = []

    # lag from past to present
    for t in range(lookback, n_timesteps):
        # construct feature vectors
        subset_rt = asset_returns[t - lookback: t]
        subset_pt = asset_prices[t - lookback: t]
        # standerdise after split to avoid lookahead bias
        lag_rt = tensor_standardise(subset_rt)
        lag_p = tensor_standardise(subset_pt) # will get (n-timesteps - lookback + 1) number of batches

        # labels
        features = torch.concat([lag_p, lag_rt], axis=1) # each batch is of size n_assets*2, returns are second set of columns

        # append to feature/label arrays
        X.append(features)
        x_last.append(asset_returns[t]) # get current day returns (r_pt = w_t-1^T * r_t)
        inv_X.append(subset_rt)
        
        if target_returns is not None:
            Y.append(target_returns[t]) # next day target return

    # features 
    x_array = np.array(X)
    x_last_ary = np.array(x_last)
    inv_ary = np.array(inv_X)
    # labels (zero-filled when no target labels are supplied, so tensor lengths still match)
    y_array = np.array(Y) if target_returns is not None else np.zeros(len(X))

    return torch.tensor(x_array, dtype = torch.float32), \
        torch.tensor(y_array, dtype = torch.float32), \
        torch.tensor(x_last_ary, dtype = torch.float32), \
        torch.tensor(inv_ary, dtype = torch.float32)

def data_pre_process(returns: np.ndarray, 
                     prices: np.ndarray, 
                     lookback: int = 21, 
                     labels: np.ndarray | None = None,
                     mini_batches: int = 64) -> data.DataLoader:
    """builds data loader for lstm"""
    x, y, x_last, x_inv = prepare_features(returns, prices, labels, lookback)
    return data.DataLoader(data.TensorDataset(x, y, x_last, x_inv), shuffle = False, batch_size = mini_batches, drop_last = False)

def train_eval_test_loaders(loaders: dict[str]):
    """returns train, evaluation and test loaders"""
    tr_loader = data_pre_process(**loaders['train'])
    eval_loader = data_pre_process(**loaders['eval'])
    ts_loader = data_pre_process(**loaders['test'])

    return {
        'train_loader': tr_loader, 
        'eval_loader': eval_loader, 
        'test_loader': ts_loader
    }