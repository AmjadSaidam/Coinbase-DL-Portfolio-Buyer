import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from trading_logic.weight_constraint import min_rebalance
from models.device import get_device
from tqdm import tqdm
import models.training_progress_function as tpf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

# =============================================
# Data Pre-Processing
# =============================================
def pad(returns: np.ndarray, lookback: int) -> np.ndarray:
    """
    zero left pad output function to get equal size vectot as input after laged by lookback 
    """
    return np.pad(returns, constant_values = 0, pad_width = (lookback, 0))

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
 
def tensor_standardise(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    z-score row normalisation (does not brake computation graph)
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype = torch.float32)
    mean = torch.mean(x, dim = axis, keepdim = True)
    std = torch.std(x, dim = axis, keepdim = True)
    return (x - mean) / std

# (NOT USED)f
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
def train_test_split_time_series(data, train_size = 0.7):
    """ Train Test Split Function:
    """
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    train, test = train_test_split(data, train_size = train_size, shuffle = False)

    return train, test

# feature preperation, here we get our batches 
def prepare_features(asset_returns: np.ndarray, 
                     asset_prices: np.ndarray, 
                     target_returns: np.ndarray,
                     lookback: int) -> tuple[torch.Tensor]:
    """ 
    split data using rolling lookback, must be called after train/eval/test splits to avoid lookahead bias \\ 
    Eg
    X = [t1, t2, t3, t4, t5] \\
    a = [t1, t2, t3] \\ 
    b = [t2, t3, t4] \\
    if I split, making a = rain and b = test, then I introduce lookahead bias 
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
        lag_p = tensor_standardise(subset_pt) # will get n-timesteps - lookback + 1 number of batches
        lag_rt = tensor_standardise(subset_rt)
        
        # labels 
        features = torch.concat([lag_p, lag_rt], axis=1) # each batch is of size n_assets*2, returns are second set of columns

        # append to feature/label arrays 
        X.append(features)        
        Y.append(target_returns[t]) # next day target return 
        x_last.append(asset_returns[t]) # get current day returns (r_pt = w_t-1^T * r_t)
        inv_X.append(subset_rt)

    x_array = np.array(X)
    y_array = np.array(Y) 
    x_last_ary = np.array(x_last)
    inv_ary = np.array(inv_X)
    
    return torch.tensor(x_array, dtype = torch.float32), \
        torch.tensor(y_array, dtype = torch.float32), \
        torch.tensor(x_last_ary, dtype = torch.float32), \
        torch.tensor(inv_ary, dtype = torch.float32)

def data_pre_process(returns: np.ndarray, prices: np.ndarray, labels: np.ndarray, lookback: int, batches: int = 64) -> data.DataLoader:
    """
    """
    x, y, x_last, x_inv = prepare_features(returns, prices, labels, lookback)
    return data.DataLoader(data.TensorDataset(x, y, x_last, x_inv), shuffle = False, batch_size = batches, drop_last = False)
    
# =============================================
# Volatility Scaling 
# =============================================
def vol_scale(a: torch.Tensor, target_vol: float, vol_lookback) -> torch.Tensor: 
    """
    exponential weighted moving average of standard deviation of asset returns 
    returns matrix same size as input 
    """
    g, h, k = a.shape # batch, sequence_length, n_assets

    if vol_lookback > h-1:
        vol_lookback = h-1 # so dimensions always match
    
    batch_outputs = []
    eps = 1e-8

    alpha = 2 / (vol_lookback + 1) # half life
    for batch in range(g):
        time_outputs = []
        prev_scale = torch.ones(k, device = a.device, dtype = a.dtype) # (n_assets, )
        time_outputs.append(prev_scale) 
        for t in range(1, h): 
            if t < vol_lookback:
                current_scale = prev_scale # vol_t is 1 if t < lookback
            else:
                subset = a[batch, t - vol_lookback: t, :] # (1, lookback_t, n_assets)
                exenate_vol = torch.std(subset, dim = 0).clamp_min(eps) # (1, lookback_t)
                vol_t = target_vol / exenate_vol # levergae
                current_scale = alpha*vol_t + (1 - alpha)*prev_scale # ema 
            time_outputs.append(current_scale)
            prev_scale = current_scale
        batch_outputs.append(torch.stack(time_outputs, dim = 0))

    return torch.stack(batch_outputs, dim = 0)
    
# =============================================
# Loss Functions
# =============================================
def wasserstein_distance(X: torch.Tensor, Y: torch.Tensor, p: int = 1, moment_penalty: float = 0.1) -> torch.Tensor:
    """
    compute difference of distributions in geometry of space, w1 OT solution plus moment regulerisation 
    """ 
    # wp loss
    X_sort = torch.sort(X, dim = 0)[0]
    Y_sort = torch.sort(Y, dim = 0)[0]
    wp = torch.mean(torch.abs(X_sort - Y_sort).pow(p)).pow(1.0/p)
    # moments reguleriser
    moment_loss = 0
    if X.shape[0] > 1 and Y.shape[0] > 1:
        mu_X, mu_Y = torch.mean(X, dim = 0), torch.mean(Y, dim = 0)
        std_X, std_Y = torch.std(X, dim = 0), torch.std(Y, dim = 0)
        moment_loss = (mu_X - mu_Y).pow(2) + (std_X - std_Y).pow(2)

    return wp + moment_penalty*moment_loss

def mmd(X: torch.Tensor, Y: torch.Tensor, sigma: float = 0.05):
    """
    (un)biased empirical mmd 
    """
    n, m = X.shape[0], Y.shape[0]
    if X.ndim == 1:
        X = X.unsqueeze(1)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    # diff matrices 
    xx = X.unsqueeze(1) - X.unsqueeze(0)
    yy = Y.unsqueeze(1) - Y.unsqueeze(0)
    xy = X.unsqueeze(1) - Y.unsqueeze(0)
    # gram matrices 
    rbf = lambda x: torch.exp(-x**2/(2*sigma**2))
    kxx = rbf(xx)
    kyy = rbf(yy)
    kxy = rbf(xy)
    # fill unbiased terms 
    mask = torch.eye(n, dtype = bool)
    kxx = kxx.masked_fill(mask, 0.0)
    kyy = kyy.masked_fill(mask, 0.0)
    # similar terms 
    sim_x = kxx.sum()
    sim_y = kyy.sum()
    # cross terms 
    cross_xy = kxy.sum()

    return 1/(n**2) * (sim_x + sim_y) - 2/(n**2) * cross_xy
    
def sharpe_ratio_loss(Y: torch.Tensor):
    """
    sharpe ratio loss function for dls model 
    """
    exp_rt = torch.mean(Y, dim = 0)
    std_rt = torch.std(Y, dim = 0)

    return -exp_rt / std_rt

# =============================================
# Lstm model
# =============================================
class lstm_model(nn.Module):
    """
    instentiate lstm model, with predict method 
    """
    def __init__(self, input_dim: int, output_dim: int, shorts: bool = False, hidden_dim=100, num_layers=1):
        super().__init__() # super init inherts nn.Module (parent) class methods and attributes
        self.shorts = shorts 

        # LSTM layer for sequential processing 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim) # linear layer to map LSTM output to weights
        self.output = nn.Tanh() if self.shorts else nn.Softmax(dim = -1) # apply last activation to inforce long-only sum to 1 constraint

    def forward(self, x: torch.Tensor, use_weight_constraint: bool = False, weight_constraint: float = None) -> torch.Tensor:
        """
        Forwards Pass:

        Input: dim(x) = (batch_size, sequence_length, i = n_assets * n_features)
        Output: dim(x) = (batch_size, sequence_length, output_size = n_assets)
        """
        # (batch, sequence, hidden) -> (batch, sequence, n_assets)
        output, (hn, cn) = self.lstm(x)
        # attention 
        hidden = output[:, -1, :].unsqueeze(2) # adds t dimension
        scores = torch.bmm(output, hidden)
        alpha = torch.softmax(scores, dim = 1)
        context = (alpha * output).sum(dim = 1)
        # to weights
        output = self.linear(context) # (batch, n_assets)
        final_output = self.output(output) # bound weights 

        # weight constraint
        if self.shorts:
            final_output = final_output / torch.sum(torch.abs(final_output), dim = 1, keepdim = True) # normalise by row sum
        else: 
            if use_weight_constraint: 
                final_output = torch.stack([min_rebalance(weights = row, minimum_value = weight_constraint) for row in final_output], dim = 0)
        
        return final_output # last prediction in batch at t=lookback, this will yeild output dim = n_assets
    
# lstm pipeline 
# =============================================
class lstm():
    """
    instentiate, train and predict using lstm model 
    """
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 shorts: bool = False,
                 hidden_dim = 64, 
                 num_layers = 1, 
                 vol_scaling: bool = False,
                 volatility_lookback: int = None,
                 weight_constraint: bool = False, 
                 min_weight: float = 0.0, 
                 sharpe_loss: bool = False): 
        self.device = get_device()
        self.model: lstm_model = lstm_model(input_dim, output_dim, shorts, hidden_dim, num_layers).to(self.device)
        self.w_constraint = weight_constraint
        self.w_min = min_weight
        self.vol_scaling = vol_scaling
        self.vol_scale_lkb = volatility_lookback
        self.sharpe_loss = sharpe_loss
        self.vol_trg = 0.0
        self.p = 1
        self.tr_loss = 0.0
        self.tr_loss_container = []
        self.eval_loss = 0.0
        self.eval_loss_container = []
        self.opt_res = {}
        self.cost = 0
        
    def __portfolio_returns(self, port_returns, port_weight):
        prev_w_p = torch.roll(port_weight, shifts = 1, dims = 0) 
        prev_w_p[0] = 0
        return torch.sum(port_weight * port_returns, dim = 1) - self.cost*torch.sum(torch.abs(port_weight - prev_w_p), dim = 1)
    
    def lstm_train(self, 
                   train_loader: data.DataLoader,
                   eval_loader: data.DataLoader, 
                   learning_rate: float = 0.01, 
                   n_epochs: int = 100, 
                   show_progress: bool = False) -> lstm_model: 
        """
        """
        best_eval_loss = float('inf')
        best_model_params = None

        self.model.train() # set model in training mode
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate) # instentiate model and optimiser
        
        # run backprop
        for epoch in tqdm(range(n_epochs)):
            total_p = 0
            epoch_loss = 0
            self.tr_loss = 0
            for (x_tr, y_tr, rt, x_inv) in train_loader:
                # send to devices 
                x_tr = x_tr.to(self.device)
                y_tr = y_tr.to(self.device)
                rt = rt.to(self.device)
                x_inv = x_inv.to(self.device)
                # forward pass
                w_p = self.model(x_tr, self.w_constraint, self.w_min) # (batch, output_dim)
                if self.vol_scaling: 
                    vol_scaler = vol_scale(x_inv, self.vol_trg, self.vol_scale_lkb)[:, -1, :] # (batch, lookback, features)
                    w_p *= vol_scaler # (batch, features) = recent exenate volatility estimate
                
                # Compute portfolio weighted returns 
                rt_p = self.__portfolio_returns(rt, w_p)

                # Compute loss
                if self.sharpe_loss:
                    loss = sharpe_ratio_loss(rt_p)
                else:
                    loss = wasserstein_distance(rt_p, y_tr, self.p)

                # backward pass
                optimizer.zero_grad() # clear gradients
                loss.backward()
                optimizer.step()

                # prediction and prediction loss 
                self.tr_loss += loss.item()
                total_p += 1

            # epoch loss
            epoch_loss = self.tr_loss / total_p
            self.tr_loss_container.append(epoch_loss)

            # evaluate model
            if eval_loader is not None: 
                y_eval = self.lstm_evaluate(eval_loader)
                self.eval_loss_container.append(self.eval_loss)
                if self.eval_loss < best_eval_loss: 
                    best_eval_loss = self.eval_loss
                    best_model_params = copy.deepcopy(self.model.state_dict())
                    self.opt_res = y_eval
            
            # append best model to model attribute
            if best_model_params is not None: 
                self.model.load_state_dict(best_model_params)

            # print progress per epoch
            tpf.train_progress(epoch, n_epochs, loss, show_progress)   

        # final loss
        self.tr_loss /= total_p

    def lstm_evaluate(self, eval_loader: data.DataLoader) -> dict:
        """
        """
        all_rt_p = [] # pad vectors to account for lookback 
        all_w_p = []
        all_vol_scale = []

        self.eval_loss = 0
        total_p = 0
        scaled_pos = 0
        
        self.model.eval() # model in evaluation model 
        with torch.no_grad(): # disable gradient tracking 
            for (x_eval, y_eval, rt, x_inv) in eval_loader:
                # send to device 
                x_eval = x_eval.to(self.device)
                y_eval = y_eval.to(self.device)
                rt = rt.to(self.device)
                x_inv = x_inv.to(self.device)                
                # prediction
                w_p = self.model(x_eval, self.w_constraint, self.w_min) # weights 
                if self.vol_scaling: 
                    vol_scaler = vol_scale(x_inv, self.vol_trg, self.vol_scale_lkb)[:, -1, :]
                    w_p *= vol_scaler
                    scaled_pos = torch.sum(w_p * vol_scaler, dim = 1) # inner product of weight and volatility vector
                
                # predicted returns
                rt_p = self.__portfolio_returns(rt, w_p)

                # loss
                if self.sharpe_loss:
                    loss = sharpe_ratio_loss(rt_p)
                else:
                    loss = wasserstein_distance(rt_p, y_eval, self.p)

                # log stats 
                self.eval_loss += loss.item()
                total_p += 1
                
                # logs 
                all_rt_p.append(rt_p)
                all_w_p.append(w_p)
                all_vol_scale.append(scaled_pos)

        # final loss
        self.eval_loss /= total_p

        all_rt_p = torch.cat(all_rt_p, dim = 0).detach().cpu().numpy() # returns portfolio 
        all_w_p = torch.cat(all_w_p, dim = 0).detach().cpu().numpy()
        all_vol_scale = torch.cat(all_vol_scale, dim = 0).detach().cpu().numpy() if self.vol_scaling else []

        return {'weights': all_w_p, 
                'returns': all_rt_p, 
                'vol_scale': all_vol_scale} 