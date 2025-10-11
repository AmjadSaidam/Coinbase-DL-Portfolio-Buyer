import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import training_progress_function as tpf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================
# Data Pre-Processing
# =============================================
# standerdise inputs 
def difference(x):
    res = None
    if x.ndim == 1:
        x = x.reshape(-1, 1) # single feature data 
        res = StandardScaler().fit_transform(x).squeeze()
    else:
        res = StandardScaler().fit_transform(x)
    return res
 
# standardise function for LSTM
def tensor_standardise(x: torch.Tensor):
    mean = torch.mean(x, dim = 1, keepdim = True)
    std = torch.std(x, dim = 1, keepdim = True)
    return (x - mean) / std

# applies function to LSTM batch
def function_to_lstm_batch(x: torch.Tensor, f):
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

# make price series stationary process by differencing, not used 
def de_correlate_prices(prices):
    prices = pd.DataFrame(prices)
    asset_prices_shift = prices.shift(1).bfill() # fill na's with last value
    stationary_returns = np.log(prices) - np.log(asset_prices_shift)

    return np.array(stationary_returns)

# feature preperation, here we get our batches 
def prepare_features(asset_returns: np.ndarray, 
                     asset_prices: np.ndarray, 
                     target_returns: np.ndarray,
                     lookback: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ Feature and Target Pre-processing for Training:
    """    

    n_timesteps, n_assets = asset_returns.shape

    # deal with insufficient data
    if n_timesteps < lookback:
        raise ValueError("lookbakc must be less than timeseries length")
    if n_timesteps != asset_prices.shape[0]:
        raise ValueError("feature and target data must be of equal length")
    
    # feature and target label arrays
    X, Y = [], [] # list of numpy arrays 

    # Lag features and get lagged fearture set with observations from lookback starting from lookback (first feature entry) up to current index
    # i.e. get lagged variables set, lagged by lookback
    for t in range(lookback, n_timesteps):
        # features, time series properties are invariant to z-transform, standerdise =/ stationary
        past_prices = difference(asset_prices[t - lookback: t]) # will get n-timesteps - lookback number of batches
        past_returns = difference(asset_returns[t - lookback: t])
        
        # labels 
        features = np.concatenate([past_prices, past_returns], axis=1) # each batch is a of size n_assets*2, returns are second set of columns

        # append to feature/label arrays 
        X.append(features)        
        Y.append(target_returns[t-1]) # get current sampeld return. The return is our label and is only used in loss function

    x_array = np.array(X)
    y_array = np.array(Y) 

    return torch.tensor(x_array, dtype = torch.float32), torch.tensor(y_array, dtype = torch.float32) # pass list of arrays transform to tensors, do not use torch.FloatTensor() because slow

# =============================================
# DmLSTM
# =============================================
class DmLSTM(nn.Module):
    """DmLSTM model:
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=100, num_layers=2):
        super().__init__() # super init inherts nn.Module (parent) class methods and attributes
        
        if input_dim != output_dim:
            raise ValueError("input and ouput dimensions must be euqal")

        # LSTM layer for sequential processing 
        self.lstm = nn.LSTM(input_dim*2, hidden_dim, num_layers, batch_first=True, dropout = 0.3)
        self.fc = nn.Linear(hidden_dim, output_dim) # Final linear layer to map LSTM output to weights
        self.output = nn.Softmax(dim=-1) # apply last activation to inforce long-only sum to 1 constraint

    def forward(self, x: torch.Tensor, hidden=None) -> torch.Tensor:
        """Forwards Pass:
        Input: dim(x) = (batch_size, sequence_length, input_size = n_assets * n_features)
        Output: dim(x) = (batch_size, sequence_length, hidden_dimensions)
        """
        x = function_to_lstm_batch(x, tensor_standardise)

        if hidden is None:
            output, hidden = self.lstm(x) 
        else:
            output, hidden = self.lstm(x, hidden)

        # now apply the two activations to transform (batch, sequence, hidden) = (batch, sequence, n_assets)
        # note in this steup we have n batches and in each batch we have a sequence of length looback, at each t in lookback the LSTM will product n_assets weights
        output = self.fc(output)
        output = self.output(output)

        return output[:, -1, :] # last prediction in batch at t=lookback, this will yeild output dim = n_assets
    
# =============================================
# Loss Function 
# =============================================
def wasserstein_distance(X, Y):
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):  
        # get the order statistics
        X_sort = torch.sort(X, dim = 0)[0]
        Y_sort = torch.sort(Y, dim = 0)[0]
    else:
        raise TypeError("inputs must be torch tensors")

    return torch.sum((X_sort - Y_sort)**2)**0.5   # Mean absolute difference

# =============================================
# DmLSTM Backpropagation 
# =============================================
def backprop_rnn(asset_returns, asset_prices, target_returns,
                 lookback: int, batch_size = 64, hidden_dim=100, num_layers=2, n_epochs=100, learning_rate=0.001, 
                 show_progress = False) -> dict:
    """
    Train PortfolioRNN with backpropagation, avoiding information leakage.
    """
    final_weights = None
    portfolio_returns_final = None

    m, n_assets = asset_returns.shape 
    if m < 2:
        raise ValueError("Insufficient time steps for sequential processing.")
    
    # instentiate model and optimiser
    model = DmLSTM(input_dim=n_assets, output_dim=n_assets, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
    
    # Prepare sequences to avoid leakage (potential integrate into forward method)
    X, Y = prepare_features(asset_returns, asset_prices, target_returns, lookback) # returns tensors
    train_loader = data.DataLoader(data.TensorDataset(X, Y), batch_size = batch_size, shuffle = False, drop_last = True) # droplast as sequence-lookbaack may not be divisable by batch
    
    # run backprop
    for epoch in tqdm(range(n_epochs)):
        for x_batch, y_batch in train_loader:
            weights = model(x_batch)  # output = (batch_size, sequence_length, hidden_size)
            # weights = weights.squeeze(0) # reduces list dimension to hidden dimension size
            
            # Compute portfolio weighted returns 
            returns_t = x_batch[:, -1, n_assets:] # or returns_t[:, ::2]  (batch, n_assets * k)
   
            # compute portfolio return
            Rhat = torch.sum(weights * returns_t, dim=1)
            # print(Rhat.shape, y_batch.shape)

            # Compute loss
            loss = wasserstein_distance(Rhat, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print progress status 
        tpf.train_progress(epoch, n_epochs, loss, show_progress)
    
    # After training, compute final weights/returns 
    with torch.no_grad(): # detach from computation graph 
        weights = model(X)
        final_weights = weights.detach().numpy()  # (m-lookback, n_assets)

        # Final portfolio returns 
        portfolio_returns_final = torch.sum(torch.FloatTensor(final_weights) * asset_returns[lookback:], dim=1).detach().numpy()

    # For publication: Return loss history for plotting convergence
    return {"trained_weights": final_weights, 
            "predicted_returns": portfolio_returns_final, 
            "trained_model": model}