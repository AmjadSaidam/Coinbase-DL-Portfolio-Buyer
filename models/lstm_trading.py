"""
"""
# general
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from sparsemax import Sparsemax
# .py
from models.weight_constraint import min_rebalance
from models.device import get_device
import models.training_progress_function as tpf
from models.loss_functions import (kl_divergance, sharpe_ratio_loss)
import copy

# ---------------------------------------------
# Lstm model
# ---------------------------------------------
class lstm_model(nn.Module):
    """
    instentiate lstm model, with predict method 
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 shorts: bool = False, 
                 hidden_dim = 100, 
                 num_layers = 1):
        super().__init__() # super init inherts nn.Module (parent) class methods and attributes
        self.shorts = shorts 

        # LSTM layer for sequential processing 
        dpo = 0.2 if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout = dpo, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim) # linear layer to map LSTM output to weights
        self.output = nn.Tanh() if self.shorts else Sparsemax() # apply last activation to inforce long-only sum to 1 constraint

    def forward(self, 
                x: torch.Tensor, 
                weight_constraint: float | None = None) -> torch.Tensor:
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
            if weight_constraint is not None: 
                final_output = torch.stack([min_rebalance(weights = row, minimum_value = weight_constraint) for row in final_output], dim = 0)
        
        return final_output # last prediction in batch at t=lookback, this will yeild output dim = n_assets
    
class lstm():
    """
    instentiate, train and predict using lstm model 
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 shorts: bool = False,
                 hidden_dim = 252, 
                 num_layers = 1, 
                 volatility_lookback: int | None = None,
                 weight_constraint: float | None = None, 
                 sharpe_loss: bool = True): 
        self.device = get_device()
        self.model: lstm_model = lstm_model(input_dim, output_dim, shorts, hidden_dim, num_layers).to(self.device)
        self.w_min = weight_constraint
        self.vol_scale_lkb = volatility_lookback
        self.vol_trg = 0.0
        self.sharpe_loss = sharpe_loss
        self.p = 2
        self.avg_tr_loss = 0.0
        self.tr_loss_container = []
        self.eval_loss = 0.0
        self.eval_loss_container = []
        self.opt_res = {}
        self.cost = 0
        
    def lstm_train(self, 
                   train_loader: data.DataLoader,
                   eval_loader: data.DataLoader, 
                   learning_rate: float = 0.0001, 
                   n_epochs: int = 100, 
                   show_progress: bool = False) -> lstm_model: 
        """in-sample model training function"""
        best_eval_loss = float('inf')
        best_model_params = None

        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = 1e-5) # instentiate model and optimiser

        # run backprop
        for epoch in tqdm(range(n_epochs)):
            self.model.train() # set model in training mode
            total_p = 0
            epoch_loss = 0
            self.tr_loss = 0
            for (x_tr, y_tr, rt_tr, x_inv_tr) in train_loader:
                # send to devices 
                x_tr, y_tr, rt_tr, x_inv_tr, w_p, _ = self.__forward_pass(x_tr, y_tr, rt_tr, x_inv_tr) # w_p of dim (batch, n_assets)
                
                # Compute portfolio weighted returns 
                rt_p = self.__portfolio_returns(w_p, rt_tr) # (batches)

                # Compute loss
                if self.sharpe_loss:
                    loss = sharpe_ratio_loss(rt_p)
                else:
                    #loss = wasserstein_distance(rt_p, y_tr, self.p)
                    #loss = mmd(rt_p, y_tr, self.device)
                    loss = kl_divergance(rt_p, y_tr) 

                # backward pass
                optimizer.zero_grad() # clear gradients
                loss.backward()
                optimizer.step()

                # prediction and prediction loss 
                self.tr_loss += loss.item()
                total_p += 1

            # average epoch loss
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
            if show_progress:
                tpf.train_progress(epoch, n_epochs, loss)   

        # final loss
        self.avg_tr_loss = np.sum(self.tr_loss_container) / n_epochs 

    def lstm_evaluate(self, 
                      eval_loader: data.DataLoader) -> dict:
        """set model in evaluation mode for out-of-sample predictions"""
        all_rt_p = [] # pad vectors to account for lookback 
        all_w_p = []
        all_vol_scale = []

        self.eval_loss = 0
        total_p = 0
        scaled_pos = 0
        
        self.model.eval() # model in evaluation model 
        with torch.no_grad(): # disable gradient tracking 
            for (x_eval, y_eval, rt_eval, x_inv_eval) in eval_loader:
                # forward pass
                x_eval, y_eval, rt_eval, x_inv_eval, w_p, vol_scaler = self.__forward_pass(x_eval, y_eval, rt_eval, x_inv_eval)
                
                if self.vol_scale_lkb is not None:
                    scaled_pos = torch.sum(vol_scaler, dim = 1) # inner product of weight and volatility vector
                
                # predicted returns
                rt_p = self.__portfolio_returns(w_p, rt_eval)

                # loss
                if self.sharpe_loss:
                    loss = sharpe_ratio_loss(rt_p)
                else:
                    #loss = wasserstein_distance(rt_p, y_eval, self.p)
                    #loss = mmd(rt_p, y_eval, self.device)
                    loss = kl_divergance(rt_p, y_eval) 

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
        all_vol_scale = torch.cat(all_vol_scale, dim = 0).detach().cpu().numpy() if (self.vol_scale_lkb is not None) else []

        return {
            'weights': all_w_p, 
            'returns': all_rt_p, 
            'vol_scale': all_vol_scale
        } 
    
    # helpers 
    def __portfolio_returns(self, 
                            port_weight, 
                            port_returns):
        """portfolio return, assume log returns"""
        prev_w_p = torch.cat((torch.zeros_like(port_weight[:1]), torch.roll(port_weight, shifts = 1, dims = 0)[:-1]))
        return torch.log(torch.sum(port_weight * torch.exp(port_returns), dim = -1)) - self.cost * torch.sum(torch.abs(port_weight - prev_w_p), dim = 1)

    def __forward_pass(self, 
                       x, 
                       y, 
                       rt, 
                       x1_inv):
        """batched forward pass"""
        # feature/labels 
        x = x.to(self.device)
        y = y.to(self.device)
        # rt and inv feature 1 batch 
        rt = rt.to(self.device)
        x1_inv = x1_inv.to(self.device)
        # forward pass - optimal model used if trained
        w_p = self.model(x, self.w_min)
        # output activation 
        vol_scaler = None
        if self.vol_scale_lkb is not None: 
            vol_scaler = vol_scale(x1_inv, self.vol_trg, self.vol_scale_lkb)[:, -1, :] # (batch, lookback, features) -> last(batch, features)
            w_p = w_p * vol_scaler
        return x, y, rt, x1_inv, w_p, vol_scaler

# ---------------------------------------------
# Volatility Scaling 
# ---------------------------------------------
def vol_scale(a: torch.Tensor, 
              target_vol: float, 
              vol_lookback) -> torch.Tensor: 
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
                exenate_vol = torch.std(subset, dim = 0).clamp_min(eps) # (1, lookback_t), set min val to avoid div by zero error
                vol_t = target_vol / exenate_vol # levergae
                current_scale = alpha * vol_t + (1 - alpha) * prev_scale # ema 
            time_outputs.append(current_scale)
            prev_scale = current_scale
        batch_outputs.append(torch.stack(time_outputs, dim = 0))

    return torch.stack(batch_outputs, dim = 0)