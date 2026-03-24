import numpy as np 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils import data 
from models.device import get_device
from models.lstm_trading import wasserstein_distance, mmd, tensor_standardise
import copy

class neural_net_model(nn.Module):
    """
    """
    def __init__(self, in_dim, out_dim, num_neurons=100, num_layers=10, long_only: bool = True):
        super().__init__()
        self.long_only = long_only
        layers = []
        layers.append(nn.Linear(in_dim , num_neurons))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            al = nn.Linear(num_neurons, num_neurons)
            zl = nn.ReLU()
            layers.append(al)
            layers.append(zl)
        aL = nn.Linear(num_neurons, out_dim)
        layers.append(aL)
        self.fc = nn.Sequential(*layers)
        self.last = nn.Softmax(dim = -1) if long_only else nn.Tanh()
    
    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        out = self.last(out) 
        if not self.long_only:
            out / torch.sum(torch.abs(out), dim = 1, keepdim = True)

        return out
    
def data_pre_process(feature_data: torch.Tensor, current_return: torch.Tensor, labels: torch.Tensor, batches: int):
    """
    """
    return data.DataLoader(data.TensorDataset(feature_data, current_return, labels), shuffle = False, drop_last = False, batch_size = batches)

class simple_neural_net():
    def __init__(self, in_dim, out_dim, num_neurons=100, num_layers=10, long_only: bool = True):
        self.device = get_device()
        self.model: neural_net_model = neural_net_model(in_dim, out_dim, num_neurons, num_layers, long_only).to(self.device)
        self.tr_loss_container: list = []
        self.eval_loss_container: list = []
        self.learning_rate: float = 1e-3
        self.eval_loss: float = 0
        self.opt_train_res: float = []
    
    def simple_neural_net_train(self, train_loader, eval_loader, epochs):
        """"""
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        min_loss = float('inf')
        best_model_params = None

        for epoch in tqdm(range(epochs)): 
            tr_loss = 0
            total_pred = 0 
            for (x_tr, rt, y_tr) in train_loader:
                # send to devices 
                x_tr = x_tr.to(self.device)
                rt = rt.to(self.device)
                y_tr = y_tr.to(self.device)
                # predict 
                w_p = self.model(x_tr)
                rp = torch.sum(w_p * rt, dim = 1)
                loss = wasserstein_distance(rp, y_tr)

                # backpropegate
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()

                # loss
                tr_loss += loss.item()
                total_pred += 1
            
            epoch_loss = tr_loss / total_pred
            self.tr_loss_container.append(epoch_loss)

            # evaluate model
            eval_res = self.simple_neural_net_evaluate(eval_loader)
            if self.eval_loss < min_loss:
                best_model_params = copy.deepcopy(self.model.state_dict())
                min_loss = self.eval_loss
                self.opt_train_res = eval_res
        
            if best_model_params is not None:
                self.model.load_state_dict(best_model_params)

    def simple_neural_net_evaluate(self, test_loader): 
        """
        """
        all_rp, all_w = [], []

        self.eval_loss = 0
        total_pred = 0

        self.model.eval()
        with torch.no_grad():
            for (x_ts, rt, y_ts) in test_loader:
                # pin 
                x_ts = x_ts.to(self.device)
                rt = rt.to(self.device)
                y_ts = y_ts.to(self.device)
                # predict 
                w_p = self.model(x_ts)
                rp = torch.sum(w_p * rt, dim = 1)

                loss = wasserstein_distance(rp, y_ts)

                # log outputs
                self.eval_loss += loss.item()
                total_pred += 1
                all_w.append(w_p)
                all_rp.append(rp)
        
        self.eval_loss /= total_pred

        all_rp = torch.cat(all_rp, dim = 0).detach().cpu().numpy()
        all_w = torch.cat(all_w, dim = 0).detach().cpu().numpy()
        
        return {
            'weights': all_w, 
            'returns': all_rp, 
        }
