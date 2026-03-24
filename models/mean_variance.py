import numpy as np 
import pypfopt 

def mvo(returns: np.ndarray, time_scale: float, rfr: float = 0):
    mu = returns.mean(axis = 0) * time_scale # asset mean 
    cov = np.cov(returns.T, ddof = 1) * time_scale # var-covariance matrix 
    ef = pypfopt.EfficientFrontier(mu, cov, weight_bounds = (0, 1))
    w = ef.max_sharpe(rfr)
    w_opt = ef.clean_weights()

    return np.array(list(w_opt.values())), ef
