# run script using <python -m multi-processing.wfa_multi_processing>

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

def rolling_volatility(prices: pd.DataFrame):
    """
    rolling volatility given asset prices
    """
    n, m = prices.shape
    rolling_vol = np.zeros((n, m))
    lookback = 10
    for t in range(lookback, n):
        subset = prices.iloc[t - lookback:t, :]
        std = np.std(subset, axis = 0)
        rolling_vol[t, :] = std
    
    return rolling_vol

def target_returns_label(returns: pd.DataFrame, skew: int = 1, mean: float = 0.0, varaince: float = 0.01):
    """
    sampels from target skewed normal distribution
    """ 
    np.random.seed()
    n = returns.shape[0]
    target_returns = skewnorm.rvs(size = n, **{"a": skew, 'loc': mean, 'scale': varaince})  

    return target_returns
