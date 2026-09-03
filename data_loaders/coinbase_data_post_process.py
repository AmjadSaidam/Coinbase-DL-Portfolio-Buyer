"""
standerdises data dowloaded from coinbase_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd


def coinbase_price_return_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """loads coinbase OHLCV csvs, aligns them on a common start and returns prices and returns"""
    data_universe = {}

    for f in Path(data_path).iterdir():
        data = pd.read_csv(f, index_col='Unnamed: 0')
        data.set_index(pd.DatetimeIndex(data.index), inplace=True)
        symbol = f.name.split('_')[0]
        data_universe[symbol] = data

    # drop leading rows so every symbol starts once all assets have live (non-NaN) data
    drop_index = max(np.isnan(data).sum().max() for data in data_universe.values())
    for symbol, data in data_universe.items():
        data_universe[symbol] = data.iloc[drop_index:, :]

    dataset_prices = pd.concat(
        [data['close'].rename(symbol) for symbol, data in data_universe.items()],
        axis=1
    ).dropna()
    dataset_returns = np.log(dataset_prices / dataset_prices.shift(periods=1, axis=0)).fillna(0)

    return dataset_prices, dataset_returns