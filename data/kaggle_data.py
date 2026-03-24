import os 
import pandas as pd 
import numpy as np
import kagglehub 

def kaggle_price_return_data(data_path: str, min_data_length: int = 5000):
    """
    pulls multi-csv data from kaggle given data path and retunrs prices and returns over largest common timespan
    """
    df_path = kagglehub.dataset_download(data_path)

    latest_start = pd.Timestamp.min
    earliest_end = pd.Timestamp.max 
    tickers = []
    dataset_prices = []

    for asset_data in os.listdir(df_path): 
        asset = pd.read_csv(
            os.path.join(df_path, asset_data)
        )
        n = asset.shape[0]
        ticker = asset_data.split('.')[0]
        asset['Date'] = pd.to_datetime(asset['Date'], format = '%Y-%m-%d')
        asset.set_index(asset['Date'], inplace = True)
        # features 
        prices = asset['Close']
        # min 
        t1 = asset.index[0]
        tn = asset.index[-1]
        if n >= min_data_length:
            if (t1 > latest_start) and (tn < earliest_end): 
                latest_start = t1
                earliest_end = tn
            # log 
            tickers.append(ticker)
            dataset_prices.append(pd.Series(prices, name = ticker))

    # loop over returns and align data
    for idx, asset in enumerate(dataset_prices):
        dataset_prices[idx] = asset.loc[latest_start:earliest_end]    
    dataset_prices = pd.concat(dataset_prices, axis=1).dropna()
    dataset_returns = np.log(dataset_prices/dataset_prices.shift(periods = 1, axis = 0)).fillna(0)

    return dataset_prices, dataset_returns