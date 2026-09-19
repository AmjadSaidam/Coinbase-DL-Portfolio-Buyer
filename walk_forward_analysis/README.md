# **How to Run run_wfa.py Using Multiprocessing**

A brief step-by-step guide on how to run `run_wfa.py` using a rented GPU via SSH. 

`run_wfa.py` is compute intensive, be default each in-sample (IS)fold of the walk forward analysis contains $14,400$, observations with each feature vector containing $20$ features, that is price and retunr data for $5$ assets, `BTC-GBP, ETH-GBP, LINK-GBP, SOL-GBP, USDT-GBP`. Given the model is trained independently on in each IS fold we can distribute teh training over different processes, this allows for parallel training, which is must faster than traditinaly training one process at one time. 

## Step 1

```
git clone https://github.com/AmjadSaidam/Coinbase-DL-Portfolio-Buyer.git \Coinbase-DL-Portfolio-Buyer
```

## Step 2

```
uv sync
```

## Step 3

```
uv run -m walk_forward_analysis.run_wfa
```

## Step 4

```
scp vast-gpu:/workspace/Coinbase-DL-Portfolio-Buyer/walk_forward_analysis/wfa.pkl "/Users/amjadsaidam/Desktop/Quant stuff /Algorithmic Strategies /coinbase_trading_bot/walk_forward_analysis"
```