# default inputs
data_frequencies = ['UNKNOWN_GRANULARITY', 
                    'ONE_MINUTE', 
                    'FIVE_MINUTE', 
                    'FIFTEEN_MINUTE', 
                    'THIRTY_MINUTE', 
                    'ONE_HOUR', 
                    'TWO_HOUR', 
                    'FOUR_HOUR', 
                    'SIX_HOUR', 
                    'ONE_DAY']

# seconds per candle, keyed by granularity
granularity_seconds = {
    'ONE_MINUTE': 60,
    'FIVE_MINUTE': 300,
    'FIFTEEN_MINUTE': 900,
    'THIRTY_MINUTE': 1800,
    'ONE_HOUR': 3600,
    'TWO_HOUR': 7200,
    'FOUR_HOUR': 14400,
    'SIX_HOUR': 21600,
    'ONE_DAY': 86400,
}