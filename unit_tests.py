# This script should be ran after the market_algorithm.py file has been ran and the model has produces some valid trades (fingers crossed hehe)
# we can then validate if the order functions are correctey allocating weights to each asset equal to those outputed by the model 

import pandas as pd 
import numpy as np 
import unittest
import sqlite3
import data_to_sql
from market_algorithm import portfolio_assets

# data fetch function
def get_sql_data(data_base_columns, database_name, data_name):
    sql_db = data_to_sql.CreateSQLightDatabase(tickers = data_base_columns, name_database = database_name, name_data = data_name)
    sql_data = sql_db.create_access_database_file() # access data
    return sql_db.query_database()

# get both datasets
data_model_weights = get_sql_data(portfolio_assets, 'model_output.db', 'model_output')
print(data_model_weights)
data_real_weights = get_sql_data(portfolio_assets, 'real_invested_weights.db', 'real_invested_weights')
print(data_real_weights)

# check weight diff 
class CheckWeights(unittest.TestCase): # inherits form TestCase module, each test will get a new instance if the TestCase class
    def check_weights(self):
        self.assertEqual(data_model_weights, data_real_weights) # are we assigning the correct weight to each asset

# run 
if __name__ == '__main__':
    unittest.main()