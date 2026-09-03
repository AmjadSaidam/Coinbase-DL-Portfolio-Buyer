# This script should be ran after the market_algorithm.py file has been ran and the model has produces some valid trades (fingers crossed hehe)
# we can then validate if the order functions are correctey allocating weights to each asset equal to those outputed by the model 

import pandas as pd 
import numpy as np 
import unittest

# check weight diff 
class CheckWeights(unittest.TestCase): # inherits form TestCase module, each test will get a new instance if the TestCase class
    def check_weights(self, database1: np.ndarray, database2: np.ndarray, tolarance: float = 0.01):
        np.testing.assert_allclose(database1, database2, atol = tolarance) # error interval is 1/100 percision 

# run 
if __name__ == '__main__':
    unittest.main()