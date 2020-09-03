import os
os.chdir('/Users/tinglam/Documents/GitHub/options_trading_simulator')
import bsm
import matplotlib.pyplot as plt

x = bsm.put(S=23.9,K=23,T=26/365,r=0,q=0,IV=0.728,position='long')

x.describe()

bsm.get_simulation(option = x,expected_price = 23.6, std = 0.03, skew = 0, n = 25000, Tt = 25/365)