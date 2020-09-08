import os
os.chdir('/Users/tinglam/Documents/GitHub/options_trading_simulator')
import bsm
import option_data



# 700 HK Call
S = 525
K = 530
T = 21/365
r = 0
q = 0
IV = 0.3854

x = bsm.call(S,K,T,r,q,IV,position='long')

x.describe()

bsm.get_simulation(option = x,expected_price = 515, std = 0.036, skew = 0, n = 2500, Tt = 20/365)



# 1810 HK Call
S = 24.6
K = 26
T = 21/365
r = 0
q = 0
IV = 0.6761

x = bsm.call(S,K,T,r,q,IV,position='long')

x.describe()

bsm.get_simulation(option = x,expected_price = 25, std = 0.03, skew = -2, n = 25000, Tt = 21/365)



# Short 700 HK Put
S = 505
K = 480
T = 21/365
r = 0
q = 0
IV = 0.385

x = bsm.put(S,K,T,r,q,IV,position='short')

x.describe()

bsm.get_simulation(option = x,expected_price = 505, std = 0.036, skew = 0, n = 2500, Tt = 19/365)




# Short Straddle
S = 24.6
K = 22
T = 21/365
r = 0
q = 0
IV = 0.75

x = bsm.straddle(S,K,T,r,q,IV,position='short')

x.describe()

bsm.get_simulation(option = x,expected_price = 25, std = 0.03, skew = -2, n = 25000, Tt = 21/365)



# Test
S = 515.5
K = 530
T = 22/365
r = 0
q = 0
IV = 0.42

x = bsm.straddle(S,K,T,r,q,IV,position='long')

x.describe()

bsm.get_simulation(option = x,expected_price = 520, std = 0.036, skew = 0, n = 2500, Tt = 20/365)


# Test 2
S = 515.5
K1 = 510
K2 = 530
T = 22/365
r = 0
q = 0
IV = 0.42

x = bsm.spread(S,K1,K2,T,r,q,IV,option_type = 'put')

x.describe()

bsm.get_simulation(option = x,expected_price = 515.5, std = 0.01, skew = 0, n = 2500, Tt = 20/365)