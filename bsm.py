'''
TODO:
1. Calculate IV from current option prices ####
2. Compare between options (costs, expected payoff) -> Optimization
3. Currently assume hold to maturity. Calculate resold prices (with MS simulation)
4. Simulation for other strategies
'''

import os
os.chdir('/Users/tinglam/Documents/GitHub/options_trading_simulator')
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as si
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from copy import deepcopy

class black_scholes:
    def __init__(self, S, K, T, r, q, IV, position = 'long'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.IV = IV
        self.d1 = (np.log(S / K) + (r - q + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
        self.d2 = (np.log(S / K) + (r - q - 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))

    def price(self,option_type, S = None):
        """Calcaulate price of options.

        Args:
            option_type (str): Type of option. Either 'call' or 'put'.
            S (float or int, optional): Spot price used to evaluate option price. Defaults to None.

        Returns:
            float: Price of option.
        """        
        S  = self.S if S == None else S # If S not provided, use S_0 as St
        if option_type == 'call':
            price = S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0)
        elif option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1, 0.0, 1.0)
        return price

    def delta(self,option_type):
        if option_type == 'call':
            delta = si.norm.cdf(self.d1, 0.0, 1.0)
        elif option_type == 'put':
            delta = -si.norm.cdf(-self.d1, 0.0, 1.0)
        return delta

    def gamma(self):
        gamma = self.S / 100 * np.exp((-self.q * self.T)) * norm.pdf(self.d1) / ( self.S * self.IV * np.sqrt(self.T))
        return gamma
    
    def vega(self):
        vega = 0.01 * self.S * np.exp(- self.q * self.T) * norm.pdf (self.d1) * np.sqrt(self.T) 
        return vega

    def theta(self,option_type):
        df = np.exp(-(self.r * self.T))
        dfq = np.exp((-self.q * self.T))
        if option_type == 'call':
            theta = (1/ 365) * (-0.5 * self.S * dfq * norm.pdf(self.d1) * self.IV / (self.T ** 0.5) + 1 * (self.q * self.S * dfq * norm.cdf(1 * self.d1) - self.r * self.K * df * norm.cdf(1 * self.d2)))
        elif option_type == 'put':
            theta = (1/ 365) * (-0.5 * self.S * dfq * norm.pdf(self.d1) * self.IV / (self.T ** 0.5) + -1 * (self.q * self.S * dfq * norm.cdf(-1 * self.d1) - self.r * self.K * df * norm.cdf(-1 * self.d2)))
        return theta

    def rho(self,option_type):
        if option_type == 'call':
            rho = 1 * self.K * self.T * np.exp(-(self.r * self.T)) * 0.01 * norm.cdf (1 * self.d2)
        elif option_type == 'put':
            rho = -1 * self.K * self.T * np.exp(-(self.r * self.T)) * 0.01 * norm.cdf (-1 * self.d2)
        return rho

    def leverage(self): # Effective gearing
        return self.S * self.delta / self.price

    def exposure(self): # Delta-adjusted notional value
        return self.S * self.delta / self.S
    
    def profit(self,St):
        return self.payoff(St) - self.price

    def plot_profit(self,n=2500):
        pct_chg = np.random.normal(1, self.IV * self.T, n)
        ms_St = pct_chg * self.S
        ms_profit = self.profit(ms_St)
        df = pd.DataFrame({'St' : ms_St, 'Profit' : ms_profit}).sort_values('St').reset_index(drop = True)
        fig, ax = plt.subplots() # Create the plot objects
        ax.scatter(df['St'], df['Profit'],s = 5) # Input the data for the graph and plot
        ax.set_xlabel('St')
        ax.set_ylabel('Profit')
        plt.grid()
        plt.show()

    def describe(self):
        print("Strategy: " + self.strategy_name)
        print("Price: " + str("{:.3f}".format(self.price)))
        print("Delta: " + str("{:.3f}".format(self.delta)))
        print("Gamma: " + str("{:.3f}".format(self.gamma)) + " (per % of Spot)")
        print("Vega: " + str("{:.3f}".format(self.vega)) + " (per % of IV)")
        print("Theta: " + str("{:.3f}".format(self.theta)) + " (daily)")
        print("Rho: " + str("{:.3f}".format(self.rho)) + " (per % of Interest Rate)")
        print("Leverage: " + str("{:.3f}".format(self.leverage)))
        print("Exposure: " + str("{:.3f}".format(self.exposure)))
        
    def short_position(self):
        variables = ['price','delta','gamma','vega','theta','rho','leverage','exposure']
        for i in variables:
            try:
                vars(self)[i] = vars(self)[i] * -1
            except:
                pass

class call(black_scholes):
    def __init__(self, S, K, T, r, q, IV, position='long'):
        super().__init__(S, K, T, r, q, IV, position)
        self.position = position
        self.strategy_name = self.position.capitalize() + ' Call @ K = ' + str(K)
        
        self.price = black_scholes.price(self,'call')
        self.delta = black_scholes.delta(self,'call')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'call')
        self.rho = black_scholes.rho(self,'call')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)
        
        if self.position == 'short':
            self.short_position()
        
    def payoff(self,St):
        position_adjust = -1 if self.position == 'short' else 1
        return np.where(St > self.K, St - self.K, 0) * position_adjust

    def profit(self,St):
        return self.payoff(St) - self.price

class put(black_scholes):
    def __init__(self, S, K, T, r, q, IV, position='long'):
        super().__init__(S, K, T, r, q, IV, position)
        self.position = position
        self.strategy_name = self.position.capitalize() + ' Call @ K = ' + str(K)
        
        self.price = black_scholes.price(self,'put')
        self.delta = black_scholes.delta(self,'put')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'put')
        self.rho = black_scholes.rho(self,'put')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)
        
        if self.position == 'short':
            self.short_position()

    def payoff(self,St):
        position_adjust = -1 if self.position == 'short' else 1
        return np.where(St < self.K, self.K - St, 0) * position_adjust
    
    def profit(self,St):
        return self.payoff(St) - self.price

class straddle(call,put):
    def __init__(self, S, K, T, r, q, IV, position='long'):
        super().__init__(S, K, T, r, q, IV, position)
        self.position = position
        self.strategy_name = self.position.capitalize() + ' Straddle @ K = ' + str(K)
        
        option_1 = call(S, K, T, r, q, IV, position)
        option_2 = put(S, K, T, r, q, IV, position)
        
        self.option_1 = option_1
        self.option_2 = option_2
        self.price = option_1.price + option_2.price
        self.delta = option_1.delta + option_2.delta
        self.gamma = option_1.gamma + option_2.gamma
        self.vega = option_1.vega + option_2.vega
        self.theta = option_1.theta + option_2.theta
        self.rho = option_1.rho + option_2.rho
        self.leverage = option_1.leverage + option_2.leverage
        self.exposure = option_1.vega + option_2.vega
        
    def payoff(self,St):
        return self.option_1.payoff(St) + self.option_2.payoff(St)
    
    def profit(self,St):
        return self.payoff(St) - self.price

class spread(call,put):
    '''
    K1 < K2 means a bull spread.
    Always long K1, short K2
    option_type = call or put
    '''
    def __init__(self, S, K1, K2, T, r, q, IV, option_type):
        super().__init__(S, np.nan, T, r, q, IV)
        
        self.direction = 'Bull' if K1 < K2 else 'Bear'
        self.option_type = option_type
        
        if self.option_type == 'call':
            self.strategy_name = self.direction + ' Call Spread @ K1 = ' + str(K1) + ', K2 = ' + str(K2)
            option_1 = call(S, K1, T, r, q, IV) # Long call
            option_2 = call(S, K2, T, r, q, IV) # Short call
        elif self.option_type == 'put':
            self.strategy_name = self.direction + ' Put Spread @ K1 = ' + str(K1) + ', K2 = ' + str(K2)
            option_1 = put(S, K1, T, r, q, IV) # Long Put
            option_2 = put(S, K2, T, r, q, IV) # Short Put
        
        
        self.option_1 = option_1
        self.option_2 = option_2     
        self.price = option_1.price - option_2.price
        self.delta = option_1.delta - option_2.delta
        self.gamma = option_1.gamma - option_2.gamma
        self.vega = option_1.vega - option_2.vega
        self.theta = option_1.theta - option_2.theta
        self.rho = option_1.rho - option_2.rho
        self.leverage = option_1.leverage - option_2.leverage
        self.exposure = option_1.vega - option_2.vega
        
    def payoff(self,St):
        return self.option_1.payoff(St) - self.option_2.payoff(St)
    
    def profit(self,St):
        return self.option_1.payoff(St) - self.option_2.payoff(St) - self.option_1.price + self.option_2.price

def get_simulation(option, expected_price, std, skew, n=2500, Tt='maturity'):
    
    def simulate(option, expected_price, std, skew, n, Tt):
        """Create a dataframe and perform monte carlo simulation.

        Args:
            option (pur or call object): The option object that was used in the simulation
            expected_price (int): The expected price at T = Tt
            std ([type]): [description] #TODO: its scale
            skew (int): Skewness of St's distribution at T = Tt
            n (int): Number of simulations to run
            Tt (str): Time left (in years) before expiry when profits are simulated

        Returns:
            DataFrame: Columns = ['St','Profit']
        """            
        pct_chg = skewnorm.rvs(skew,loc = 0, scale = std,size=n)
        ms_St = expected_price * (1+pct_chg)
        
        if Tt == 'maturity': # If option will be held to maturity
            ms_profit = option.profit(ms_St)
        else: # If option will be resold at T = Tt
            cost = option.price
            constructor = globals()[option.__class__.__name__]
            if option.__class__.__name__ == 'spread':
                option_simulated = constructor(S=ms_St,K1=option.option_1.K,K2=option.option_2.K,T=Tt,r=option.r,q=option.q,IV=option.IV,option_type=option.option_type)
            else:
                option_simulated = constructor(S=ms_St,K=option.K,T=Tt,r=option.r,q=option.q,IV=option.IV,position=option.position)
            
            ms_profit = option_simulated.price - cost

        return pd.DataFrame({'St' : ms_St, 'Profit' : ms_profit}).sort_values('St').reset_index(drop = True)

    simulation = simulate(option=option, expected_price=expected_price, std=std, skew=skew, n=n, Tt=Tt)

    def plot_St():
        plt.hist(simulation['St'], density=True, bins=30)  # `density=False` would make counts
        plt.ylabel('Probability')
        plt.xlabel('St')
        plt.grid()
        plt.show()

    def plot_profit():
        fig, ax = plt.subplots()
        ax.scatter(simulation['St'], simulation['Profit'],s = 5)
        ax.set_xlabel('St')
        ax.set_ylabel('Profit')
        plt.grid()
        plt.show()
        
    def forecasting_profit():
        l = simulation['Profit']
        profit_prob = len([1 for i in l if i > 0]) / len(l) 
        print('Probability of making profit = ' + "{:.2f}".format(profit_prob*100) + '%')
        
        break_even_index = next(i for i,v in enumerate(sorted(l)) if v > 0)
        break_even_St = simulation.sort_values('Profit').reset_index(drop=True)['St'][break_even_index]
        
        print('Break-even St = $' + "{:.3f}".format(break_even_St))
        
        print('Median Profit = $' + "{:.2f}".format(np.median(l)))
        
        print('Std of profit = ' + "{:.2f}".format(np.std(l)))
        
        K_list  = np.concatenate([np.arange(5,10,0.25),np.arange(10,20,0.5),np.arange(20,50,1),np.arange(20,50,1),np.arange(50,200,2.5),np.arange(200,300,5),np.arange(300,1000,10)])
        idx = (np.abs(K_list - option.S)).argmin()
        test_K = list(K_list[idx-5:idx+6])
        test_profit_prob = []
        
        constructor = globals()[option.__class__.__name__]
        if option.__class__.__name__ == 'spread':
            pass
        else:
            for K in test_K:
                test_option = constructor(option.S, K, option.T, option.r, option.q, option.IV, option.position)
                test_df = simulate(test_option,expected_price, std, skew, n=n, Tt = Tt)
                test_l = test_df['Profit']
                test_profit_prob.append(len([1 for i in test_l if i > 0]) / len(test_l))
                best_K = test_K[test_profit_prob.index(max(test_profit_prob))]
            print('Best K = ' + str(best_K))

    
    plot_St()
    plot_profit()
    forecasting_profit()



"""
example_1 = call(S=530,K=540,T=29/365,r=0.03,q=0,IV=0.48,position='long')
example_1.describe()
get_simulation(example_1, expected_price = 50, std = 0.1,skew = -5,n = 2500,Tt = 10/365)



x = straddle(S=23.65,K=22,T=29/365,r=0.03,q=0,IV=0.7,position = 'short')
x.describe()


x1 = call(S=23.65,K=22,T=29/365,r=0.03,q=0,IV=0.7,position = 'short')
x1.describe()

x2 = put(S=23.65,K=22,T=29/365,r=0.03,q=0,IV=0.7,position = 'short')
x2.describe()

get_simulation(x,expected_price = 22,std=0.02,skew=0,n=2500,Tt = 10/365)



example_2.get_simulation(expected_price = 23.65, std = 0.2,skew = -5,n = 2500)


example_3 = spread(S=23.65,K1=22,K2=24,T=29/365,r=0.03,q=0,IV=0.7,option_type='call')
example_3.describe()
example_3.plot_profit(n = 2500)



x = put(S=23.9,K=23,T=26/365,r=0,q=0,IV=0.728,position='long')

get_simulation(example_3,expected_price=23.9,std=0.16,skew=0, Tt=20/365)

x.describe()
y = put(S=24.2,K=23,T=25/365,r=0,q=0,IV=0.728,position='long')
y.price-x.price


x.get_simulation(expected_price=23.9,std=0.05,skew=5)

#x.plot_profit()


x.get_simulation(expected_price=23.9,std=0.06,skew=0, Tt=20/365)

"""