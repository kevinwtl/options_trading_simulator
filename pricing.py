import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as si
from scipy.stats import norm
from scipy.stats import skewnorm
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt

'''
TODO:
1. describe the expected profit, probability of making profits, breakeven
2. Compare between options (costs, expected payoff)
3. describe IV from current option prices
'''

class black_scholes:
    def __init__(self, S, K, T, r, q, IV):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.IV = IV
        self.d1 = (np.log(S / K) + (r - q + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
        self.d2 = (np.log(S / K) + (r - q - 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))

    def price(self,option_type):
        if option_type == 'call':
            price = self.S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0)
        elif option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - self.S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1, 0.0, 1.0)
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
    
    def plot_payoff(self,n=2500):
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

class call(black_scholes):
    def __init__(self, S, K, T, r, q, IV):
        super().__init__(S, K, T, r, q, IV)
        self.strategy_name = 'Call @ K = ' + str(K)
        
        self.price = black_scholes.price(self,'call')
        self.delta = black_scholes.delta(self,'call')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'call')
        self.rho = black_scholes.rho(self,'call')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)
        
    def payoff(self,St):
        return np.where(St > self.K, St - self.K, 0)
    
    def profit(self,St):
        return np.where(St > self.K, St - self.K, 0) - self.price

class put(black_scholes):
    def __init__(self, S, K, T, r, q, IV):
        super().__init__(S, K, T, r, q, IV)
        self.strategy_name = 'Put @ K = ' + str(K)
        
        self.price = black_scholes.price(self,'put')
        self.delta = black_scholes.delta(self,'put')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'put')
        self.rho = black_scholes.rho(self,'put')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)

    def payoff(self,St):
        return np.where(St < self.K, self.K - St, 0)
    
    def profit(self,St):
        return np.where(St < self.K, self.K - St, 0) - self.price

class straddle(call,put):
    def __init__(self, S, K, T, r, q, IV):
        super().__init__(S, K, T, r, q, IV)
        self.strategy_name = 'Straddle @ K = ' + str(K)
        
        option_1 = call(S, K, T, r, q, IV) # Long Call
        option_2 = put(S, K, T, r, q, IV) # Long Put
        
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
        return self.option_1.payoff(St) + self.option_2.payoff(St) - self.price

class spread(call,put):
    '''
    K1 < K2 means a bull spread.
    Always long K1, short K2
    option_type = call or put
    '''
    def __init__(self, S, K1, K2, T, r, q, IV, option_type):
        super().__init__(S, np.nan, T, r, q, IV)
        
        self.direction = 'Bull' if K1 < K2 else 'Bear'
        
        if option_type == 'call':
            self.strategy_name = self.direction + ' Call Spread @ K1 = ' + str(K1) + ', K2 = ' + str(K2)
            option_1 = call(S, K1, T, r, q, IV) # Long call
            option_2 = call(S, K2, T, r, q, IV) # Short call
        elif option_type == 'put':
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

class montecarlo:
    def __init__(self, option, expected_price, std, skew, n=1000): #TODO:  calculate value when resold (add variable t)
        self.option = option
        self.expected_price = expected_price
        self.std = std
        self.skew = skew
        self.n = n
        self.simulation = self.get_simulation()

    def get_simulation(self):
        #sns.set_style('whitegrid')
        #pct_chg = np.random.lognormal(sigma=self.sd,size=self.n)
        pct_chg = skewnorm.rvs(self.skew,loc = 0, scale = 0.1,size=self.n)
        ms_St = self.expected_price * (1+pct_chg)
        ms_profit = self.option.profit(ms_St)
        return pd.DataFrame({'St' : ms_St, 'Profit' : ms_profit}).sort_values('St').reset_index(drop = True)

    def plot_St(self):
        plt.hist(self.simulation['St'], density=True, bins=30)  # `density=False` would make counts
        plt.ylabel('Probability')
        plt.xlabel('St')
        plt.grid()
        plt.show()

    def plot_payoff(self):
        fig, ax = plt.subplots()
        ax.scatter(self.simulation['St'], self.simulation['Profit'],s = 5)
        ax.set_xlabel('St')
        ax.set_ylabel('Profit')
        plt.grid()
        plt.show()

"""
example_1 = call(S=530,K=540,T=29/365,r=0.03,q=0,IV=0.48)
example_1.describe()
example_1.plot_payoff(n = 2500)

example_2 = straddle(S=23.65,K=22,T=29/365,r=0.03,q=0,IV=0.7)
example_2.describe()
example_2.plot_payoff(n = 2500)

example_3 = spread(S=23.65,K1=22,K2=24,T=29/365,r=0.03,q=0,IV=0.7,option_type='call')
example_3.describe()
example_3.plot_payoff(n = 2500)
"""



x = straddle(S=25.6,K=24,T=27/365,r=0.03,q=0,IV=0.6874)
x.describe()
x.plot_payoff()

montecarlo(option = x,expected_price = 24.5,std = 1.5, skew = -5, n=2500).plot_St()
montecarlo(option = x,expected_price = 24.5,std = 1.5, skew = -5, n=2500).plot_payoff()