import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as si
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
'''
TODO: payoff of shorted options are not correctly reflected
'''


class black_scholes:

    def __init__(self, S, K, T, r, q, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

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
        gamma = self.S / 100 * np.exp((-self.q * self.T)) * norm.pdf(self.d1) / ( self.S * self.sigma * np.sqrt(self.T))
        return gamma
    
    def vega(self):
        vega = 0.01 * self.S * np.exp(- self.q * self.T) * norm.pdf (self.d1) * np.sqrt(self.T) 
        return vega

    def theta(self,option_type):
        df = np.exp(-(self.r * self.T))
        dfq = np.exp((-self.q * self.T))
        if option_type == 'call':
            theta = (1/ 365) * (-0.5 * self.S * dfq * norm.pdf(self.d1) * self.sigma / (self.T ** 0.5) + 1 * (self.q * self.S * dfq * norm.cdf(1 * self.d1) - self.r * self.K * df * norm.cdf(1 * self.d2)))
        elif option_type == 'put':
            theta = (1/ 365) * (-0.5 * self.S * dfq * norm.pdf(self.d1) * self.sigma / (self.T ** 0.5) + -1 * (self.q * self.S * dfq * norm.cdf(-1 * self.d1) - self.r * self.K * df * norm.cdf(-1 * self.d2)))
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

    def calculate(self):
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
    def __init__(self, S, K, T, r, q, sigma):
        super().__init__(S, K, T, r, q, sigma)
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
        return np.where(St > self.S, St - self.S, 0) - self.price
    
    def profit(self,St):
        return np.where(St > self.S, St - self.S, 0)

class put(black_scholes):
    def __init__(self, S, K, T, r, q, sigma):
        super().__init__(S, K, T, r, q, sigma)
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
        return np.where(St < self.S, self.S - St, 0)
    
    def profit(self,St):
        return np.where(St < self.S, self.S - St, 0) - self.price

class straddle(call,put):
    def __init__(self, S, K, T, r, q, sigma):
        super().__init__(S, K, T, r, q, sigma)
        self.strategy_name = 'Straddle @ K = ' + str(K)
        
        option_1 = call(S, K, T, r, q, sigma) # Long Call
        option_2 = put(S, K, T, r, q, sigma) # Long Put
        
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
    def __init__(self, S, K1, K2, T, r, q, sigma, option_type):
        super().__init__(S, K, T, r, q, sigma)
        
        if option_type == 'call':
            self.strategy_name = 'Call Spread @ K1 = ' + str(K1) + ', K2 = ' + str(K2)
            option_1 = call(S, K1, T, r, q, sigma) # Long call
            option_2 = call(S, K2, T, r, q, sigma) # Short call
        elif option_type == 'put':
            self.strategy_name = 'Put Spread @ K1 = ' + str(K1) + ', K2 = ' + str(K2)
            option_1 = put(S, K1, T, r, q, sigma) # Long Put
            option_2 = put(S, K2, T, r, q, sigma) # Short Put
        
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
        return self.option_1.payoff(St) - self.option_2.payoff(St) - self.price


# Options at interception
S,K,T,r,q,sigma = 23.65,24,29/365,0.003,0,0.6951

option = straddle(S=23.65,K=24,T=29/365,r=0.03,q=0,sigma=0.7)
option = spread(S=23.65,K1=23.5,K2=24,T=29/365,r=0.03,q=0,sigma=0.7,option_type='put')

long_put = put(S,K,T,r,q,sigma)
long_call = call(S,K,T,r,q,sigma)




## Payoff diagram


def payoff_diagram(option, n):

    def montecarlo(option, n):
        # Only for long options currently
        sns.set_style('whitegrid')
        pct_chg = np.random.normal(1, option.sigma * option.T, n)
        ms_S = pct_chg * option.S
        ms_profit = option.profit(ms_S)
        return pd.DataFrame({'St' : ms_S, 'Profit' : ms_profit})
    
    df = montecarlo(option, n)
    t = np.arange(0,len(df)) # number of values on the x axis
    fig, ax = plt.subplots() # Create the plot objects
    ax.scatter(df['St'], df['Profit'],s = 5) # Input the data for the graph and plot
    ax.set_xlabel('St')
    ax.set_ylabel('Profit')

    plt.show()


payoff_diagram(spread(S=24.65,K1=22,K2=26,T=29/365,r=0.03,q=0,sigma=0.3,option_type='call'),n = 2500)