import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as si
from scipy.stats import norm
import random
import matplotlib.pyplot as plt


# avg = 1
# std_dev = .1
# num_reps = 500
# num_simulations = 1000

# pct_to_target = np.random.normal(avg, std_dev, num_reps).round(2)



class black_scholes:

    def __init__(self,S, K, T, r, q, sigma):
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
        print("Price: " + str("{:.2f}".format(self.price)))
        print("Delta: " + str("{:.2f}".format(self.delta)))
        print("Gamma: " + str("{:.2f}".format(self.gamma)) + " (per % of Spot)")
        print("Vega: " + str("{:.2f}".format(self.vega)) + " (per % of IV)")
        print("Theta: " + str("{:.2f}".format(self.theta)) + " (daily)")
        print("Rho: " + str("{:.2f}".format(self.rho)) + " (per % of Interest Rate)")


class call(black_scholes):
    def __init__(self,S, K, T, r, q, sigma):
        super().__init__(S, K, T, r, q, sigma)
        self.price = black_scholes.price(self,'call')
        self.delta = black_scholes.delta(self,'call')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'call')
        self.rho = black_scholes.rho(self,'call')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)

class put(black_scholes):
    def __init__(self,S, K, T, r, q, sigma):
        super().__init__(S, K, T, r, q, sigma)
        self.price = black_scholes.price(self,'put')
        self.delta = black_scholes.delta(self,'put')
        self.gamma = black_scholes.gamma(self)
        self.vega = black_scholes.vega(self)
        self.theta = black_scholes.theta(self,'put')
        self.rho = black_scholes.rho(self,'put')
        self.leverage = black_scholes.leverage(self)
        self.exposure = black_scholes.exposure(self)

# Options at interception
S,K,T,r,q,sigma = 25089.17,25000,75/365,0.0001,0.04,0.2605

call(S,K,T,r,q,sigma).calculate()

long_put_0 = put(S,K,T,r,q,sigma)
short_call_0 = call(S,K,T,r,q,sigma)
short_call_0.calculate()
port_value_0 = short_call_0.price - long_put_0.price

port_exposure = 

# Options at T > 0
S1,T1 = 24000,30/365

long_put_1 = put(S1,K,T1,r,q,sigma)
short_call_1 = call(S1,K,T1,r,q,sigma)

port_value_1 = short_call_1.price - long_put_1.price

# Calculate Profit
profit = port_value_1 - port_value_0

