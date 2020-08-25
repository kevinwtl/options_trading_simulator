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



class black_scholes():

    def __init__(self,S, K, T, r, q, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.call = self.call()
        self.put = self.put()
        self.call_delta = self.delta('call')
        self.put_delta = self.delta('put')
        self.gamma = self.gamma()
        self.vega = self.vega()
        self.call_theta = self.theta('call')
        self.put_theta = self.theta('put')
        self.call_rho = self.rho('call')
        self.put_rho = self.rho('put')


    def call(self):
        call = (self.S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0))
        return call

    def put(self):
        put = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - self.S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1, 0.0, 1.0))
        return put

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





black_scholes(25182.15,25200,1/12,0.0001,0.04,0.26).call_delta


