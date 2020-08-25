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
        #self.call = (self.S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0))
        #self.put = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - self.S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1, 0.0, 1.0))

    def call(self):
        call = (self.S * np.exp(-self.q * self.T) * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0))
        return call

    def put(self):
        put = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - self.S * np.exp(-self.q * self.T) * si.norm.cdf(-self.d1, 0.0, 1.0))
        return put

    def delta(self,option_type):
        if option_type == 'call':
            delta = si.norm.cdf(self.d1, 0.0, 1.0)
        if option_type == 'put':
            delta = -si.norm.cdf(-self.d1, 0.0, 1.0)
        return delta

    # def theta(self,Type):
    #     df = np.exp(-(self.r * self.T))
    #     dfq = np.exp((-self.q * self.T))
    #     tmptheta = (1/ 365) \
    #         * (-0.5 * self.S * dfq * norm.pdf(self.d1) * \
    #            self.sigma / (self.T ** 0.5) + \
	#         Type * (self.q * self.S * dfq * norm.cdf(Type * self.d1) \
    #         - self.r * self.K * df * norm.cdf(Type * self.d2)))
    #     return tmptheta
    def theta(self,option_type):
        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-self.d1 ** 2 * 0.5)
        if option_type == 'call':
            theta = (1/365) * (-self.sigma * self.S * prob_density) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2, 0.0, 1.0)
        if option_type == 'put':    
            theta = (1/365) * (-self.sigma * self.S * prob_density) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0)
        return theta





print(black_scholes(25182.15,25200,1/12,0.0001,0.04,0.26).theta('put'))



