# Option Payoff Simulator

## Introduction
This repo uses Black-Scholes Model as the foundation.
It generates payoff diagrams and evaluates strategies using Monte Carlo Simulation.

## Examples
```
import bsm

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
```

![example](https://tva1.sinaimg.cn/large/007S8ZIlgy1gijbr70xv1j30bx0lf75j.jpg)

## Disclaimer
This repository is intended for educational purposes only, and should not rely solely on this repo for making investment decisions.
