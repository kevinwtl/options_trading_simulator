# Option Payoff Simulator

## Disclaimer
This repository is intended for educational purposes only, and should not rely solely on this repo for making investment decisions.

## Examples
First Example is a call option of 700.HK 
Spot Price = 530; Stike Price = 540; Time to expiry = 29 days; Risk-free rate = 3%; Dividend = 0%; Volatility = 48%

```
example_1 = call(S=530,K=540,T=29/365,r=0.03,q=0,sigma=0.48)
example_1.calculate()
payoff_diagram(example_1,n = 2500)
```
![example_1](https://tva1.sinaimg.cn/large/007S8ZIlgy1gia6ncawdfj30b30b60td.jpg)
---
Second Example is a straddle created for 1810.HK
Spot Price = 23; Stike Price = 22; Time to expiry = 29 days; Risk-free rate = 3%; Dividend = 0%; Volatility = 70%

```
example_2 = straddle(S=23.65,K=22,T=29/365,r=0.03,q=0,sigma=0.7)
example_2.calculate()
payoff_diagram(example_2,n = 2500)
```
![example_2](https://tva1.sinaimg.cn/large/007S8ZIlgy1gia6mt89s6j30au0b10td.jpg)
---
Third Example is a Bull Call Spread created also for 1810.HK
Two Strike Prices will be 22 and 24. It will be recognized as a Bull Spread if K2 > K1.

```
example_3 = spread(S=23.65,K1=22,K2=24,T=29/365,r=0.03,q=0,sigma=0.7,option_type='call')
example_3.calculate()
payoff_diagram(example_3,n = 2500)
```
![example_3](https://tva1.sinaimg.cn/large/007S8ZIlgy1gia6pehantj30bb0b5q3n.jpg)
