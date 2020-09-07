import os
os.chdir('/Users/tinglam/Documents/GitHub/value_investing')
import pandas as pd
from pandas.tseries.offsets import BMonthEnd
from bs4 import BeautifulSoup
import requests
import numpy as np
import re
import datetime


date = '200904'
code = 'TCH'

def option_data(date,code):
    url = 'https://www.hkex.com.hk/eng/stat/dmstat/dayrpt/dqe{}.htm'.format(date)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text,'lxml')

    table = soup.find('a',{'name':code}).text

    table = re.split(r'\s{2,}', table)[23:]

    call_list = table[:table.index('TOTAL CALL')]
    put_list = table[table.index('TOTAL CALL')+4:table.index('TOTAL PUT')]

    my_list = []
    for row in range(int(len(call_list)/11)):
        i = row * 11
        l = call_list[i:i+11]
        my_list.append(l)

    for row in range(int(len(put_list)/11)):
        i = row * 11
        l = put_list[i:i+11]
        my_list.append(l)

    headers = ['Expiry','Contract','Opening Price','Daily High','Daily Low','Settlement Price','Chainge in Settlement Price','IV','Volume','OI','Change in OI']
    df = pd.DataFrame(my_list,columns = headers)
    
    # Data Procesing
    
    df['Contract Type'] = df['Contract'].str[-1]
    df['Strike'] = df['Contract'].str[:-2].astype(float)

    df['Expiry Year'] = ('20' + df['Expiry'].str[-2:]).astype(int)
    df['Expiry Month'] = df['Expiry'].str[:3].apply(lambda x: datetime.datetime.strptime(x, "%b").month)
    func = lambda x: BMonthEnd().rollforward(datetime.datetime(x['Expiry Year'],x['Expiry Month'],1))
    df['Expiry Date'] = df.apply(func,axis=1)
    df['T'] = (df['Expiry Date'] - datetime.datetime.today()).astype('timedelta64[D]')
    
    return df

df = option_data(date,code)


def IV_implied(df,S,K,expiry='SEP20'):
    # Currently use Call prices to estimate IV only
    temp_df = df[(df['Strike'] == K) & (df['Expiry'] == expiry)]
    T = temp_df['T'].iloc[0]
    C = temp_df[temp_df['Contract Type'] == 'C']['Settlement Price']
    P = temp_df[temp_df['Contract Type'] == 'P']['Settlement Price']
    
    IV = np.sqrt(2*np.pi/(T/365)) * float(C) / S
    
    return IV


IV_implied(df,515,520,'SEP20')