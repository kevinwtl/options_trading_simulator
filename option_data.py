import os
os.chdir('/Users/tinglam/Documents/GitHub/value_investing')
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
import re


date = '200902'
code = 'MIU'

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
    
    
    
    return df

df = option_data(date,code)

df


import calendar
abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}

abbr_to_num['Sep']

'SEP'.capitalize()






