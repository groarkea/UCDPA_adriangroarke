import pandas as pd
import datetime
import numpy as np
import array

import numpy_financial as npf
from datetime import datetime
import matplotlib.pyplot as plt

GS = pd.read_csv("GME_stock.csv")  #Game stop prices'
GM = pd.read_csv("GS.csv")  #Goldman Sachs prices
JP = pd.read_csv("JPM.csv")  #JPM prices

GS['year']= pd.DatetimeIndex(GS['date']).year
GM['year']= pd.DatetimeIndex(GM['Date']).year
JP['year']= pd.DatetimeIndex(JP['Date']).year

#print(GS.columns)
#print(GM.columns)
print(JP.columns)
#print(GS.shape)
#print(GM.shape)
#print(JP.shape)

Merged_Prices= GS.merge(GM, left_on='date', right_on='Date') \
    .merge(JP, on='Date', suffixes=('_GM','_JP'))

#print(Merged_Prices.shape)
print(Merged_Prices)
print(Merged_Prices.shape)
print(Merged_Prices.columns)
#print(Merged_Prices.describe())
#print(Merged_Prices.values.transpose())

print(Merged_Prices['close_price'].std())
print(Merged_Prices['Close_GM'].std())
print(Merged_Prices['Close_JP'].std())

group_merged_prices = Merged_Prices.groupby('year')['Close_JP'].mean()
print(group_merged_prices)


# Cl_Pr_JP = np.array(Merged_Prices['Close_JP'])
# Vol_JP = np.array(Merged_Prices['Volume_JP'])
#Value_JP_mil = (Cl_Pr_JP * Vol_JP)/1000000
#print(Vol_JP, Cl_Pr_JP, Value_JP_mil)


#arr = np.array([1, 2, 3])
#print(f'NumPy Array:\n{arr}')

#list1 = arr.tolist()
#print(f'List: {list1}')

Opt_strike=[-40]
Inv = np.array(Opt_strike)
print(Opt_strike)
arr = np.array(group_merged_prices)
print(arr)
arr1 =arr + Opt_strike
boolean_filter = arr1 > 0

print(arr1)
print(boolean_filter)
arr2 = (boolean_filter * arr1)*1
print(arr2)
print(type(arr2))
print(f'NumPy Array:\n{arr2}')
list1 = arr2.tolist()
print(f'List: {list1}')
print(type(list1))

Opt_cost=-20


list2=[Opt_cost]
print(list2)
list2.extend(list1)

#list3=list2[0:6]

print(f'List for IRR calc: {list2}')
clc = npf.irr(list2)
print(clc)







