import pandas as pd
import numpy as np
import array

import numpy_financial as npf
from datetime import datetime
import matplotlib.pyplot as plt

GS = pd.read_csv("GME_stock.csv")  #Game stop prices'
GM = pd.read_csv("GS.csv")  #Goldman Sachs prices
JP = pd.read_csv("JPM.csv")  #JPM prices

print(GS.columns)
print(GM.columns)
print(JP.columns)
print(GS.shape)
print(GM.shape)
print(JP.shape)

Merged_Prices= GS.merge(GM, left_on='date', right_on='Date') \
    .merge(JP, on='Date', suffixes=('_GM','_JP'))

print(Merged_Prices.shape)
print(Merged_Prices.head())
print(Merged_Prices.shape)
print(Merged_Prices.columns)

# project1 = GS('close_price')
print(Merged_Prices['close_price'].std())
print(Merged_Prices['Close_GM'].std())
print(Merged_Prices['Close_JP'].std())

Cl_Pr_JP = np.array(Merged_Prices['Close_JP'])
Vol_JP = np.array(Merged_Prices['Volume_JP'])
Value_JP_mil = (Cl_Pr_JP * Vol_JP)/1000000
print(Vol_JP, Cl_Pr_JP, Value_JP_mil)


arr = np.array([1, 2, 3])
print(f'NumPy Array:\n{arr}')

list1 = arr.tolist()
print(f'List: {list1}')

Invest_Pr=-114
Inv = np.array(Invest_Pr)

arr = np.array(Merged_Prices['Close_JP'])
arr1 =arr + Inv
print(arr1)
print(f'NumPy Array:\n{arr1}')
list1 = arr1.tolist()
print(f'List: {list1}')
print(type(list1))
Invest_Pr=-114


list2=[Invest_Pr]
list2.extend(list1)
list3=list2[0:6]
#List2_filter = (list2 >= 100)
print(f'List for IRR calc: {list3}')
clc = npf.irr(list3)
print(clc)

#test =np.append(np.arr1,np.arr2)
#print(test)


#clc = np.irr(Cl_Pr_JP)
#print(clc)

#np.irr(proj)



#value= Merged_Prices['close_price']*
# abd = np.std(project1)
# print(abd)
