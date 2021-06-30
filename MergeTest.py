import pandas as pd
import numpy as np
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
clc = npf.irr(-100, Cl_Pr_JP)
print(clc)


#clc = np.irr(Cl_Pr_JP)
#print(clc)

#np.irr(proj)



#value= Merged_Prices['close_price']*
# abd = np.std(project1)
# print(abd)
