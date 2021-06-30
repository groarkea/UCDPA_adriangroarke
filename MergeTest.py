import pandas as pd

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


