import pandas as pd
import numpy as np
import numpy_financial as npf
import myCalcs as Cl

import matplotlib.pyplot as plt

# Import/Read 3 stock price datasets
GS = pd.read_csv("GME_stock.csv")  #Game stop prices'
GM = pd.read_csv("GS.csv")  #Goldman Sachs prices
JP = pd.read_csv("JPM.csv")  #JPM prices

# check null values
print(JP.isna().sum())

# Add year column to dataset based on pricing date
GS['year']= pd.DatetimeIndex(GS['date']).year
GM['year']= pd.DatetimeIndex(GM['Date']).year
JP['year']= pd.DatetimeIndex(JP['Date']).year
#Calculate trading volume value
GS['tradeValue'] = Cl.Vol_by_Pr(GS['close_price'],GS['volume'])
GM['TradeValue'] = Cl.Vol_by_Pr(GM['Close'],GM['Volume'])
JP['TradeValue'] = Cl.Vol_by_Pr(JP['Close'],JP['Volume'])

# describe shape and column of individual datasets
print(f'Shape of Gamestop is:{GS.shape}')
print(f'Columns of Gamestop is:{GS.columns}')
print(f'Shape of JP Morgan is:{JP.shape}')
print(f'Columns of JP Morgan are:{JP.columns}')

#Merage 3 data sets on date
Merged_Prices= GS.merge(GM, left_on='date', right_on='Date', suffixes=('_GS','_GM')) \
    .merge(JP, on='Date', suffixes=('_GM','_JP'))
print(Merged_Prices.isna().sum())

#Describe Merged Prices
print(f'Describe:{Merged_Prices}')
print(f'Shape of Merged prices is:{Merged_Prices.shape}')
print(f'Columns of Merged prices are:{Merged_Prices.columns}')

# calculate total trading value across all 3 stocks
Merged_Prices['TotalValue'] = Cl.All_stocks_Value(Merged_Prices['tradeValue'],Merged_Prices['TradeValue_GM'],Merged_Prices['TradeValue_JP'])
print(Merged_Prices['TotalValue'].head())
a= Merged_Prices['TotalValue'].sum()
print(f'Total Trading value is:{a}')
print(Merged_Prices.shape)
print(Merged_Prices.describe)
print(f'Columns of Merged prices are:{Merged_Prices.columns}')

# Visualisation
fig,ax = plt.subplots()
x = Merged_Prices.groupby('year')['year'].mean()
y= Merged_Prices.groupby('year')['close_price'].mean()
y1= Merged_Prices.groupby('year')['Close_GM'].mean()
y2= Merged_Prices.groupby('year')['Close_JP'].mean()
ax.plot(x,y, marker="v", linestyle="dotted", color="r", label='GameStop')
ax.plot(x,y1, marker="v", linestyle="--", color="b", label='Goldman Sachs')
ax.plot(x,y2, marker="v", linestyle="--", color="g", label='JP Morgan')
ax.set(title='Mean Price', ylabel='Price', xlabel='Year')
ax.legend(loc='best')
#ax.set_xlabel("Time (months)")
plt.show()


fig,ax = plt.subplots()
g = Merged_Prices.groupby('year')['year'].max()
h= Merged_Prices.groupby('year')['TotalValue'].sum()
ax.bar(g,h)
ax.set(title='Total Trade Value', ylabel='Value in bns', xlabel='Year')
ax.legend(loc='best')
plt.show()

fig,ax = plt.subplots()
i = Merged_Prices.groupby('year')['year'].max()
h= Merged_Prices.groupby('year')['TotalValue'].sum()
j1= Merged_Prices.groupby('year')['tradeValue'].sum()
j2= Merged_Prices.groupby('year')['TradeValue_GM'].sum()
j3= Merged_Prices.groupby('year')['TradeValue_JP'].sum()
ax.bar(i,j1, label="GameStop")
ax.bar(i,j2, bottom=j1, label='Goldman Sachs')
ax.bar(i,j3, bottom=j1+j2, label='JP Morgan')
ax.plot(i,h)
ax.set(title='Total Trade Value Stacked', ylabel='Value in bns', xlabel='Year')
ax.legend(loc='best')
plt.show()

fig, ax = plt.subplots()
q = Merged_Prices.groupby('year')['close_price'].mean()
s = Merged_Prices.groupby('year')['volume'].mean()
q1 = Merged_Prices.groupby('year')['Close_GM'].mean()
s1 = Merged_Prices.groupby('year')['Volume_GM'].mean()
q2 = Merged_Prices.groupby('year')['Close_JP'].mean()
s2 = Merged_Prices.groupby('year')['Volume_JP'].mean()
ax.scatter(q, s)
ax.scatter(q1, s1)
ax.scatter(q2, s2)
ax.set_xlabel("Mean Prices")
ax.set_ylabel("Mean Volume")
plt.show()

fig, ax = plt.subplots()
q = Merged_Prices['close_price']
s = Merged_Prices['volume']
q1 = Merged_Prices['Close_GM']
s1 = Merged_Prices['Volume_GM']
q2 = Merged_Prices['Close_JP']
s2 = Merged_Prices['Volume_JP']
ax.scatter(q, s)
ax.scatter(q1, s1)
ax.scatter(q2, s2)
ax.set_xlabel("Mean Prices")
ax.set_ylabel("Mean Volume")
plt.show()

# Calculate volatility of Stocks
volatility_GS =Merged_Prices['close_price'].std()
volatility_GM = Merged_Prices['Close_GM'].std()
volatility_JP = Merged_Prices['Close_JP'].std()
x = np.array(Merged_Prices['close_price'],Merged_Prices['Close_GM'])
CoVar = np.cov(x)
Corr = np.corrcoef(x)
print(f'Covariance: {CoVar}')
print(f'Correlation: {Corr}')
print(f'Volatility of GameStop Stock: {volatility_GS}')
print(f'Volatility of Goldman Stock: {volatility_GM}')
print(f'Volatility of JP Morgan Stock: {volatility_JP}')

#Group mean prices annually
group_merged_prices = Merged_Prices.groupby('year')['Close_JP'].mean()
print(group_merged_prices)


# Investor has option to buy stock each year.
# If average annual price is above the strike price, the option holder makes a return (Average price minus the strike price).
# THe cost of purchasing the option is 20 and entitle es holder to buy shares every year.
# The strike price is 40.
# The question is what would be the return since 2006 and secondly, at what point does the holder make a positive return.
Opt_strike=[-40]
arr = np.array(group_merged_prices) #arr is the monthly mean prices.
arr1 =arr + Opt_strike # arr1 is the monthly return (mean price - Strike).
boolean_filter = arr1 > 0 # Filter is used to eliminate negative returns as option won't be executed that year.
arr2 = boolean_filter * arr1 # arr2 is the positive returns only.


print(f'NumPy Array type is:\n{type(arr2)}')
list1 = arr2.tolist() # converts Numpy array to list
print(f'List of positive returns: {list1}') # list1 is the positive returns only.

Opt_cost=-20

list2=[Opt_cost] # lst 1 is the option cost
list2.extend(list1) # list 2 extends list 1 so there is a list with the initial cost and annual positive returns.

print(f'List for IRR calc: {list2}')

clc = npf.irr(list2)
clc2 = npf.irr(list2[0:9])
print(f'Stock Option return since 2006: {round(clc,2)}') # Investor has option to buy stock each month.
print(f'Stock Option return from 2006 to 2013: {round(clc2,2)}')



print(Merged_Prices.groupby('year')['TotalValue'].sum())
print(Merged_Prices.groupby('year')['tradeValue'].sum())
print(Merged_Prices.groupby('year')['TradeValue_GM'].sum())
print(Merged_Prices.groupby('year')['TradeValue_JP'].sum())






