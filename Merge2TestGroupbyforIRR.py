import pandas as pd
import numpy as np
import numpy_financial as npf
import myCalcs as Cl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set_theme(style="darkgrid")
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

# Calculate percentage change for all 3 stocks

GS['prct_chg_pr_7d'] = GS['close_price'].pct_change(periods=7).fillna(float(0))
GM['Prct_chg_pr_7d'] = GM['Close'].pct_change(periods=7).fillna(float(0))
JP['Prct_chg_pr_7d'] = JP['Close'].pct_change(periods=7).fillna(float(0))

GS['prct_chg_vol_7d'] = GS['volume'].pct_change(periods=7).fillna(float(0))
GM['Prct_chg_vol_7d'] = GM['Volume'].pct_change(periods=7).fillna(float(0))
JP['Prct_chg_vol_7d'] = JP['Volume'].pct_change(periods=7).fillna(float(0))

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
# 1 line graph of prices
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
plt.show()



#2 bar graph of trade value
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

#3 scatter plt all 3 close prices and volume
fig, ax = plt.subplots()
No_Obs =1500
#q = Merged_Prices['close_price'].tail(No_Obs)
#s = Merged_Prices['volume'].tail(No_Obs)
q1 = Merged_Prices['Close_GM'].tail(No_Obs)
s1 = Merged_Prices['Volume_GM'].tail(No_Obs)
q2 = Merged_Prices['Close_JP'].tail(No_Obs)
s2 = Merged_Prices['Volume_JP'].tail(No_Obs)
#ax.scatter(q, s, label='GameStop', alpha=0.5)
ax.scatter(q1, s1, label='Goldman', alpha=0.5, color='y')
ax.scatter(q2, s2, label='JP Morgan', alpha=0.5)
ax.set(title='Vol/Price Scatter plot', ylabel='Volume"', xlabel='Prices')

plt.show()

#3a seaborn plot of 1 stock price and volume (similar to above
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q2, y=s2, s=5, color=".15", label='JP Morgan')
sns.histplot(x=q2, y=s2, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q2, y=s2, levels=5, color="w", linewidths=1)
ax.set(title='Vol/Price Scatter plot', ylabel='Volume"', xlabel='Price')
ax.legend(loc='best')
plt.show()

#4a seaborn scatter plot Change in pr / change in volume
No_Obs =1500
q = Merged_Prices['prct_chg_pr_7d'].tail(No_Obs)
s = Merged_Prices['prct_chg_vol_7d'].tail(No_Obs)
q1 = Merged_Prices['Prct_chg_pr_7d_GM'].tail(No_Obs)
s1 = Merged_Prices['Prct_chg_vol_7d_GM'].tail(No_Obs)
q2 = Merged_Prices['Prct_chg_pr_7d_JP'].tail(No_Obs)
s2 = Merged_Prices['Prct_chg_vol_7d_JP'].tail(No_Obs)

# 4a JP
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q2, y=s2, s=5, color=".15", label='JP Morgan')
sns.histplot(x=q2, y=s2, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q2, y=s2, levels=5, color="w", linewidths=1)
ax.set(title='6mth Percentage Change/Price Scatter plot', ylabel='Percentage Change Price"', xlabel='Percentage Change Volume')
ax.legend(loc='best')
plt.show()

#4b  GM
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q1, y=s1, s=5, color=".15", label='Goldman Sachs')
sns.histplot(x=q1, y=s1, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q1, y=s1, levels=5, color="w", linewidths=1)
ax.set(title='6mth Percentage Change/Price Scatter plot', ylabel='Percentage Change Price"', xlabel='Percentage Change Volume')
ax.legend(loc='best')
plt.show()

# 5 scatter of high and low
fig, ax = plt.subplots()
No_Obs =1500
#q = Merged_Prices['low_price'].tail(No_Obs)
#s = Merged_Prices['high_price'].tail(No_Obs)
#q1 = Merged_Prices['Low_GM'].tail(No_Obs)
#s1 = Merged_Prices['High_GM'].tail(No_Obs)
q2 = Merged_Prices['Low_JP'].tail(No_Obs)
s2 = Merged_Prices['High_JP'].tail(No_Obs)
#ax.scatter(q, s, label='GameStop', alpha=0.5)
#ax.scatter(q1, s1, label='Goldman', alpha=0.5, color='y')
ax.scatter(q2, s2, label='JP Morgan', alpha=0.5)
ax.set(title='Vol/Price Scatter plot', ylabel='high"', xlabel='Prices')
X = q2.values.reshape(-1, 1)
Y = s2.values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
plt.plot(X, Y_pred, color='red')
plt.show()

#5a
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q2, y=s2, s=5, color=".15", label='JP Morgan')
sns.histplot(x=q2, y=s2, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q2, y=s2, levels=5, color="w", linewidths=1)
ax.set(title='Vol/Price Scatter plot', ylabel='high"', xlabel='Price')
ax.legend(loc='best')
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

#print(Merged_Prices['PercentChange_JP'])


print(q2, s2)

