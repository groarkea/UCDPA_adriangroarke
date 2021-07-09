import datetime

import pandas as pd
import numpy as np
import numpy_financial as npf
import myCalcs as Cl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from numpy import cov

# Graph Theme
sns.set_theme(style="darkgrid")

# Import/Read stock price datasets
GS = pd.read_csv("Datasets/GME_stock.csv")  #Game stop prices'
GM = pd.read_csv("Datasets/GS.csv")  #Goldman Sachs prices
JP = pd.read_csv("Datasets/JPM.csv")  #JPM prices
AZ = pd.read_csv("Datasets/Amazon.com Inc.stock.csv")  #Amazon prices
FB = pd.read_csv("Datasets/Facebook Inc.stock.csv")  #Facebook prices
MS = pd.read_csv("Datasets/Microsoft Corporationstock.csv")  #Microsoft prices

# Add year, month and year-month columns to dataset based on pricing date
GS['year']= pd.DatetimeIndex(GS['date']).year
GM['year']= pd.DatetimeIndex(GM['Date']).year
JP['year']= pd.DatetimeIndex(JP['Date']).year
AZ['year']= pd.DatetimeIndex(AZ['Date']).year
FB['year']= pd.DatetimeIndex(FB['Date']).year
MS['year']= pd.DatetimeIndex(MS['Date']).year

GS['mth']= pd.DatetimeIndex(GS['date']).month
GM['mth']= pd.DatetimeIndex(GM['Date']).month
JP['mth']= pd.DatetimeIndex(JP['Date']).month
AZ['mth']= pd.DatetimeIndex(AZ['Date']).month
FB['mth']= pd.DatetimeIndex(FB['Date']).month
MS['mth']= pd.DatetimeIndex(MS['Date']).month


GS['yr_mth']= GS['year'].map(str) + '-' + GS['mth'].map(str)
GM['yr_mth']= GM['year'].map(str) + '-' + GM['mth'].map(str)
JP['yr_mth']= JP['year'].map(str) + '-' + JP['mth'].map(str)
AZ['yr_mth']= AZ['year'].map(str) + '-' + AZ['mth'].map(str)
FB['yr_mth']= FB['year'].map(str) + '-' + FB['mth'].map(str)
MS['yr_mth']= MS['year'].map(str) + '-' + MS['mth'].map(str)

#drop unwanted columns
GS.drop(columns=['open_price', 'high_price', 'low_price','adjclose_price'], axis=1, inplace=True)
GM.drop(columns=['High', 'Low', 'Open','Adj Close'], axis=1, inplace=True)
JP.drop(columns=['High', 'Low', 'Open','Adj Close'], axis=1, inplace=True)
AZ.drop(columns=['High', 'Low', 'Open','Adj Close','Company'], axis=1, inplace=True)
FB.drop(columns=['High', 'Low', 'Open','Adj Close','Company'], axis=1, inplace=True)
MS.drop(columns=['High', 'Low', 'Open','Adj Close','Company'], axis=1, inplace=True)

GS.rename(columns={'date':'Date','close_price':'Close','volume':'Volume'}, inplace=True)
print(f'Gamestop overview:{GS.head(3).transpose()}')
print(f'Columns of Gamestop are:{GS.columns}')

#Calculate trading volume value
GS['TradeValue'] = Cl.Vol_by_Pr(GS['Close'],GS['Volume'])
GM['TradeValue'] = Cl.Vol_by_Pr(GM['Close'],GM['Volume'])
JP['TradeValue'] = Cl.Vol_by_Pr(JP['Close'],JP['Volume'])
AZ['TradeValue'] = Cl.Vol_by_Pr(AZ['Close'],AZ['Volume'])
FB['TradeValue'] = Cl.Vol_by_Pr(FB['Close'],FB['Volume'])
MS['TradeValue'] = Cl.Vol_by_Pr(MS['Close'],MS['Volume'])

# Calculate 7 day percentage change for all stocks for price and volume

GS['Prct_chg_pr_7d'] = GS['Close'].pct_change(periods=-1).fillna(float(0)) # period set to minus 1 to ensure the % chg is based on the date before
GM['Prct_chg_pr_7d'] = GM['Close'].pct_change(periods=1).fillna(float(0)) # period set to minus 1 to ensure the % chg is based on the date before
JP['Prct_chg_pr_7d'] = JP['Close'].pct_change(periods=1).fillna(float(0)) # period set to minus 1 to ensure the % chg is based on the date before
AZ['Prct_chg_pr_7d'] = AZ['Close'].pct_change(periods=1).fillna(float(0)) # period set to minus 1 to ensure the % chg is based on the date before
FB['Prct_chg_pr_7d'] = FB['Close'].pct_change(periods=1).fillna(float(0)) # period set to plus 1 to ensure the % chg is based on the date before
MS['Prct_chg_pr_7d'] = MS['Close'].pct_change(periods=1).fillna(float(0)) # period set to plus 1 to ensure the % chg is based on the date before


print(GS.loc[7:10].transpose())
print(GM.loc[7:10].transpose())

GS['Prct_chg_vol_7d'] = GS['Volume'].pct_change(periods=7).fillna(float(0))
GM['Prct_chg_vol_7d'] = GM['Volume'].pct_change(periods=7).fillna(float(0))
JP['Prct_chg_vol_7d'] = JP['Volume'].pct_change(periods=7).fillna(float(0))
AZ['Prct_chg_vol_7d'] = AZ['Volume'].pct_change(periods=7).fillna(float(0))
FB['Prct_chg_vol_7d'] = FB['Volume'].pct_change(periods=7).fillna(float(0))
MS['Prct_chg_vol_7d'] = MS['Volume'].pct_change(periods=7).fillna(float(0))


# describe shape and column of individual datasets (Gamestop, JP Morgan, Microsoft
print(f'Shape of Gamestop is:{GS.shape}')
print(f'Columns of Gamestop are:{GS.columns}')
print(f'Shape of JP Morgan is:{JP.shape}')
print(f'Columns of JP Morgan are:{JP.columns}')
print(f'Shape of Microsoft is:{MS.shape}')
print(f'Columns of Microsoft are:{MS.columns}')

# Merge prices
Merged_Prices= GS.merge(GM, on=['Date','year','mth','yr_mth'], suffixes=('_GS','_GM')) \
     .merge(JP, on=['Date','year','mth','yr_mth'], suffixes=('_GM','_JP')) \
     .merge(AZ, on=['Date','year','mth','yr_mth'], suffixes=('_JP','_AZ')) \
     .merge(FB, on=['Date','year','mth','yr_mth'], suffixes=('_AZ','_FB')) \
     .merge(MS, on=['Date','year','mth','yr_mth'], suffixes=('_FB','_MS'))

Merged_Prices1= GS.merge(FB, on=['Date','year','mth','yr_mth'], suffixes=('_GS','')) \
     .merge(MS, on=['Date','year','mth','yr_mth'], suffixes=('_FB','_MS'))

print(Merged_Prices1.transpose())

# calculate total trading value across all 3 stocks
Merged_Prices['TotalValue(Bn)'] = Cl.All_stocks_Value(Merged_Prices['TradeValue_GS'],Merged_Prices['TradeValue_GM'],Merged_Prices['TradeValue_JP'])
a= Merged_Prices['TotalValue(Bn)'].sum()
print(f'Total Trading value in billions is:{a}')

#Describe Merged Prices
print(f'Merged data overview:{Merged_Prices.transpose()}')
print(f'Shape of Merged prices is:{Merged_Prices.shape}')
print(f'Columns of Merged prices are:{Merged_Prices.columns}')
print(f'Columns of Merged prices1 are:{Merged_Prices1.columns}')
# check null values
print(Merged_Prices.isna().sum())

# Visualisations
fig,ax = plt.subplots(3,2, sharex=True)
fig.set_size_inches([10, 6])
fig.suptitle('Historical average Prices', fontsize=16)

x = GS.groupby('year')['year'].median()
x1 = GM.groupby('year')['year'].median()
x2 = JP.groupby('year')['year'].median()
x3 = AZ.groupby('year')['year'].median()
x4 = FB.groupby('year')['year'].median()
x5 = MS.groupby('year')['year'].median()
y= GS.groupby('year')['Close'].mean()
y1= GM.groupby('year')['Close'].mean()
y2= JP.groupby('year')['Close'].mean()
y3= AZ.groupby('year')['Close'].mean()
y4= FB.groupby('year')['Close'].mean()
y5= MS.groupby('year')['Close'].mean()


ax[0,0].plot(x,y, marker="v", linestyle="dotted", color="r", label='GameStop')
ax[1,0].plot(x1,y1, marker="v", linestyle="--", color="b", label='Goldman Sachs')
ax[2,0].plot(x2,y2, marker="v", linestyle="--", color="b", label='JP Morgan')
ax[0,1].plot(x3,y3, marker="v", linestyle="--", color="b", label='Amazon')
ax[1,1].plot(x4,y4, marker="v", linestyle="--", color="b", label='Facebook')
ax[2,1].plot(x5,y5, marker="v", linestyle="--", color="b", label='Microsoft')

ax[0,0].set( ylabel='Price')
ax[2,1].set(xlabel='Year')

ax[0,0].legend(loc='upper left')
ax[1,0].legend(loc='upper left')
ax[2,0].legend(loc='best')
ax[0,1].legend(loc='best')
ax[1,1].legend(loc='best')
ax[2,1].legend(loc='best')
#fig.savefig("Stock_Prices.png")
plt.show()


# 1 line graph of prices
fig,ax = plt.subplots()
x = Merged_Prices1.groupby(['year'])['year'].max()
y1= Merged_Prices1.groupby(['year'])['Prct_chg_pr_7d_GS'].mean()
y2= Merged_Prices1.groupby(['year'])['Prct_chg_pr_7d_FB'].mean()
y3= Merged_Prices1.groupby(['year'])['Prct_chg_pr_7d_MS'].mean()

ax.plot(x,y1, marker="v", linestyle="dotted",  label='GameStop')
ax.plot(x,y2, marker="v", linestyle="--", label='Facebook')
ax.plot(x,y3, marker="v", linestyle="--", label='Micrcosoft')
ax.set(title='Mean Price', ylabel='Price', xlabel='Year')
ax.legend(loc='best')
#fig.savefig("Stock_Prices.png")
plt.show()



#2 bar graph of trade value
fig,ax = plt.subplots()
i = Merged_Prices.groupby('year')['year'].max()
h= Merged_Prices.groupby('year')['TotalValue(Bn)'].sum()
j1= Merged_Prices.groupby('year')['TradeValue_GS'].sum()
j2= Merged_Prices.groupby('year')['TradeValue_GM'].sum()
j3= Merged_Prices.groupby('year')['TradeValue_JP'].sum()
ax.bar(i,j1, label="GameStop")
ax.bar(i,j2, bottom=j1, label='Goldman Sachs')
ax.bar(i,j3, bottom=j1+j2, label='JP Morgan')
ax.plot(i,h)
ax.set(title='Total Trade Value Stacked', ylabel='Value in bns', xlabel='Year')
ax.legend(loc='best')
#plt.show()

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
#add linear regression line
X = q1.values.reshape(-1, 1)
Y = s1.values.reshape(-1, 1)
ln = LinearRegression()
ln.fit(X, Y)
Y_pred = ln.predict(X)
plt.plot(X, Y_pred, color='red')
#add linear regression line2
X1 = q2.values.reshape(-1, 1)
Y1 = s2.values.reshape(-1, 1)
ln1 = LinearRegression()
ln1.fit(X1, Y1)
Y_pred1 = ln1.predict(X1)
plt.plot(X1, Y_pred1, color='red')
#plt.show()

#3a seaborn plot of 1 stock price and volume (similar to above
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q2, y=s2, s=5, color=".15", label='JP Morgan')
sns.histplot(x=q2, y=s2, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q2, y=s2, levels=5, color="w", linewidths=1)
ax.set(title='Vol/Price Scatter plot', ylabel='Volume"', xlabel='Price')
ax.legend(loc='best')
#plt.show()

#4a seaborn scatter plot Change in pr / change in volume
No_Obs =1500
q = Merged_Prices['Prct_chg_pr_7d_GS'].tail(No_Obs)
s = Merged_Prices['Prct_chg_vol_7d_GS'].tail(No_Obs)
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
#plt.show()

#4b  GM
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q1, y=s1, s=5, color=".15", label='Goldman Sachs')
sns.histplot(x=q1, y=s1, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q1, y=s1, levels=5, color="w", linewidths=1)
ax.set(title='6mth Percentage Change/Price Scatter plot', ylabel='Percentage Change Price"', xlabel='Percentage Change Volume')
ax.legend(loc='best')
#plt.show()

# 5 scatter of high and low
fig, ax = plt.subplots()
No_Obs =1500
#q = Merged_Prices['low_price'].tail(No_Obs)
#s = Merged_Prices['high_price'].tail(No_Obs)
#q1 = Merged_Prices['Low_GM'].tail(No_Obs)
#s1 = Merged_Prices['High_GM'].tail(No_Obs)
q2 = Merged_Prices['Close_JP'].tail(No_Obs)
s2 = Merged_Prices['Close_MS'].tail(No_Obs)
#ax.scatter(q, s, label='GameStop', alpha=0.5)
#ax.scatter(q1, s1, label='Goldman', alpha=0.5, color='y')
ax.scatter(q2, s2, label='JP Morgan', alpha=0.5)
ax.set(title='Vol/Price Scatter plot', ylabel='high"', xlabel='Prices')
X = q2.values.reshape(-1, 1)
Y = s2.values.reshape(-1, 1)
ln = LinearRegression()
ln.fit(X, Y)
Y_pred = ln.predict(X)
plt.plot(X, Y_pred, color='red')
#plt.show()
r_sq=ln.score(X,Y)
print('coefficient of determination:', r_sq)
print('slope:', ln.coef_)


#5a
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=q2, y=s2, s=5, color=".15", label='JP Morgan')
sns.histplot(x=q2, y=s2, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q2, y=s2, levels=5, color="w", linewidths=1)
ax.set(title='Vol/Price Scatter plot', ylabel='high"', xlabel='Price')
ax.legend(loc='best')
#plt.show()


# Calculate volatility of Stocks
volatility_GS =Merged_Prices['Close_GS'].std()
volatility_GM = Merged_Prices['Close_GM'].std()
volatility_JP = Merged_Prices['Close_JP'].std()
x = np.array(Merged_Prices['Close_GS'],Merged_Prices['Close_GM'])
CoVar = np.cov(x)
Corr = np.corrcoef(x)
print(f'Covariance: {CoVar}')
print(f'Correlation: {Corr}')
print(f'Volatility of GameStop Stock: {volatility_GS}')
print(f'Volatility of Goldman Stock: {volatility_GM}')
print(f'Volatility of JP Morgan Stock: {volatility_JP}')

# calculate portfolio variance

cov_matrix = cov(Merged_Prices['Prct_chg_pr_7d_GM'],Merged_Prices['Prct_chg_pr_7d_JP'])
weights= np.array([0.5,0.5])
port_variance = np.dot(weights.T,np.dot(cov_matrix,weights))
GM_Std= Merged_Prices['Prct_chg_pr_7d_GM'].std()
JP_Std= Merged_Prices['Prct_chg_pr_7d_JP'].std()
port_variance_frmt = (str(np.round(port_variance, 3) * 100) + '%')
port_std=np.sqrt(port_variance)
GM_JP_Std = Cl.PC_Format(port_std)

print(f'Portfolio variance: {port_variance}')
print(f'Portfolio variance: {port_variance_frmt}')
print(f'Portfolio Covariance: {cov_matrix}')
print(f'GM Std: {GM_Std}')
print(f'JP Std: {JP_Std}')
print(f'GM_JP Std: {GM_JP_Std}')

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







