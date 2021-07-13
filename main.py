
import requests
import pandas as pd
import numpy as np
import numpy_financial as npf


import myCalcs as Cl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from numpy import cov

#data for Gamestop to create dataframe from Dictionary
data=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=GME&apikey=RVZ3KJWM4Q6RNVCJ')
print(data.text)
print(type(data))
parsed_data=data.json()
print(type(parsed_data))
output="GameStop's 52 Week high is " + parsed_data['52WeekHigh'] + " and GameStop's 52 Week low is " + parsed_data['52WeekLow']
print(output)

#data for JPM to create dataframe from Dictionary
dataJP=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
print(dataJP.text)
print(type(dataJP))
parsed_dataJP=dataJP.json()
print(type(parsed_dataJP))
outputJP="JP Morgan's 52 Week high is " + parsed_dataJP['52WeekHigh'] + " and JP Morgan's 52 Week low is " + parsed_dataJP['52WeekLow']
print(outputJP)


#create dictionary
a=parsed_data['Symbol']
b=parsed_data['52WeekHigh']
c=parsed_data['52WeekLow']
a1=parsed_dataJP['Symbol']
b1=parsed_dataJP['52WeekHigh']
c1=parsed_dataJP['52WeekLow']
my_dictnry={'ID':[a,a1],'52High':[b,b1],'52Low':[c,c1]}
print(my_dictnry)
print(type(my_dictnry))

#create dataframe
my_df= pd.DataFrame(my_dictnry)
print(my_df)
print(my_df['52High'])

# API for JP Morgan description and EPS
data4=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
print(data4.text)
print(type(data4))
parsed_data4=data4.json()
print(type(parsed_data4))
output=parsed_data4['Description'] + "' The company's earnings per share is '" + parsed_data4['EPS']
print(output)

# API for JP Morgan total assets reported annually
data5=requests.get('https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
#print(data5.text)
print(type(data5))
parsed_data5 = data5.json()
#print(type(parsed_data5))

parsed_data5a = parsed_data5['annualReports']
print(parsed_data5a)

for temp_variable in parsed_data5a:
    x = np.array(temp_variable['fiscalDateEnding'])
    y = np.array(temp_variable['totalAssets'])
    print(x,y)



# Graph Theme
sns.set_theme(style="darkgrid")

# Import/Read stock price datasets
GS = pd.read_csv("Datasets/GME_stock.csv")  #Game stop prices'
GS = GS.sort_values('date', ascending=True)
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
print(f'FB overview:{FB.head(3).transpose()}')
print(f'Columns of Gamestop are:{GS.columns}')


#Calculate trading volume value
GS['TradeValue'] = Cl.Vol_by_Pr(GS['Close'],GS['Volume'])
GM['TradeValue'] = Cl.Vol_by_Pr(GM['Close'],GM['Volume'])
JP['TradeValue'] = Cl.Vol_by_Pr(JP['Close'],JP['Volume'])
AZ['TradeValue'] = Cl.Vol_by_Pr(AZ['Close'],AZ['Volume'])
FB['TradeValue'] = Cl.Vol_by_Pr(FB['Close'],FB['Volume'])
MS['TradeValue'] = Cl.Vol_by_Pr(MS['Close'],MS['Volume'])

# Calculate 7 day percentage change for all stocks for price and volume

GS['Prct_chg_pr_7d'] = GS['Close'].pct_change(periods=7).fillna(float(0)) # check not -1 for preceeding day
GM['Prct_chg_pr_7d'] = GM['Close'].pct_change(periods=7).fillna(float(0))
JP['Prct_chg_pr_7d'] = JP['Close'].pct_change(periods=7).fillna(float(0))
AZ['Prct_chg_pr_7d'] = AZ['Close'].pct_change(periods=7).fillna(float(0))
FB['Prct_chg_pr_7d'] = FB['Close'].pct_change(periods=7).fillna(float(0))
MS['Prct_chg_pr_7d'] = MS['Close'].pct_change(periods=7).fillna(float(0))

GS['Prct_chg_pr_1d'] = GS['Close'].pct_change(periods=1).fillna(float(0))
GM['Prct_chg_pr_1d'] = GM['Close'].pct_change(periods=1).fillna(float(0))
JP['Prct_chg_pr_1d'] = JP['Close'].pct_change(periods=1).fillna(float(0))
AZ['Prct_chg_pr_1d'] = AZ['Close'].pct_change(periods=1).fillna(float(0))
FB['Prct_chg_pr_1d'] = FB['Close'].pct_change(periods=1).fillna(float(0))
MS['Prct_chg_pr_1d'] = MS['Close'].pct_change(periods=1).fillna(float(0))

print(GS.loc[10:7].transpose())
print(GM.loc[7:10].transpose())
print(FB.loc[2177:2181].transpose())

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

Merged_Prices1= GS.merge(FB, on=['Date','year','mth','yr_mth'], suffixes=('_GS','_FB')) \
     .merge(MS, on=['Date','year','mth','yr_mth'], suffixes=('_FB','_MS')) \
     .merge(AZ, on=['Date','year','mth','yr_mth'], suffixes=('_MS','_AZ'))
print(Merged_Prices1.transpose())

# calculate total trading value across all 3 stocks
Merged_Prices['TotalValue(Bn)'] = Cl.All_stocks_Value(Merged_Prices['TradeValue_GS'],\
                                                      Merged_Prices['TradeValue_GM'],Merged_Prices['TradeValue_JP'],\
                                                      Merged_Prices['TradeValue_AZ'],Merged_Prices['TradeValue_FB'],\
                                                      Merged_Prices['TradeValue_MS'])
a= Merged_Prices['TotalValue(Bn)'].sum()
print(f'Total Trading value in billions is:{a}')

#Describe Merged Prices
print(f'Merged data overview:{Merged_Prices.transpose()}')
print(f'Shape of Merged prices is:{Merged_Prices.shape}')
print(f'Columns of Merged prices are:{Merged_Prices.columns}')
print(f'Columns of Merged prices1 are:{Merged_Prices1.columns}')
print(f'Shape of Merged prices1 is:{Merged_Prices1.shape}')
print(f'Merged data 1 overview:{Merged_Prices1.transpose()}')
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
fig.savefig("Charts/Historical Stock_Prices.png")
plt.show()


# 1 line graph of prices

ref_date = '2020-11-30'
mask_dt = Merged_Prices1['Date'] > ref_date

Merged_Prices_masked = Merged_Prices1.loc[mask_dt]
print(Merged_Prices_masked['Close_GS'].tail())
print(Merged_Prices_masked['Prct_chg_pr_7d_GS'].tail())

fig,ax = plt.subplots()
fig.set_size_inches([8, 5])
x0= Merged_Prices_masked['Date']
y0= Merged_Prices_masked['Prct_chg_pr_7d_GS']
y1= Merged_Prices_masked['Prct_chg_pr_7d_FB']
y2= Merged_Prices_masked['Prct_chg_pr_7d_MS']
y3= Merged_Prices_masked['Prct_chg_pr_7d_AZ']
ax.plot(x0,y0, marker="v", linestyle="dotted",  label='GameStop')
ax.plot(x0,y1, marker="v", linestyle="dotted",  label='Facebook')
ax.plot(x0,y2, marker="v", linestyle="dotted",  label='Microsoft')
ax.plot(x0,y3, marker="v", linestyle="dotted",  label='Amazon')
ax.set(title='7 day percentage price change', ylabel='Percentage Change in Price', xlabel='Date')
ax.set_xticklabels(x0, rotation=90, size=12)
ax.legend(loc='best')
fig.savefig("Charts/7 day percentage price changes.png")
plt.show()







#2 bar graph of trade value
fig,ax = plt.subplots()
i = Merged_Prices.groupby('year')['year'].max()
h= Merged_Prices.groupby('year')['TotalValue(Bn)'].sum()
j1= Merged_Prices.groupby('year')['TradeValue_GS'].sum()
j2= Merged_Prices.groupby('year')['TradeValue_GM'].sum()
j3= Merged_Prices.groupby('year')['TradeValue_JP'].sum()
j4= Merged_Prices.groupby('year')['TradeValue_AZ'].sum()
j5= Merged_Prices.groupby('year')['TradeValue_FB'].sum()
j6= Merged_Prices.groupby('year')['TradeValue_MS'].sum()
ax.bar(i,j1, label="GameStop")
ax.bar(i,j2, bottom=j1, label='Goldman Sachs')
ax.bar(i,j3, bottom=j1+j2, label='JP Morgan')
ax.bar(i,j4, bottom=j1+j2+j3, label='Amazon')
ax.bar(i,j5, bottom=j1+j2+j3+j4, label='Facebook')
ax.bar(i,j6, bottom=j1+j2+j3+j4+j5, label='Microsoft')
ax.plot(i,h)
ax.set(title='Total Trade Value Stacked', ylabel='Value in bns', xlabel='Year')
ax.legend(loc='best')
fig.savefig("Charts/Trading Value.png")
plt.show()


#3 scatter plt all 3 close prices and volume
fig, ax = plt.subplots(3,2)
fig.set_size_inches([10, 6])
fig.suptitle('Volume / Price Scatter', fontsize=16)
q = GS['Close']
s = GS['Volume']
q1 = GM['Close']
s1 = GM['Volume']
q2 = JP['Close']
s2 = JP['Volume']
q3 = AZ['Close']
s3 = AZ['Volume']
q4 = FB['Close']
s4 = FB['Volume']
q5 = MS['Close']
s5 = MS['Volume']
ax[0,0].scatter(q, s, label='GameStop', c='tab:red')
ax[1,0].scatter(q1, s1, label='Goldman', c='tab:blue')
ax[2,0].scatter(q2, s2, label='JP Morgan', c='tab:orange')
ax[0,1].scatter(q3, s3, label='Amazon', c='tab:purple')
ax[1,1].scatter(q4, s4, label='Facebook', c='tab:pink')
ax[2,1].scatter(q5, s5, label='Microsoft', c='tab:cyan')
ax[0,0].legend(loc='upper right')
ax[1,0].legend(loc='upper right')
ax[2,0].legend(loc='upper right')
ax[0,1].legend(loc='upper right')
ax[1,1].legend(loc='upper right')
ax[2,1].legend(loc='upper right')
ax[0,0].set(ylabel='Volume')
ax[2,1].set(xlabel='Price')
fig.savefig("Charts/Volume and Price Scatter.png")
plt.show()



#4a seaborn scatter plot Change in pr / change in volume
#No_Obs =2184 #1500
q = Merged_Prices1['Prct_chg_pr_7d_GS']#.head(No_Obs)
s = Merged_Prices1['Prct_chg_pr_7d_AZ']#.head(No_Obs)


#5a
f, ax = plt.subplots() #figsize=(3,3)
sns.scatterplot(x=q, y=s, s=5, color=".15")
sns.histplot(x=q, y=s, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=q, y=s, levels=5, color="w", linewidths=1)
ax.set(title='Relative 7 day percentage change in price', ylabel='Amazon', xlabel='GameStop')
ax.legend(loc='best')
f.savefig("Charts/Relative Price change.png")
plt.show()





# 5 linear regression
fig, ax = plt.subplots()
#No_Obs =2184
#q=Merged_Prices1.sort_values('Date', ascending=True)
q2 = Merged_Prices1['Prct_chg_pr_7d_GS']#.head(No_Obs)
s2 = Merged_Prices1['Prct_chg_pr_7d_AZ']#.head(No_Obs)
q2_last50 = Merged_Prices1['Prct_chg_pr_7d_GS'].tail(50)
s2_last50 = Merged_Prices1['Prct_chg_pr_7d_AZ'].tail(50)
ax.scatter(q2, s2, label='GameStop / Amazon price relationship', alpha=0.5)
ax.set(title='7-day percentage price changes', ylabel='Amazon', xlabel='GameStop')
X = q2.values.reshape(-1, 1)
Y = s2.values.reshape(-1, 1)
ln = LinearRegression()
ln.fit(X, Y)
Y_pred = ln.predict(X)
plt.plot(X, Y_pred, color='red')
fig.savefig("Charts/Linear regression.png")
plt.show()

r_sq=ln.score(X,Y)
print('coefficient of determination (GameStop/Amazon):', r_sq)
print('slope (GameStop/Amazon):', ln.coef_)

print('Standard deviation of GameStop:', q2.std())
print('Standard deviation of Amazon:', s2.std())
cov_matrix1 = cov(q2,s2)
weights1= np.array([0.5,0.5])
port_variance1 = np.dot(weights1.T,np.dot(cov_matrix1,weights1))
port_std1=np.sqrt(port_variance1)

Corr = np.corrcoef(q2,s2)
print('Correlation:', Corr)
print('Standard deviation of GameStop and Amazon combined:', port_std1)
print('Recent Standard deviation of GameStop:', q2_last50.std())
print('Recent Standard deviation of Amazon:', s2_last50.std())
cov_matrix2 = cov(q2_last50,s2_last50)
weights2= np.array([0.5,0.5])
port_variance2 = np.dot(weights2.T,np.dot(cov_matrix2,weights2))
port_std2=np.sqrt(port_variance2)
print('Recent Standard deviation of GameStop and Amazon combined:', port_std2)


# Calculate volatility of Bank Stocks
GM_returns = Merged_Prices['Prct_chg_pr_1d_GM']
JP_returns = Merged_Prices['Prct_chg_pr_1d_JP']
GM_volatility =GM_returns.std()
JP_volatility =JP_returns.std()

x = np.array(GM_returns,JP_returns)
CoVar = np.cov(x)
Corr1 = np.corrcoef(GM_returns,JP_returns)

print('Correlation bank stocks:', Corr1)

# calculate portfolio variance

c= GM_returns.mean()
d= JP_returns.mean()
e=(c+d)/2
cov_matrix = cov(GM_returns,JP_returns)
weights= np.array([0.5,0.5])
port_variance = np.dot(weights.T,np.dot(cov_matrix,weights))



port_variance_frmt = (str(np.round(port_variance, 3) * 100) + '%')
port_std=np.sqrt(port_variance)
portfolio_Std = Cl.Percent_Format(port_std)

print(f'Goldman annualised return: {c}')
print(f'JP Morgan mean return: {d}')
print(f'Portfolio return: {e}')
print(f'Portfolio Covariance: {cov_matrix}')
print(f'Goldman Std: {GM_volatility}')
print(f'JP Morgan Std: {JP_volatility}')
print(f'Portfolio Std: {portfolio_Std}')

#Group mean prices annually
group_merged_prices = Merged_Prices.groupby('year')['Close_JP'].mean()
print(group_merged_prices)


# Investor has option to buy stock each year.
# If average annual price is above the strike price, the option holder makes a return (Average price minus the strike price).
# THe cost of purchasing the option is 20 and entitle es holder to buy shares every year.
# The strike price is 60.
# The question is what would be the return since 2006 and secondly, at what point does the holder make a positive return.
Opt_strike=[-60]
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
clc2 = npf.irr(list2[0:6])

print(f'Stock Option return from 2012 to 2020: {round(clc,2)}') # Investor has option to buy stock each month.
print(f'Stock Option return from 2012 to 2016: {round(clc2,2)}')





