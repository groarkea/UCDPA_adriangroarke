import pandas as pd
import numpy as np
import numpy_financial as npf

import matplotlib.pyplot as plt

# Import/Read 3 stock price datasets
GS = pd.read_csv("GME_stock.csv")  #Game stop prices'
GM = pd.read_csv("GS.csv")  #Goldman Sachs prices
JP = pd.read_csv("JPM.csv")  #JPM prices

# Add year column to dataset based on pricing date
GS['year']= pd.DatetimeIndex(GS['date']).year
GM['year']= pd.DatetimeIndex(GM['Date']).year
JP['year']= pd.DatetimeIndex(JP['Date']).year

# describe shape and column of individual datasets
print(f'Shape of Gamestop is:{GS.shape}')
print(f'Shape of JP Morgan is:{JP.shape}')
print(f'Columns of JP Morgan are:{JP.columns}')

#Merage 3 data sets on date
Merged_Prices= GS.merge(GM, left_on='date', right_on='Date') \
    .merge(JP, on='Date', suffixes=('_GM','_JP'))

#Describe Merged Prices
print(f'Describe:{Merged_Prices}')
print(f'Shape of Merged prices is:{Merged_Prices.shape}')
print(f'Columns of Merged prices are:{Merged_Prices.columns}')



# Calculate volatility of Stocks
volatility_GS =Merged_Prices['close_price'].std()
volatility_GM = Merged_Prices['Close_GM'].std()
volatility_JP = Merged_Prices['Close_JP'].std()
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
print(f'Stock Option return since 2006: {clc}') # Investor has option to buy stock each month.
print(f'Stock Option return from 2006 to 2013: {clc}')







