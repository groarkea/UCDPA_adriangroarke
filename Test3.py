import numpy as np
import pandas as pd

from datetime import datetime
import matplotlib.pyplot as plt
IRE = pd.read_csv("Property_Price_Register_Ireland-28-05-2021.csv")







missing_values_count = IRE.isnull().sum()
print(missing_values_count[0:10])
IRE1 = IRE.dropna(axis=1) # drop columns
print(IRE.shape,IRE1.shape)

missing_values_count = IRE1.isnull().sum()
print(missing_values_count[0:10])

plt.hist(IRE['COUNTY']) # 'COUNTY'
plt.xticks(rotation='vertical')
plt.show()



#cnty=IRE1['COUNTY']
#saledt=IRE1['SALE_DATE'].tail(2)
#prices=IRE1['SALE_PRICE'].tail(2)
#IRE1.plot(x=saledt, y=prices, kind='bar')
#plt.hist(x=prices, x1=prices)
#plt.show()

# IRE1 = IRE.sort_values(["SALE_DATE"] < ref_start, ascending=False)
# print(IRE1["SALE_DATE"])
#print(IRE.tail(10))
# print(GS.tail(2))
# print(IRE.info())
# print(IRE.shape)
# print(IRE.describe())
# print(IRE.values)
# print(IRE.columns)
