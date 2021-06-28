import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt
GS = pd.read_csv("GME_stock.csv")
print(GS.columns)
missing_values_count = GS.isnull().sum()
print(missing_values_count[0:10])

GS.plot("date","close_price")
plt.show()
print(datetime(2020,9,20))
print(GS['date'].head())
print(GS.describe())
print(GS.describe(include='object'))
maskpr = GS['high_price'] > 100 # GS.high_price work too!
GS1 = GS.loc[maskpr]
print(GS1.describe())
print(GS1.describe(include='object'))

mask_dt = GS.date < datetime(2021, 1, 4) # GS['date']
GS2 = GS.loc[mask_dt]
print(GS2.describe())

print(GS2.describe(include='object'))

#ref_start = GS['date'] > datetime.(2020-01-01)
# ref_end = GS['date'] < datetime(2020,9,25)
# dtrnge = ref_start & ref_end
# GS.loc[dtrnge]

