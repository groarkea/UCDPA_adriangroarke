import pandas as pd

from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
GS = pd.read_csv("GME_stock.csv")
print(GS.columns
      )
missing_values_count = GS.isnull().sum()
print(missing_values_count[0:10])

GS.plot("date","close_price")
plt.show()
print(datetime(2020,9,20))
print(GS['date'].head())
print(type(GS['date']))

ref_start = GS['date'] > datetime(2020-01-01)
# ref_end = GS['date'] < datetime(2020,9,25)
# dtrnge = ref_start & ref_end
# GS.loc[dtrnge]

