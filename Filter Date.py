import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
GS = pd.read_csv("GME_stock.csv")
print(GS.columns)

mask_dt = GS['date'] > datetime(2021, 1, 4)
GS2 = GS.loc[mask_dt]
print(GS2.describe())
print(GS2.describe(include='object'))

#maskpr = GS['high_price'] > 100 # GS.high_price work too!
#GS1 = GS.loc[maskpr]
#print(GS1.describe())
#print(GS1.describe(include='object'))

#mask_dt = GS.date < datetime(2021, 1, 4) # GS['date']
#GS2 = GS.loc[mask_dt]
#print(GS2.describe())
#print(GS2.describe(include='object'))