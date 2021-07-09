import pandas as pd

GS = pd.read_csv("Datasets/GME_stock.csv")
print(GS.columns)
end_date = '2021-01-20'
print(end_date)
mask_dt = GS['date'] < end_date

GS2 = GS.loc[mask_dt]
print(GS2.describe())
print(GS2.describe(include='object'))
