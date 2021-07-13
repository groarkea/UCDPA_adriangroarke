import numpy as np
import requests
import pandas as pd
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





