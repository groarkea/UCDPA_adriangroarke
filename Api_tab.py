import numpy as np
import requests


data4=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
print(data4.text)
print(type(data4))
parsed_data4=data4.json()
print(type(parsed_data4))
output=parsed_data4['Description'] + "' The company's earnings per share is '" + parsed_data4['EPS']
print(output)

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





