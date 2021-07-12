import numpy as np
import requests

data=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=GME&apikey=RVZ3KJWM4Q6RNVCJ')
print(data.text)
print(type(data))
parsed_data=data.json()
print(type(parsed_data))
output="GameStop's 52 Week high is " + parsed_data['52WeekHigh'] + " and GameStop's 52 Week low is " + parsed_data['52WeekLow']
print(output)

a=parsed_data['Symbol']
b=parsed_data['52WeekHigh']
c=parsed_data['52WeekLow']
my_dic={'ID':a,'52High':b,'52Low':c}
print(my_dic)
print(type(my_dic))




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





