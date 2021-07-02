import numpy as np
import requests
data=requests.get('http://api.open-notify.org/iss-now.json')

parsed_data=data.json() # looking for dictionary instead of string
print(data.text)
print(type(data.text))
print(parsed_data)
print(type(parsed_data)) # once dictionary then can use index []
print(parsed_data['iss_position'])

data2=requests.get('http://api.open-notify.org/astros.json')
parsed_data2=data2.json()
print(parsed_data2)
for temp_variable in parsed_data2['people']:
    print(temp_variable['name'])

data4=requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
print(data4.text)
print(type(data4))
parsed_data4=data4.json()
print(type(parsed_data4))
output=parsed_data4['Description'] + ' The companies earnings per share is ' + parsed_data4['EPS']
print(output)

data5=requests.get('https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=JPM&apikey=RVZ3KJWM4Q6RNVCJ')
print(data5.text)
print(type(data5))
parsed_data5 = data5.json()
print(type(parsed_data5))

parsed_data5a = parsed_data5['annualReports']
print(parsed_data5a)

for temp_variable in parsed_data5a:
    x = np.array(temp_variable['fiscalDateEnding'])
    y = np.array(temp_variable['fiscalDateEnding'])

    print(x,y)





 #   print(temp_variable['totalAssets'])
 #   tb = {'dt':temp_variable['fiscalDateEnding'],'Assets':temp_variable['totalAssets']}

#print(tb)
#for temp_variable in parsed_data5a:
#    Total_assets = temp_variable['totalAssets'])

#print(Annual_Rpt)

