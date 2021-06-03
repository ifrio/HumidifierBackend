from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm
import pandas as pd
from sklearn import linear_model
import requests
import json
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

"""
Building the data table
"""
data = [
[1, 8, 8, 93],
[1, 7, 9, 94],
[1, 7, 10, 91],
[0, 10, 16, 77],
[0, 14, 22, 55],
[0, 15, 22, 49],
[0, 12, 19, 58],
[0, 10, 16, 65],
[1, 6, 13, 82],
[1, 7, 10, 85],
[1, 7, 6, 84],
[1, 9, 3, 76],
[0, 12, 3, 64],
[0, 13, 3, 63],
[0, 12, 7, 70],
[0, 10, 7, 74]]

column =  ["On/Off", "Temp","Wind Speed", "Humidity"]
data = pd.DataFrame(data,columns =column)
print(" ")
print(data)

def pushData(location, recordLabel, record):
  res = {recordLabel : record}

  response = requests.put(location, data=json.dumps(res))
  
  print(response.content)

def linearRegressHumidity(temperature, windSpeed):
  xData = data[["Temp", "Wind Speed"]]
  yData = data["Humidity"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  model = make_pipeline(PolynomialFeatures(3), Ridge())
  model.fit(X_train, Y_train)
  predictedHumidity = model.predict([[temperature, windSpeed]])
  
  return predictedHumidity

#Link for Thunkable
#1) Pulls data from FireBase
#2) Makes prediction
#3) Pushes prediction to FireBase
@app.route("/predictHumidity")
def predictHumidity():
  #get the curret Humidity
  print("requesting current temp and UV data from fire base")
  temp = requests.get('https://codingminds-default-rtdb.firebaseio.com/devices/d001/temperature')
  UV = requests.get('https://codingminds-default-rtdb.firebaseio.com/devices/d001/UV')
  print(temp,UV)

  #predicting next days humidity
  print("Making device turn on or off recommendation")
  predictedHumidity = linearRegressHumidity(temp, UV)
  print(predictedHumidity)

  #push predicted humidity
  print("pushing recommendation to firebase")
  location = "https://codingminds-default-rtdb.firebaseio.com/devices/d001/predictedHumidity"
  response = pushData(location, "predictedHumidity", predictedHumidity)
  print("Success Status: ", response)

def humidifierOnOff(temperature, currentHumidity):
  xData = data[["Temp", "Humidity"]]
  yData = data["On/Off"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  recommendation_model = svm.SVC()
  recommendation_model.fit(xData, yData)
  recommendation = recommendation_model.predict([[temperature, currentHumidity]])

  return recommendation

#Link for Thunkable
#1) Pulls data from FireBase
#2) Makes prediction
#3) Pushes prediction to FireBase
@app.route("/turnOnOffDevice")
def turnOnOffDevice():
  #get the curret Humidity
  print("requesting current humidity data from fire base")
  record = requests.get('https://codingminds-default-rtdb.firebaseio.com/devices/d001/current')
  print(record)

  #decide whether or not to turn on or off humidifier
  print("Making device turn on or off recommendation")
  deviceRecommendationStatus = predictHumidity(record)
  print(deviceRecommendationStatus)

  #push devices updated power on or off (1/0) status
  print("pushing recommendation to firebase")
  location = "https://codingminds-default-rtdb.firebaseio.com/devices/d001/deviceStatus"
  response = pushData(location, "deviceStatus", deviceRecommendationStatus)
  print("Success Status: ", response)

def researchQuestion1():
  """ 
  Research Quesiton #1:
  Predicting Humidity based on Temp in Celcius, and the Wind speed using
  a polynomial machine learning model
  """
  xData = data[["Temp", "Wind Speed"]]
  yData = data["Humidity"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  model = linear_model.LinearRegression()
  model.fit(X_train, Y_train)
  print('Res:', model.score(X_test, Y_test))

  model = make_pipeline(PolynomialFeatures(2), Ridge())
  model.fit(X_train, Y_train)
  print('Res:', model.score(X_test, Y_test))

  model = make_pipeline(PolynomialFeatures(3), Ridge())
  model.fit(X_train, Y_train)
  print('Res:', model.score(X_test, Y_test))

def researchQuestion2():
  """
  Research Question #2:
  Classifying between whether the machine was turned on or off based on the humidity, temperature, and wind speed
  """
  xData = data[["Temp", "Wind Speed"]]
  yData = data["On/Off"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  recommendation_model = svm.SVC()
  recommendation_model.fit(xData, yData)
  print('Res:', recommendation_model.score(X_test, Y_test))

  xData = data[["Wind Speed","Humidity"]]
  yData = data["On/Off"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  recommendation_model = svm.SVC()
  recommendation_model.fit(xData, yData)
  print('Res:', recommendation_model.score(X_test, Y_test))

  xData = data[["Temp", "Wind Speed","Humidity"]]
  yData = data["On/Off"]
  X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

  recommendation_model = svm.SVC()
  recommendation_model.fit(xData, yData)
  print('Res:', recommendation_model.score(X_test, Y_test))