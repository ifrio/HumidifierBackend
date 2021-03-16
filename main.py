from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm
import pandas as pd

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
print(data)
print(" ")


""" 
Research Quesiton #1:
Predicting Humidity based on Temp in Celcius, and the Wind speed using
a polynomial machine learning model
"""
xData = data[["Temp", "Wind Speed"]]
yData = data["Humidity"]
X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

model = make_pipeline(PolynomialFeatures(2), Ridge())

model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))

"""
Research Question #2:
Classifying between whether the machine was turned on or off based on the humidity, temperature, and wind speed
"""
xData = data[["Temp", "Wind Speed","Humidity"]]
yData = data["On/Off"]
X_train, X_test, Y_train, Y_test = train_test_split(xData,  yData,test_size=0.3,random_state=0)

recommendation_model = svm.SVC()
recommendation_model.fit(xData, yData)
print(recommendation_model.score(X_test, Y_test))
