import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# stock_name=input()
# stock_data = pd.read_csv(f'{stock_name}.csv')
stock_data = pd.read_csv('GOOGLE2004.csv')
stock_data=stock_data.drop(['Adj Close'],axis=1)

# for i in range(1, 6):  
#     stock_data[f'Low_Lag_{i}'] = stock_data['Low'].shift(i)

stock_data.dropna(inplace=True)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day



X=stock_data.drop(['Low'], axis=1)
y=stock_data['Low']

train_size = int(len(stock_data)*0.8)



train=stock_data.iloc[:train_size]

X_train, y_train = train.drop(['Volume','Low','High','Close','Date'],axis=1), train['Low']
# print(X_train, y_train)

test=stock_data.iloc[train_size:]
X_test, y_test = test.drop(['Volume','Low','High','Close','Date'],axis=1), test['Low']
# print(X_test,y_test)

model=LinearRegression()
model.fit(X_train, y_train)
# joblib.dump(model,'AMZN.joblib')
score = model.score(X_test, y_test)

print(score*100)

model = model.predict([[142.13,2023,12,22]])
print(model)

# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=500, random_state=0)
# regressor.fit(X_train, y_train)
# score = regressor.score(X_test,y_test)
# print(score)

# y_pred= model.predict(X_test)
# # print(y_pred)
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# plt.scatter(stock_data['Date'],stock_data['Low'])
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Linear Regression: Actual vs Predicted')
# plt.show()

# plt.scatter(predict,y_test)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Linear Regression: Actual vs Predicted')
# plt.show()


# plt.figure(figsize=(12, 6))
# plt.plot(stock_data['Low'], label='Actual Low Prices')
# plt.plot(predict, label='Predicted Low Prices', linestyle='--')
# plt.title('Low Price Prediction using ARIMA')
# plt.xlabel('Date')
# plt.ylabel('Low Price')
# plt.legend()
# plt.show()