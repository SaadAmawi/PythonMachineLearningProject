import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# pd.options.display.max_rows = 3
stock_data = pd.read_csv('GOOGL.csv')

for i in range(1, 6):  
    stock_data[f'Low_Lag_{i}'] = stock_data['Low'].shift(i)
stock_data.dropna(inplace=True)

X=stock_data.drop(['Low'], axis=1)
y=stock_data['Low']

train_size = int(len(stock_data))
test_size = int(len(stock_data)*0.99)
# test_size = int(3931)
print(test_size, train_size)
# print(test_size)
train=stock_data.iloc[:train_size]
test=stock_data.iloc[test_size:]



model = ARIMA(train['Low'], order=(5, 1, 0))  # Adjust the order as needed
model_fit = model.fit()

predictions=model_fit.forecast(steps=len(test))
print(predictions)

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

stock_data = pd.read_csv('GOOGL.csv')



# for i in range(1,6):  
#     stock_data[f'Low_Lag_{i}'] = stock_data['Low'].shift(i)
# stock_data.dropna(inplace=True)

X=stock_data.drop(['Low'], axis=1)
y=stock_data['Low']
# X
train_size = int(len(stock_data)-3)
# test_size = int(len(stock_data)-1)
train=stock_data.iloc[:train_size]
test=stock_data.iloc[train_size:]

model = ARIMA(train['Low'], order=(0, 1, 0))  # Adjust the order as needed
model_fit = model.fit()

# rmse = sqrt(mean_squared_error(test['Low'], predictions))
# print(f'Root Mean Squared Error (RMSE): {rmse}')

predictions=model_fit.forecast(steps=3)
predictions

plt.figure(figsize=(12, 6))
plt.plot(test['Low'], label='Actual Low Prices')
plt.plot(predictions, label='Predicted Low Prices', linestyle='--')
plt.title('Low Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Low Price')
plt.legend()
plt.show()