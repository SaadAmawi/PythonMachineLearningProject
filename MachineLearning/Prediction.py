import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def predictClass(score):
    if(score==0):
        return "Sane"
    elif(score==1):
        return "Mild"
    elif(score==2):
        return "Serious"

# stock_name=input()
# stock_data = pd.read_csv(f'{stock_name}.csv')
stock_data = pd.read_csv('ALZ.csv')
stock_data=stock_data.drop(['MMSE2'],axis=1)


stock_data.dropna(inplace=True)




X=stock_data.drop(['Dx'], axis=1)
y=stock_data['Dx']

train_size = int(len(stock_data)*0.8)



train=stock_data.iloc[:train_size]

X_train, y_train = train.drop(['Dx'],axis=1), train['Dx']
# print(X_train, y_train)

test=stock_data.iloc[train_size:]
X_test, y_test = test.drop(['Dx'],axis=1), test['Dx']
# print(X_test,y_test)

# model=LinearRegression()
# model.fit(X_train, y_train)
# # joblib.dump(model,'AMZN.joblib')
# score = model.score(X_test, y_test)
# predict = model.predict(np.array([1, 66, 12, 26]).reshape(1, -1))
# print(score*100)
# print(predict)


# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=500, random_state=0)
# regressor.fit(X_train, y_train)

# score = regressor.score(X_test,y_test)
# print(score*100)



# new_data_point = np.array([1, 66, 12, 3]).reshape(1, -1)
# predicted_class = regressor.predict(new_data_point)
# print("Predicted class:", predictClass(predicted_class))

random_forest_model = RandomForestClassifier(n_estimators=500, random_state=0)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
accuracy = random_forest_model.score(X_test, y_test)
print("Accuracy of RandomForestClassifier:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = random_forest_model.predict(new_data_point)
print("Predicted class using RandomForestClassifier:", predictClass(predicted_class))



knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
knn_model.fit(X_train, y_train)

# Evaluate the model
accuracy = knn_model.score(X_test, y_test)
print("Accuracy of KNN:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 20]).reshape(1, -1)
predicted_class = knn_model.predict(new_data_point)
print("Predicted class using KNN:", predictClass(predicted_class))