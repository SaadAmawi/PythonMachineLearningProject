import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
# stock_data=stock_data.drop(['MMSE2'],axis=1)


stock_data.dropna(inplace=True)




X=stock_data.drop(['Dx'], axis=1)
y=stock_data['Dx']

train_size = int(len(stock_data)*0.7)



train=stock_data.iloc[:train_size]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Started Training")

# Predict the class for a new data point
new_data_point = pd.DataFrame([[1, 66, 12, 30, 30]], columns=X_train.columns)
new_data_point_scaled = scaler.transform(new_data_point)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN_model = KNN.fit(X_train, y_train)
accuracy = KNN_model.score(X_test, y_test)
print("Accuracy of KNN Model: ",accuracy*100)
predicted = KNN_model.predict(new_data_point_scaled)
print("Predicted Class: ",predictClass(predicted[0]))


# Logistic Regression
logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression_model.fit(X_train_scaled, y_train)
accuracy = logistic_regression_model.score(X_test_scaled, y_test)
print("Accuracy of Logistic Regression:", accuracy * 100)
predicted_class = logistic_regression_model.predict(new_data_point_scaled)
print("Predicted class using Logistic Regression:", predictClass(predicted_class[0]))




# TODO: TENFOLD CROSS VALIDATION
# TODO: AREA UNDER SCORE, F1 SCORE, SENSITIVITY AND SPECIFICITY FIGURES
# TODO: TRY DIFFERENT MODELS (GRADIENT BOOSTING CLASSIFIER, LOGISTIC REGRESSION, DECISION TREE)





