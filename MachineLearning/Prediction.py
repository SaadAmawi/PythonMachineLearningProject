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

train_size = int(len(stock_data)*0.7)



train=stock_data.iloc[:train_size]

X_train, y_train = train.drop(['Dx'],axis=1), train['Dx']
# print(X_train, y_train)

test=stock_data.iloc[train_size:]
X_test, y_test = test.drop(['Dx'],axis=1), test['Dx']
# print(X_test,y_test)


random_forest_model = RandomForestClassifier(n_estimators=500, random_state=0)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
accuracy = random_forest_model.score(X_test, y_test)
print("Accuracy of RandomForestClassifier:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = random_forest_model.predict(new_data_point)
print("Predicted class using RandomForestClassifier:", predictClass(predicted_class))


#K-NEAREST NEIGHBOURS
knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train, y_train)

accuracy = knn_model.score(X_test, y_test)
print("Accuracy of KNN:", accuracy * 100)

new_data_point = np.array([1, 66, 12, 20]).reshape(1, -1)
predicted_class = knn_model.predict(new_data_point)
print("Predicted class using KNN:", predictClass(predicted_class))


#SUPPORT VECTOR MACHINES MODEL
svm_model = SVC(kernel='linear', random_state=0)  # You can try different kernels like 'rbf', 'poly'
svm_model.fit(X_train, y_train)

# Evaluate the model
accuracy = svm_model.score(X_test, y_test)
print("Accuracy of SVM model:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = svm_model.predict(new_data_point)
print("Predicted class using SVM:", predictClass(predicted_class[0]))


# Gradient Boosting Classifier
gradient_boosting_model = GradientBoostingClassifier(random_state=0)
gradient_boosting_model.fit(X_train, y_train)
accuracy = gradient_boosting_model.score(X_test, y_test)
print("Accuracy of Gradient Boosting Classifier:", accuracy * 100)
predicted_class = gradient_boosting_model.predict(new_data_point)
print("Predicted class using Gradient Boosting Classifier:", predictClass(predicted_class[0]))

# Logistic Regression
logistic_regression_model = LogisticRegression(random_state=0)
logistic_regression_model.fit(X_train, y_train)
accuracy = logistic_regression_model.score(X_test, y_test)
print("Accuracy of Logistic Regression:", accuracy * 100)
predicted_class = logistic_regression_model.predict(new_data_point)
print("Predicted class using Logistic Regression:", predictClass(predicted_class[0]))

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=0)
decision_tree_model.fit(X_train, y_train)
accuracy = decision_tree_model.score(X_test, y_test)
print("Accuracy of Decision Tree Classifier:", accuracy * 100)
predicted_class = decision_tree_model.predict(new_data_point)
print("Predicted class using Decision Tree Classifier:", predictClass(predicted_class[0]))
# TODO: TENFOLD CROSS VALIDATION
# TODO: AREA UNDER SCORE, F1 SCORE, SENSITIVITY AND SPECIFICITY FIGURES
# TODO: TRY DIFFERENT MODELS (GRADIENT BOOSTING CLASSIFIER, LOGISTIC REGRESSION, DECISION TREE)


# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', random_state=0),
    "GradientBoosting": GradientBoostingClassifier(random_state=0),
    "LogisticRegression": LogisticRegression(random_state=0),
    "DecisionTree": DecisionTreeClassifier(random_state=0)
}

# Perform tenfold cross-validation and print accuracy
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f"Accuracy of {model_name} with tenfold cross-validation: {np.mean(scores) * 100:.2f}%")

# Predict the class for a new data point using the RandomForest model (example)
random_forest_model = models["RandomForest"]
random_forest_model.fit(X, y)
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = random_forest_model.predict(new_data_point)
print("Predicted class using RandomForestClassifier:", predictClass(predicted_class[0]))