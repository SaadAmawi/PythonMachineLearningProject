# %%
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle


# %%

def predictClass(score):
    if(score==0):
        return "Sane"
    elif(score==1):
        return "Mild"
    elif(score==2):
        return "Serious"
        

# %%
stock_data = pd.read_csv('ALZ.csv')
stock_data=stock_data.drop(['MMSE2'],axis=1)


stock_data.dropna(inplace=True)




X=stock_data.drop(['Dx'], axis=1)
y=stock_data['Dx']

train_size = int(len(stock_data)*0.7)

train=stock_data.iloc[:train_size]

# %%
X,y
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)

# %%
y_bin = label_binarize(y, classes=[0, 1, 2])

# %%
X_train, y_train = train.drop(['Dx'],axis=1), train['Dx']
test=stock_data.iloc[train_size:]
X_test, y_test = test.drop(['Dx'],axis=1), test['Dx']
train_data=X_train.join(y_train)
train_data

# %%
model_accuracies = {}

# Models initialization
models = {
    'RandomForest': RandomForestClassifier(n_estimators=500, random_state=0),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', probability=True, random_state=0),
   
}

# %%
for model_name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    model_accuracies[model_name] = accuracy * 100
    predicted_class = model.predict(new_data_point)
    print(f"Accuracy of {model_name}:", accuracy * 100)
    print(f"Predicted class using {model_name}:", predictClass(predicted_class[0]))

# %%
train_data.hist(figsize=(20,8))


# %%
plt.figure(figsize=(10, 6))

for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
    else:
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())  # Scale to [0, 1]

    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# %%
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

# %%
train_data.corr()

# %%
random_forest_model = RandomForestClassifier(n_estimators=500, random_state=0)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
accuracy = random_forest_model.score(X_test, y_test)
print("Accuracy of RandomForestClassifier:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = random_forest_model.predict(new_data_point)
print("Predicted class using RandomForestClassifier:", predictClass(predicted_class))

# %%
knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train, y_train)

accuracy = knn_model.score(X_test, y_test)
print("Accuracy of KNN:", accuracy * 100)

new_data_point = np.array([1, 66, 12, 20]).reshape(1, -1)
predicted_class = knn_model.predict(new_data_point)
print("Predicted class using KNN:", predictClass(predicted_class))

# %%
svm_model = SVC(kernel='linear', random_state=0)  # You can try different kernels like 'rbf', 'poly'
svm_model.fit(X_train, y_train)

# Evaluate the model
accuracy = svm_model.score(X_test, y_test)
print("Accuracy of SVM model:", accuracy * 100)

# Predict the class for a new data point
new_data_point = np.array([1, 66, 12, 30]).reshape(1, -1)
predicted_class = svm_model.predict(new_data_point)
print("Predicted class using SVM:", predictClass(predicted_class[0]))

# %%



