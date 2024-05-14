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
from sklearn.metrics import roc_curve, auc,RocCurveDisplay
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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
# stock_data=stock_data.drop(['MMSE2'],axis=1)


stock_data.dropna(inplace=True)



X=stock_data.drop(['Dx'], axis=1)
y=stock_data['Dx']



# %%
X,y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = X_train.join(y_train)
train_data

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
model_accuracies = {}

# Models initialization
models = {
    'RandomForest': RandomForestClassifier(n_estimators=500, random_state=0),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear', probability=True, random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'LogisticRegression': LogisticRegression(random_state=0),
    'DecisionTree': DecisionTreeClassifier(random_state=0)
}

# %%
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    model_accuracies[model_name] = accuracy * 100
   
    print(f"Accuracy of {model_name} with data scaling:", accuracy * 100)
print("===============================================")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    model_accuracies[model_name] = accuracy * 100
   
    print(f"Accuracy of {model_name} without data scaling:", accuracy * 100)

# %%
train_data.hist(figsize=(20,8))


# %%
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

# %%
train_data.corr()

# %%
random_forest_model = RandomForestClassifier(n_estimators=500, random_state=0)
random_forest_model.fit(X_train, y_train)
accuracy = random_forest_model.score(X_test, y_test)
print("Accuracy of RandomForestClassifier:", accuracy * 100)




# %%
knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train, y_train)
accuracy = knn_model.score(X_test, y_test)
print("Accuracy of KNN:", accuracy * 100)


# %%
svm_model = SVC(kernel='linear', random_state=0) 
svm_model.fit(X_train, y_train)
accuracy = svm_model.score(X_test, y_test)
print("Accuracy of SVM model:", accuracy * 100)



# %%
gradient_boosting_model = GradientBoostingClassifier(random_state=0)
gradient_boosting_model.fit(X_train, y_train)
accuracy = gradient_boosting_model.score(X_test, y_test)
print("Accuracy of Gradient Boosting Classifier:", accuracy * 100)

# %%
logistic_regression_model = LogisticRegression(random_state=42,max_iter=1000)
logistic_regression_model.fit(X_train, y_train)
accuracy = logistic_regression_model.score(X_test, y_test)
print("Accuracy of Logistic Regression:", accuracy * 100)



# %%
decision_tree_model = DecisionTreeClassifier(random_state=0)
decision_tree_model.fit(X_train, y_train)
accuracy = decision_tree_model.score(X_test, y_test)
print("Accuracy of Decision Tree Classifier:", accuracy * 100)


# %%
# Function to calculate AUC for different thresholds
def calculate_auc_scores(model, X_test, y_test):
    thresholds = np.arange(0.1, 1.1, 0.1)
    auc_scores = []
    for threshold in thresholds:
        predicted_probabilities = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, predicted_probabilities, multi_class='ovr')
        auc_scores.append(auc)
    return thresholds, auc_scores

# %%
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])




# Fit models and calculate AUC scores
plt.figure(figsize=(10, 6))
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predicted_probabilities = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), predicted_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.plot(fpr["micro"], tpr["micro"], label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10, 6))
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predicted_probabilities = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), predicted_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.plot(fpr["micro"], tpr["micro"], label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%
# Function to calculate macro-average F1 scores for different thresholds
def calculate_macro_f1_scores(model, X_test, y_test):
    thresholds = np.arange(0.1, 1.1, 0.1)
    f1_scores = []
    predicted_probabilities = model.predict_proba(X_test)
    for threshold in thresholds:
        predicted_classes = (predicted_probabilities >= threshold).astype(int)
        macro_f1 = f1_score(y_test_bin, predicted_classes, average='macro')
        f1_scores.append(macro_f1)
    return thresholds, f1_scores

# %%
# Plotting macro-average F1 score graph for each model
plt.figure(figsize=(10, 6))

for model_name, model in models.items():
    if(model_name!='DecisionTree'):
        model.fit(X_train, y_train)
        thresholds, f1_scores = calculate_macro_f1_scores(model, X_test, y_test)
        plt.plot(thresholds, f1_scores, label=f'{model_name}')

plt.xlabel('Threshold')
plt.ylabel('Macro-average F1 Score')
plt.title('Macro-average F1 Score for Different Models')
plt.legend()
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(10, 6))

for model_name, model in models.items():
    if(model_name!='DecisionTree'):
        model.fit(X_train_scaled, y_train)
        thresholds, f1_scores = calculate_macro_f1_scores(model, X_test, y_test)
        plt.plot(thresholds, f1_scores, label=f'{model_name}')

plt.xlabel('Threshold')
plt.ylabel('Macro-average F1 Score')
plt.title('Macro-average F1 Score for Different Models')
plt.legend()
plt.grid(True)
plt.show()

# %%



