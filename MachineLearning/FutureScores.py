import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_moca_score_all_previous(df, target_test_index):

    # Ensure the target test index is valid
    if target_test_index < 2 or target_test_index > 12:
        raise ValueError("Target test index must be between 2 and 12, inclusive.")
    
    # Prepare the features and target
    features_columns = [f'Test_{i}' for i in range(1, target_test_index)]
    target_column = f'Test_{target_test_index}'
    
    features = df[features_columns]
    target = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test,y_test)
    # Predict the scores
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2,accuracy,y_pred

# Example usage: Predict Test 7 based on Test 1 through 6
df = pd.read_csv('MoCa.csv')
model, mse, r2, accuracy, y_pred = predict_moca_score_all_previous(df, 7)
print("Mean Squared Error:", mse)
print("Predicted Score:", y_pred)
print("R² Score:", r2)
print("Accuracy: ", accuracy)
