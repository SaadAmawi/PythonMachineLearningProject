import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib


test = joblib.load('StockPredictor1.joblib')
input = [float(input()),int(input()),int(input()),int(input())]
predict = test.predict([input])
print(predict)



