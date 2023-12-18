import joblib


test = joblib.load('StockPredictor1.joblib')
input = [float(input()),int(input()),int(input()),int(input())]
predict = test.predict([input])
print(predict)



