import joblib


test = joblib.load('AMZN.joblib')
input = [float(input()),int(input()),int(input()),int(input())]
predict = test.predict([input])
print(predict)



