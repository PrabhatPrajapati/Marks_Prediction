import numpy as np
import pandas as pd
df=pd.read_csv("student_info.csv")
print(df.head())
print(df.mean())
df=df.fillna(df.mean())
print(df.head())
print(df.isnull().sum())
x=df[["study_hours"]]
y=df[["student_marks"]]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51)
print(x_train.shape)
print(y_train.shape)
print(y_test.shape)
print(x_test.shape)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pre=lr.predict([[6]])
print(y_pre)
print(lr.score(x_test,y_test))
# import joblib
# joblib.dump(lr,"student_marks_prediction_model.pkl")
# model=joblib.load("student_marks_prediction_model.pkl")
# print(model.predict([[6]]))
import joblib
joblib.dump(lr,"student_marks_prediction_model.pkl")
prabhat=joblib.load("student_marks_prediction_model.pkl")
# pickle.dump(df,open("data.pkl","wb"))

print(prabhat.predict([[9]]))
