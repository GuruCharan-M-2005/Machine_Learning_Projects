import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

data=pd.read_csv('Iris.csv')
df=pd.DataFrame(data)

df=df.drop(['Id'],axis='columns')
le=LabelEncoder()
df.Species=le.fit_transform(df.Species)

x=df.drop(['Species'],axis='columns')
y=df.Species
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

svm=SVC()
svm.fit(xtrain,ytrain)
# svm.predict([[5.1,3.5,1.4,0.2]])

st.title('Iris Prediction')

a=st.number_input('Sepal Length(cm)', min_value=0.0, format="%.2f")
b=st.number_input('Sepal Width(cm)', min_value=0.0, format="%.2f")
c=st.number_input('Petal Length(cm)', min_value=0.0, format="%.2f")
d=st.number_input('Petal Width(cm)', min_value=0.0, format="%.2f")

if st.button('Predict'):
    df=pd.DataFrame({
        "SepalLengthCm":[a],
        "SepalWidthCm":[b],
        "PetalLengthCm":[c],
        "PetalWidthCm":[d]
    })
    prediction = svm.predict(df)[0]
    species = le.inverse_transform([prediction])[0]
    st.write(f'The predicted species is: {species}')