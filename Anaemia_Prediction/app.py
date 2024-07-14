import pandas as pd
import streamlit as st
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(category=UserWarning,action='ignore')

data=pd.read_excel('Anaemia.xlsx','Anaemia')
df=pd.DataFrame(data)

df=df.drop(['Number'],axis='columns')
le=LabelEncoder()
df.Sex=le.fit_transform(df.Sex)
df.Anaemic=le.fit_transform(df.Anaemic)   

x=df.drop(['Anaemic'],axis='columns')
y=df.Anaemic
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,train_size=0.8)

le=LogisticRegression()
model=le.fit(xtrain,ytrain)

st.title('Anaemia Predictor')
Sex=st.number_input("Enter your Sex (0 for Female, 1 for Male)", min_value=0, max_value=1, step=1)
Red=st.number_input("Enter your Red Pixel Value", min_value=0.0, max_value=100.0, step=0.1)
Green=st.number_input("Enter your Green Pixel Value", min_value=0.0, max_value=100.0, step=0.1)
Blue=st.number_input("Enter your Blue Pixel Value", min_value=0.0, max_value=100.0, step=0.1)
Hb=st.number_input("Enter your Hemoglobin Level", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict"):  
    df=pd.DataFrame({
            'Sex':[Sex],
            '%Red Pixel':[Red],
            '%Green pixel':[Green],
            '%Blue pixel':[Blue],
            'Hb':[Hb],
        })
    predicted = model.predict(df)
    if predicted[0]==1:
        st.write('You are Anaemic')
    else:
        st.write('You have no Anaemia')
