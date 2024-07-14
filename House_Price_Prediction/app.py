import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

import warnings
warnings.filterwarnings(category=UserWarning,action='ignore')

data=pd.read_csv('Housing.csv')
df=pd.DataFrame(data)

df=df.drop(['stories','mainroad','guestroom','basement','parking','hotwaterheating','airconditioning'
           ,'parking','furnishingstatus','prefarea'],axis='columns')

x=df.drop(['price'],axis='columns')
y=df.price
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

lm=LinearRegression()
lm.fit(xtrain,ytrain)

st.title('House Price Prediction')

a=st.number_input('Area',min_value=0)
b=st.number_input('Bedrooms',min_value=0)
c=st.number_input('Bathrooms',min_value=0)

if st.button('Predict'):
    df=pd.DataFrame({
        'area':[a],
        'bedrooms':[b],
        'bathrooms':[c]
    })
    predicted=lm.predict(df)
    st.write(f'The predicted Price: {predicted[0]}')
