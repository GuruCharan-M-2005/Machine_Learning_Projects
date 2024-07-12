import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(category=UserWarning,action='ignore')

data=pd.read_csv('Titanic.csv')
df=pd.DataFrame(data)   

df=df.drop(['PassengerId','Pclass','Name','SibSp','Ticket','Cabin'],axis='columns')
le=LabelEncoder()
df.Sex=le.fit_transform(df.Sex)
df.Embarked=le.fit_transform(df.Embarked)
df=df.fillna({
    'Sex':np.mean(df.Sex),
    'Embarked':np.mean(df.Embarked),
    'Age':np.mean(df.Age),
    'Parch':np.mean(df.Parch),
    'Fare':np.mean(df.Fare)
})

x=df.drop(['Survived'],axis='columns')
y=df.Survived
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

model=RandomForestClassifier()
model.fit(xtrain,ytrain)

st.title('Titanic Survival Prediction')

a=st.number_input('Sex (Male->1, Female->0)',min_value=0)
b=st.number_input('Age',min_value=0.0)
c=st.number_input('Parch',min_value=0)
d=st.number_input('Fare',min_value=0.0)
e=st.number_input('Embarked (C->0, Q->1, S->2)',min_value=0)

if st.button('Predict'):
    df=pd.DataFrame({
        'Sex':[a],
        'Age':[b],
        'Parch':[c],
        'Fare':[d],
        'Embarked':[e]
    })
    predicted=model.predict(df)
    if predicted[0]==1:
        st.write('Passenger Survied Successfully')
    else:
        st.write('Passenger do not Survied')