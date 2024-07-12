import pandas as pd
import streamlit as st
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(category=UserWarning,action='ignore')

data=pd.read_csv('Sales.csv')
df=pd.DataFrame(data)

df=df.drop(['Invoice ID','Date','Time','Tax 5%','gross margin percentage','cogs','gross income'],axis='columns')
le=LabelEncoder()
df['Branch']=le.fit_transform(df['Branch'])
df['City']=le.fit_transform(df['City'])
df['Customer type']=le.fit_transform(df['Customer type'])
df['Gender']=le.fit_transform(df['Gender'])
df['Product line']=le.fit_transform(df['Product line'])
df['Payment']=le.fit_transform(df['Payment'])

x=df.drop(['Total','Rating'],axis='columns')
y=df[['Total','Rating']]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)
lm=LinearRegression()
lm.fit(xtrain,ytrain)

st.title('Sales Prediction')

a=st.number_input('Branch (A->0, C->1, B->2)', min_value=0)
b=st.number_input('City (Yangon->2, Naypyitaw->1, Mandalay->0)', min_value=0)
c=st.number_input('Customer type (Member->0, Normal->1)', min_value=0)
d=st.number_input('Gender (Female->0, Male->1)', min_value=0)
e=st.number_input('Product line (Health and beauty->3, Electronic accessories->0, Home and lifestyle->4, Sports and travel->5, Food and beverages->2, Fashion accessories->1)', min_value=0)
f=st.number_input('Unit price', min_value=0)
g=st.number_input('Quantity', min_value=0)
h=st.number_input('Payment (Ewallet->2, Cash->0, Credit card->1)', min_value=0)

if st.button('Predict'):
    df=pd.DataFrame({
        "Branch":[a],
        "City":[b],
        "Customer type":[c],
        "Gender":[d],
        "Product line":[e],
        "Unit price":[f],
        "Quantity":[g],
        "Payment":[h]
    })
    predict=lm.predict(df)
    st.write(f'Predicted Sale Total:{predict[0][0]}')
    st.write(f'Predicted Sale Rate:{predict[0][1]}')