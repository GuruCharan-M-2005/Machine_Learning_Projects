import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('Mall_Customers.csv')
df=pd.DataFrame(data)

df=df.drop(['CustomerID'],axis='columns')
le=LabelEncoder()
df['Gender']=le.fit_transform(df.Gender)

kmeans=KMeans(n_clusters=3)
kmeans.fit(df)
df['labels']=kmeans.labels_

st.title('Input New Customer Data')

gender=st.number_input('Gender (0->female, 1->male)',min_value=0,max_value=1)
age=st.number_input('Age',min_value=0,max_value=100)
annual_income=st.number_input('Annual Income (k$)',min_value=0,max_value=150)
spending_score=st.number_input('Spending Score (1-100)',min_value=0,max_value=100)

new_data = pd.DataFrame({
    'Gender': [gender] ,
    'Age': [age],
    'Annual Income (k$)': [annual_income],
    'Spending Score (1-100)': [spending_score]
})

if st.button('Group'):
    predicted=kmeans.predict(new_data)
    st.write(predicted[0])


