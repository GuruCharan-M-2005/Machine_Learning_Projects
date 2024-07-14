import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px


df=pd.DataFrame(pd.read_csv('creditcard.csv'))
df=df.drop(['Time'],axis='columns')

x=df.drop(['Class'],axis='columns')
y=df.Class
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8)

lr=LogisticRegression(max_iter=1000)
lr.fit(xtrain,ytrain)

