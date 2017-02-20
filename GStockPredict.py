#! python3

## Prediciting Google stock price for next 30 days using Linear regression and plotting our predictions using scikit learn
import pandas as pd #panda for algebraic operations
import quandl, math #quandl for fetching data
import numpy as np
from sklearn import preprocessing, svm    #preprocessing for feature scaling and mean normalization
from sklearn.model_selection import train_test_split    #for dividing data into training ans cross validation sets
from sklearn.linear_model import LinearRegression
import datetime #for working with datetime objects
import matplotlib.pyplot as plt #for plotting data
from matplotlib import style #style makes graphs look good
import pickle #for saving the trained model

quandl.ApiConfig.api_key='BQ2T1y3gqjkmk3qL7vjx'
df=quandl.get('WIKI/GOOGL') #Get google stock prices

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)  #fill NaN values with -99999 to prevent errors on NaN values

forecast_out=int(math.ceil(0.01*len(df)))   #Taking last 1% of data

df['label']=df[forecast_col].shift(-1*forecast_out) #Shift forecast column above, we are predicting for new 1% data


X=np.array(df.drop(['label'],1))    #np.array is numpy array
X=preprocessing.scale(X)    #feature scaling data
X_lately=X[-forecast_out:]  #X_lately has last 1% of data on which to make predicitons
X=X[:-forecast_out] #X has only labelled data now
df.dropna(inplace=True) #drop NaN values

y=np.array(df['label'])

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)   #split data into training and cross validation

##--------UNCOMMENT TO RETRAIN MODEL WITH NEW DATA----------- 
#clf=LinearRegression(n_jobs=-1) #n_jobs=-1 denotes parallalizing as much as possible
#clf.fit(X_train,y_train)    #train Model

#with open('GoogleStockModel.pickle','wb') as f:
#    pickle.dump(clf,f)

pickle_in=open('GoogleStockModel.pickle','rb')
clf=pickle.load(pickle_in)

confidence=clf.score(X_test,y_test) #Accuravy on cross validation data

forecast_set=clf.predict(X_lately)  #predict on Test data

style.use('ggplot')

df['Forecast']=np.nan  #Create a column named Forecast having NaN values
last_date=df.iloc[-1].name  #pickup last date from dataframe(-1 is for first from last)
last_unix=last_date.timestamp() #convert into unix datetime
one_day=24*60*60
next_unix=last_unix+one_day #next day's date
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day #next day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]    #create empty rows in dataframe so that area for forecast is reserved on plot

df['Adj. Close'].plot() #plot the existing stock price graph
df['Forecast'].plot()   #plot the predicted stock price for next month
plt.legend(loc=4)   #loc=4 means lower right
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
