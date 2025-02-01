import yfinance  as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression
from tkinter import *
from tkinter import messagebox

def get_stock_price(symbol,strat_date,end_date):
    stock_date=yf.download(symbol,start=strat_date,end=end_date)
    return stock_date

def preprocess_data(data):
    data=data['Close'] # get the close price
    data['prediction']=data[['Close']].shift(-30) # shift the data by 30 days
    
    # create the feature data set (X) and convert it to a numpy array and remove the last 30 rows/days

    scaler=MinMaxScaler(feature_range=(0,1))
    data_scaled=scaler.fit_transform(data[['close']])


    # create the target data set (y) and convert it to a numpy array and get all of the target values except the last 30 rows/days
    x=data_scaled[:-30] # remove the last 30 days
    y=data['Prediction'][:-30]  # remove the last 30 days

    # split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
#model building
def build_model(X_train, y_train):
    model=LinearRegression()
    model.fit(X_train,y_train) # train the model
    return model




