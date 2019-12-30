import numpy as np
import pandas as pd
import pandas_profiling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM, Dropout, Dense

#Extract features from timestamp
def features_from_timestamp(t):
    h = t.hour
    idx = np.searchsorted(list(range(3,25,3)),h,side='right')
    interval = np.arange(3,25,3)[idx]
    if  interval == 24 : interval = 0
    month = t.month
    season = (month in [11,12,1,2,3]) * 1 # 0: summer, 1:  winter
    return [h,t.day ,t.dayofweek, month , interval , season]

def merge_additional_features(df):
    """
    Generate aditional features from timestamp then append to exixting data
    """
    col = ["hour","day" ,"dayofweek", "month" , "interval" , "season"]
    additional_featues = pd.DataFrame(data = [features_from_timestamp(i) for i in df.index ],columns=col).set_index(df.index)
    data = df.merge(additional_featues,on="dt")
    data.sort_index(inplace=True) #make sure data is sorted by date

    return data

def get_data(data_path= "../Dataset/household_power_consumption_data.zip",do_profile=False):

    df = pd.read_csv(data_path, sep=';',parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan','?'],index_col='dt')
    print("[Reading Data] : DONE")
    df.drop(["Global_active_power","Global_reactive_power","Voltage","Global_intensity"],axis=1,inplace=True)

    if do_profile:
        profile = pandas_profiling.ProfileReport(df)
        profile.to_file("report.html")
        print("[Profiling data finished]")

    #fill nan values with column average
    for j in range(0,3):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

    print("[Filling Missing Values] : DONE")

    df["consumption"] = df.iloc[:,:].sum(axis=1) #consumption -> Energy consumption
    grouped = df["consumption"].groupby(pd.Grouper(freq='1h', base=0, label='right')).sum()
    data = pd.DataFrame(grouped/60)

    data = merge_additional_features(data)
    print("[Extracting features from timestamp] : DONE")

    xtrain = data.loc["2006":"2010"]
    ytrain = xtrain.pop("consumption")

    xtest = data.loc["2010":]
    ytest = xtest.pop("consumption")

    print("[Train Test split] : DONE")

    return xtrain, ytrain, xtest, ytest

class GradientBoostingPredictionIntervals(object):
    """
    Model that produces prediction intervals with a GradientBoostingRegressor
    """

    def __init__(self,lower_alpha = 0.25, upper_alpha = 0.75 , **kwargs):
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha

        # Three separate models
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=self.lower_alpha, **kwargs)
        self.mid_model = GradientBoostingRegressor(loss="ls", **kwargs)
        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=self.upper_alpha, **kwargs)
        self.predictions = None

    def fit(self,xtrain,ytrain):
        """
        fit models (Upper bound , lower bound and mid.)
        """
        self.lower_model.fit(xtrain, ytrain)
        self.mid_model.fit(xtrain, ytrain)
        self.upper_model.fit(xtrain, ytrain)

    def predict(self,X,y):
        """
        predict the upper, lower bound and middle
        """
        predictions = pd.DataFrame(y)
        predictions["lower"] = self.lower_model.predict(X)
        predictions["mid"] = self.mid_model.predict(X)
        predictions["upper"] = self.upper_model.predict(X)
        self.predictions = predictions

        return predictions

    def calculate_errors(self):
        assert(type(self.predictions) != None), "No predictions done by model yet!!"

        self.predictions['absolute_error_lower'] = (self.predictions['lower'] - self.predictions["consumption"]).abs()
        self.predictions['absolute_error_upper'] = (self.predictions['upper'] - self.predictions["consumption"]).abs()

        self.predictions['absolute_error_interval'] = (self.predictions['absolute_error_lower'] + self.predictions['absolute_error_upper']) / 2
        self.predictions['absolute_error_mid'] = (self.predictions['mid'] - self.predictions["consumption"]).abs()

        self.predictions['in_bounds'] = self.predictions["consumption"].between(left=self.predictions['lower'], right=self.predictions['upper'])

    def plot_predictions(self):
        """Plot the prediction intervals"""
        assert(type(self.predictions) != None), "No predictions done by model yet!!"

        # Plot first 10 samples
        fig = plt.figure(figsize=(19.20,10.80))
        # plt.plot(self.predictions.index[:10], self.predictions.lower[:10],label="Lower",linewidth=3)
        plt.plot(self.predictions.index[:10], self.predictions.mid[:10],label="Prediction",linewidth=3)
        # plt.plot(self.predictions.index[:10], self.predictions.upper[:10], label="Upper",linewidth=3)
        plt.plot(self.predictions.index[:10], self.predictions.consumption[:10],label="consumption",linewidth=3)
        plt.fill_between(self.predictions.index[:10], self.predictions.lower[:10], self.predictions.upper[:10],alpha=0.3, facecolor=colors[8],label='interval')
        plt.legend(loc=0,prop={'size': 10})
        plt.ylabel("Energy Consumption (Wh)",fontsize=20.0,fontweight="bold")
        plt.xlabel("Time",fontsize=20.0,fontweight="bold")
        plt.show()

#Data reshape for LSTM method
def reshape_for_lstm(df, lookback=1):
    l = len(df) - lookback
    X = df
    y = X[lookback:]
    res = []

    for i in range(l):
        res.append(X[i:i+lookback])
    return np.array(res), y

class EnergyConsumpLSTM(object):
    """LSTM Keras model for prediction Household Power Consumption"""
    def __init__(self, X,y):
        self.trained = False
        self.lookback = X.shape[1]
        self.X = X
        self.y = y
        self.model = None
        self.define_model()
        self.history = None

    def define_model(self):
        self.model = Sequential()
        self.model.add(LSTM(10, input_shape=(self.lookback, self.X.shape[2])))
        # model.add(Dropout(0.2))
        self.model.add(Dense(self.y.shape[1]))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        print(self.model.summary())

    def train(self,epochs = 2 ,BS = 7000):
        self.history = self.model.fit(self.X, self.y, epochs=epochs, batch_size=BS, validation_split=0.1,
                     verbose=1, shuffle=False)
        self.trained = True

    def predict(self,xtest):
        assert(self.trained == True), "Model not trained!!"
        return self.model.predict(xtest)

    def plot(self):
        assert(self.trained == True), "Model not trained!!"
        plt.figure(figsize=(12,8))
        plt.plot(self.history.epoch, self.history.history['loss'])
        plt.plot(self.history.epoch, self.history.history['val_loss'])
        plt.title("model loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
