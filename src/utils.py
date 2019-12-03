import numpy as np
import pandas as pd
import pandas_profiling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM, Dropout, Dense

#Data reshape for LSTM method
def reshape_for_lstm(df, lookback=1):
    l = len(df) - lookback
    X = df
    y = X[lookback:]
    res = []

    for i in range(l):
        res.append(X[i:i+lookback])
    return np.array(res), y

#Extract features from timestamp
def features_from_timestamp(t):
    h = t.hour
    idx = np.searchsorted(list(range(3,25,3)),h,side='right')
    interval = np.arange(3,25,3)[idx]
    if  interval == 24 : interval = 0
    month = t.month
    season = (month in [11,12,1,2,3]) * 1 # 0: summer, 1:  winter
    return [h,t.day ,t.dayofweek, month , interval , season]


def get_data(data_path= "../Dataset/household_power_consumption_data.zip",do_profile=False):
    print("Reading Data ....")
    df = pd.read_csv(data_path, sep=';',parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan','?'],index_col='dt')
    if do_profile:
        profile = pandas_profiling.ProfileReport(df)
        profile.to_file("report.html")
        print("[Profiling data finished]")

    #fill nan values with column average
    for j in range(0,7):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

    #Generate aditional features from timestamp then append to exixting data
    col = ["hour","day" ,"dayofweek", "month" , "interval" , "season"]
    additional_featues = pd.DataFrame(data = [features_from_timestamp(i) for i in df.index ],columns=col).set_index(df.index)
    data = df.merge(additional_featues,on="dt")
    data.sort_index(inplace=True) #make sure data is sorted by date

    #Split data to train and test
    test = data['2010-10-26':] #one month for testing
    train = data[:'2010-10-26']

    #Standardize the data
    scaler = StandardScaler()
    train = scaler.fit_transform(train.values)
    test = scaler.transform(test.values)

    #Prepare data for LSTM input
    lookback = 10 #look back 10min
    Xtrain, ytrain = reshape_for_lstm(train,lookback)
    Xtest, ytest = reshape_for_lstm(test,lookback)
    print("[Finished reading & preprocessing Dataset]")

    return Xtrain, Xtest, ytrain, ytest

class EnergyConsump(object):
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
