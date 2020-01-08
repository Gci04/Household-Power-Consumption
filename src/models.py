import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from utils import errors_calculation

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM, Dropout, Dense
import keras.backend as K

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

        self.predictions = errors_calculation(self.predictions)

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


class IntervalModelLightGBM(object):
    """
    Interval prediction model having LightGBM as a base model
    """
    def __init__(self,lower_alpha = 0.05, upper_alpha = 0.95 , **kwargs):
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.predictions = None

        # Three separate models
        self.lower_model_lgbm = lgb.LGBMRegressor(objective="quantile",alpha=self.lower_alpha,n_estimators=700,max_depth=15,n_jobs=-1, learning_rate= 0.1,num_leaves= 900)
        self.mid_model_lgbm = lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=700,max_depth=15,n_jobs=-1, learning_rate= 0.1,num_leaves= 900)
        self.upper_model_lgbm = lgb.LGBMRegressor(objective="quantile",alpha=self.upper_alpha,n_estimators=700,max_depth=15,n_jobs=-1, learning_rate= 0.1,num_leaves= 900)

    def fit(self,xtrain,ytrain):
        """
        fit models (Upper bound , lower bound and mid.)
        """
        self.lower_model_lgbm.fit(xtrain, ytrain)
        self.mid_model_lgbm.fit(xtrain, ytrain)
        self.upper_model_lgbm.fit(xtrain, ytrain)
        print("[Model Fit] : DONE")

    def predict(self,X,y):
        """
        predict the upper, lower bound and middle
        """
        predictions = pd.DataFrame(y)
        predictions["lower"] = self.lower_model_lgbm.predict(X)
        predictions["mid"] = self.mid_model_lgbm.predict(X)
        predictions["upper"] = self.upper_model_lgbm.predict(X)

        #clip negative predictions to zero
        predictions.loc[predictions['lower'] < 0.0, "lower"] = 0.0
        predictions.loc[predictions['mid'] < 0.0, "mid"] = 0.0

        self.predictions = predictions
        
        return predictions

    def calculate_errors(self):

        self.predictions = errors_calculation(self.predictions)

    def plot_predictions(self,n=10):
        """Plot the prediction intervals"""
        assert(type(self.predictions) != None), "No predictions done by model yet!!"

        # Plot first 10 samples
        fig = plt.figure(figsize=(19.20,10.80))
        # plt.plot(self.predictions.index[:10], self.predictions.lower[:10],label="Lower",linewidth=3)
        plt.plot(self.predictions.index[:n], self.predictions.mid[:n],label="Prediction",linewidth=3)
        # plt.plot(self.predictions.index[:10], self.predictions.upper[:10], label="Upper",linewidth=3)
        plt.plot(self.predictions.index[:n], self.predictions.consumption[:n],label="consumption",linewidth=3)
        plt.fill_between(self.predictions.index[:n], self.predictions.lower[:n], self.predictions.upper[:n],alpha=0.3, facecolor=colors[8],label='interval')
        plt.legend(loc=0,prop={'size': 10})
        plt.ylabel("Energy Consumption (Wh)",fontsize=20.0,fontweight="bold")
        plt.xlabel("Time",fontsize=20.0,fontweight="bold")
        plt.show()

class DeepQuantileRegression(object):
    """Model for Predictions of intervals using Deep Quantile Regression (Deep Neural network)."""

    def __init__(self,lower_alpha = 0.1, upper_alpha = 0.9 , **kwargs):
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha
        self.predictions = None

        self.scaler = StandardScaler()
        K.clear_session()

        self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        self.model_lower = self.__BaseModel()
        self.model_lower.compile(loss=lambda y,f: self.__quantile_loss(self.lower_alpha,y,f), optimizer='adadelta')

        self.model_middle = self.__BaseModel()
        self.model_middle.compile(loss=lambda y,f: self.__quantile_loss(0.5,y,f), optimizer='adadelta')

        self.model_upper = self.__BaseModel()
        self.model_upper.compile(loss=lambda y,f: self.__quantile_loss(self.upper_alpha,y,f), optimizer='adadelta')


    def __quantile_loss(self,q,y,f):
        # q: Quantile to be evaluated, e.g., 0.5 for median.
        # y: True value.
        # f: Fitted (predicted) value.
        e = (y-f)
        return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

    def __BaseModel(self):
        model = Sequential()
        model.add(Dense(units=10, input_dim=6,activation='relu'))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units = 1,activation="relu"))

        return model

    def fit(self,xtrain,ytrain):
        std_xtrain = self.scaler.fit_transform(xtrain)

        _ = self.model_lower.fit(std_xtrain, ytrain, epochs=2000, batch_size=32, verbose=0,validation_split=0.2, callbacks=[self.early_stop])
        _ = self.model_middle.fit(std_xtrain, ytrain, epochs=2000, batch_size=32, verbose=0,validation_split=0.2, callbacks=[self.early_stop])
        _ = self.model_upper.fit(std_xtrain, ytrain, epochs=2000, batch_size=32, verbose=0,validation_split=0.2, callbacks=[self.early_stop])

        print("[Model Fit] : DONE")

    def predict(self,xtest,y):
        """
        predict the upper, lower bound and middle
        """
        X = self.scaler.transform(xtest)

        predictions = pd.DataFrame(y)
        predictions["lower"] = self.model_lower.predict(X)
        predictions["mid"] = self.model_middle.predict(X)
        predictions["upper"] = self.model_upper.predict(X)

        self.predictions = predictions

        return predictions

    def calculate_errors(self):

        self.predictions = errors_calculation(self.predictions)

    def plot_predictions(self,n=10):
        """Plot the prediction intervals"""
        assert(type(self.predictions) != None), "No predictions done by model yet!!"

        # Plot first 10 samples
        fig = plt.figure(figsize=(19.20,10.80))
        # plt.plot(self.predictions.index[:10], self.predictions.lower[:10],label="Lower",linewidth=3)
        plt.plot(self.predictions.index[:n], self.predictions.mid[:n],label="Prediction",linewidth=3)
        # plt.plot(self.predictions.index[:10], self.predictions.upper[:10], label="Upper",linewidth=3)
        plt.plot(self.predictions.index[:n], self.predictions.consumption[:n],label="consumption",linewidth=3)
        plt.fill_between(self.predictions.index[:n], self.predictions.lower[:n], self.predictions.upper[:n],alpha=0.3, facecolor=colors[8],label='interval')
        plt.legend(loc=0,prop={'size': 10})
        plt.ylabel("Energy Consumption (Wh)",fontsize=20.0,fontweight="bold")
        plt.xlabel("Time",fontsize=20.0,fontweight="bold")
        plt.show()
