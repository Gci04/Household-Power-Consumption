import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from sklearn.metrics import mean_squared_error

def main():
    (xtrain, xtest, ytrain, ytest) = get_data()
    model = EnergyConsump(xtrain,ytrain)
    model.train()
    pred = model.predict(xtest)
    mse = mean_squared_error(pred,ytest)
    print("MSE = {}".format(mse))
    model.plot()

if __name__ == '__main__':
    main()
