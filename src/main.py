import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from sklearn.metrics import mean_squared_error

def main():
    (xtrain, ytrain, xtest, ytest) = get_data()
    model = IntervalModelLightGBM()
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest,ytest)
    model.calculate_errors()
    model.plot_predictions()

if __name__ == '__main__':
    main()
