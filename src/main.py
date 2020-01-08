from utils import get_data
from models import *

def main():
    (xtrain, ytrain, xtest, ytest) = get_data()
    model = IntervalModelLightGBM()
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest,ytest)
    # model.calculate_errors()
    # model.plot_predictions()

if __name__ == '__main__':
    main()
