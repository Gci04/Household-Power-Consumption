import numpy as np
import pandas as pd
import pandas_profiling

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

def errors_calculation(predictions):
    assert(type(predictions) != None), "No Predictions!!"

    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["consumption"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["consumption"]).abs()

    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["consumption"]).abs()

    predictions['in_bounds'] = predictions["consumption"].between(left=predictions['lower'], right=predictions['upper'])

    return predictions

def appliances_status():
    pass 
