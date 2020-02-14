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

    #dec - feb is winter, then spring, summer, fall etc
    seasons = [0,0,1,1,1,2,2,2,3,3,3,0]
    season = seasons[month-1]

    # sleep: 00-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
    times_of_day = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 0]
    time_of_day = times_of_day[h]

    return [h,t.day ,t.dayofweek, month , interval , season, time_of_day]

def merge_additional_features(df):
    """
    Generate aditional features from timestamp then append to exixting data
    """
    col = ["hour","day" ,"dayofweek", "month" , "interval" , "season", "time_of_day"]
    additional_featues = pd.DataFrame(data = [features_from_timestamp(i) for i in df.index ],columns=col).set_index(df.index)
    data = df.merge(additional_featues,on="dt")
    data.sort_index(inplace=True) #make sure data is sorted by date

    return data

def get_data(data_path= "../Dataset/household_power_consumption_data.zip",do_profile=False,with_app_stat = False):

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
    if with_app_stat :
        df_with_app_status =  appliances_status(df)
        grouped = df_with_app_status.groupby(pd.Grouper(freq='1h', base=0, label='right')).agg({"consumption": lambda x : np.sum(x)/60,"Set1": "any", "Set2": "any","Set3": "any"})
        data = grouped * 1
    else:
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

def appliances_status(df):
    df_with_app_status = df.copy()
    df_with_app_status["Set1"] = df_with_app_status.Sub_metering_1 > 0.0
    df_with_app_status["Set2"] = df_with_app_status.Sub_metering_2 > 0.0
    df_with_app_status["Set3"] = df_with_app_status.Sub_metering_3 > 0.0
    df_with_app_status.drop(["Sub_metering_1","Sub_metering_2","Sub_metering_3"],axis=1,inplace=True)
    return df_with_app_status
