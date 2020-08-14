import pandas as pd 
import numpy as np
from datetime import datetime
import math

def clean_data(df):
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    
    df["hour"]=df["hour"].apply(lambda x : np.round((math.cos(2*math.pi/24*x)),3))

    ### target encoding season
    season_map =df.groupby("season").mean()["count"].to_dict()
    df["season"]=df["season"].map(season_map)

    ### target encoding holiday
    holiday_map=df.groupby("holiday").mean()["count"].to_dict()
    df["holiday"] = df["holiday"].map(holiday_map)

    ###target encoding working day

    working_day_map=df.groupby("workingday").mean()["count"].to_dict()
    df["workingday"] = df.workingday.map(working_day_map)

    ##target encoding weather 

    weather_map=df.groupby("weather").mean()["count"].to_dict()
    df.weather=df.weather.map(weather_map)

    ##dropping atemp
    df= df.drop("atemp",axis=1)
    
    ### humidity cannot be zero

    df["humidity"] = df["humidity"].replace(0,df["humidity"].median())

    ##binning windspeed

    df["windspeed"]=pd.cut(df["windspeed"],bins=11,labels=False)


    ## BoxCoxing casual

    df["casual"] = (df["casual"])**(1/5)

    ##BoxCoxing Registered

    df["registered"] = (df["registered"])*(1/5)

    df = df.drop("datetime",axis=1)

    df.to_csv("../input/cleaned_train.csv",index=False)

if __name__=="__main__":

    df = pd.read_csv("../input/train.csv")

    clean_data(df)


