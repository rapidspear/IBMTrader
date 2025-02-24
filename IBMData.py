import pandas
import requests
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def getTrainingData():

    training_data = requests.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo")

    training_data_dict = training_data.json()
    training_df = pd.DataFrame(training_data_dict)
    print(training_df)


    training_df.drop("Meta Data", axis="columns", inplace=True)
    training_df.drop(['1. Information', '2. Symbol', '3. Last Refreshed', '4. Output Size', '5. Time Zone', ],
                     axis=0,
                     inplace=True)


    training_df, target_df, scaler, old_df = reformatData(training_df)


    return seperateTrainingTesting(training_df, target_df, getEndOfYearIndex(old_df), scaler, old_df)


def reformatData(df):
    new_df = df
    new_df['open'] = df['Time Series (Daily)'].apply(lambda x: x['1. open'])
    new_df['high'] = df['Time Series (Daily)'].apply(lambda x: x['2. high'])
    new_df['low'] = df['Time Series (Daily)'].apply(lambda x: x['3. low'])
    new_df['close'] = df['Time Series (Daily)'].apply(lambda x: x['4. close'])

    #new_df['volume'] = df['Time Series (Daily)'].apply(lambda x: x['5. volume'])
    new_df = new_df.drop(columns=['Time Series (Daily)'])
    old_df = new_df
    target_df = new_df['close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    new_df = scaler.fit_transform(new_df)


    return new_df, target_df, scaler, old_df



def getEndOfYearIndex(df):
    return df.index.get_loc('2024-06-03')


def seperateTrainingTesting(df, target_df, n, scaler, old_df):
    training_data = df[n:, :]
    testing_data = df[:n, :]

    return training_data, testing_data, target_df[n:], target_df[n:len(target_df)], old_df, scaler, n


def dataSeperator(df):
    n = int(df.size / 4) #num of columns = 4
    df = pd.DataFrame(df)
    new_df = pd.DataFrame()
    new_df['open'] = df[:n]

    print(df[n: n * 2])
    new_df['high'] = df[n: n * 2]
    print(df[n * 2:n * 3])
    new_df['low'] = df[n * 2:n * 3]
    print(df[n * 3:])
    new_df['close'] = df[n * 3:]

    return new_df
