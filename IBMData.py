import requests
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def getTrainingData():
    training_data = requests.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo")

    original_data_dict = training_data.json()
    original_df = pd.DataFrame(original_data_dict)

    original_df.drop("Meta Data", axis="columns", inplace=True)
    original_df.drop(['1. Information', '2. Symbol', '3. Last Refreshed', '4. Output Size', '5. Time Zone', ],
                     axis=0,
                     inplace=True)

    new_df, target_df, scaler, old_df = ReformatData(original_df)

    return SeparateTrainingTesting(new_df, target_df, getEndOfYearIndex(old_df), scaler, old_df)


def ReformatData(df):
    new_df = df
    new_df['volume'] = df['Time Series (Daily)'].apply(lambda x: x['5. volume'])
    new_df['open'] = df['Time Series (Daily)'].apply(lambda x: x['1. open'])
    new_df['high'] = df['Time Series (Daily)'].apply(lambda x: x['2. high'])
    new_df['low'] = df['Time Series (Daily)'].apply(lambda x: x['3. low'])
    new_df['close'] = df['Time Series (Daily)'].apply(lambda x: x['4. close'])


    new_df = new_df.drop(columns=['Time Series (Daily)'])
    old_df = new_df.copy()

    target_df = new_df['close']

    scaler = MinMaxScaler(feature_range=(0, 1))


    for col in new_df.columns:
        new_df[col] = scaler.fit_transform(new_df[[col]])


    new_df = scaler.fit_transform(new_df) #Inefficient yes, but converts the panda dataframe into a numpy array. DOESN'T AFFECT THE DATA! Change when you get the chance


    return new_df, target_df, scaler, old_df


def SeparateTrainingTesting(df, target_df, n, scaler, old_df):
    training_data = df[n:, :]
    testing_data = df[:n, :]



    return df, training_data, testing_data, target_df[n:], target_df[n:len(target_df)], old_df.astype(float), scaler, n


def getEndOfYearIndex(df):
    return df.index.get_loc('2024-06-03')
