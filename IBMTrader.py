

import numpy as np

from IBMData import *
import tensorflow as tf
from keras import *
from keras.layers import *
import matplotlib.pyplot as plt



def main():
    scaled_data, train_data, test_data, target_train, target_test, old_df, scaler, n = getTrainingData()

    #target_train_array = np.stack(target_train.values)



    train_tensor = tf.convert_to_tensor(train_data, dtype = tf.float32)
    target_train_tensor = tf.convert_to_tensor(target_train, dtype = tf.float32)
    test_tensor = tf.convert_to_tensor(test_data, dtype = tf.float32)
    timesteps = 1


    train_data_reshaped = tf.reshape(train_tensor, (train_tensor.shape[0], timesteps, train_tensor.shape[1]))
    test_data_reshaped = tf.reshape(test_tensor, (test_tensor.shape[0], timesteps, test_tensor.shape[1]))


    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, train_tensor.shape[1])))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting
    model.add(BatchNormalization())
    model.add(LSTM(100, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(train_data_reshaped, target_train_tensor, batch_size=10, epochs=15)

    train_predict = model.predict(train_data_reshaped)
    test_predict = model.predict(test_data_reshaped)


    #train_predict = scaler.inverse_transform(train_predict)
    #test_predict = scaler.inverse_transform(test_predict)
    '''
    print(train_predict)
    print('\n')
    print(test_predict)
    print('\n')
    '''
    train_predict = scaler.inverse_transform(np.hstack((np.zeros((train_predict.shape[0], scaled_data.shape[1] - 1)), train_predict)))
    test_predict = scaler.inverse_transform(np.hstack((np.zeros((test_predict.shape[0], scaled_data.shape[1] - 1)), test_predict)))

    train_predict = train_predict[:, -1]
    test_predict = test_predict[:, -1]



    train_unscaled = np.hstack(old_df.iloc[n:, -1])
    test_unscaled = np.hstack(old_df.iloc[:n, -1])

    train_score = np.mean(np.abs(train_predict - train_unscaled))
    test_score = np.mean(np.abs(test_predict - test_unscaled))

    print(f'Train Score: {train_score:.2f} RMSE')
    print(f'Test Score: {test_score:.2f} RMSE')

    plt.plot(test_predict, label="Actual")
    plt.plot(test_unscaled, label="Predicted")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()