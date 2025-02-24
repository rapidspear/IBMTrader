import numpy as np

from IBMData import *
import tensorflow as tf
from keras import *
from keras.layers import *
import pandas as pd


train_data, test_data, target_train, target_test, scaled_data, scaler, n = getTrainingData()

print(train_data.shape)
print(test_data.shape)
print(target_train.shape)

target_train_array = np.stack(target_train.values)



train_tensor = tf.convert_to_tensor(train_data, dtype = tf.float32)
target_train_tensor = tf.convert_to_tensor(target_train_array, dtype = tf.float32)
test_tensor = tf.convert_to_tensor(test_data, dtype = tf.float32)


print("======================================================================================================================")

timesteps = 1

train_data_reshaped = tf.reshape(train_tensor, (train_tensor.shape[0], timesteps, train_tensor.shape[1]))
test_data_reshaped = tf.reshape(test_tensor, (test_tensor.shape[0], timesteps, test_tensor.shape[1]))


model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, train_tensor.shape[1])))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_data_reshaped, target_train_tensor, batch_size=10, epochs=40)

train_predict = model.predict(train_data_reshaped)
test_predict = model.predict(test_data_reshaped)

train_predict = scaler.inverse_transform(np.hstack((np.zeros((train_predict.shape[0], scaled_data.shape[1] - 1)), train_predict)))
test_predict = scaler.inverse_transform(np.hstack((np.zeros((test_predict.shape[0], scaled_data.shape[1] - 1)), test_predict)))

train_predict = train_predict[:, -1]
test_predict = test_predict[:, -1]


train_scaled = scaler.inverse_transform(scaled_data[n:].values)
test_scaled = scaler.inverse_transform(scaled_data[:n].values)

print(train_scaled.shape)
print(test_scaled.shape)


print('//////////////////////////////////////////////////////////////////////////////////////////////')
#print(np.mean(test_predict - test_scaled[:, 3]**2))

train_score = np.sqrt(np.mean((train_predict - train_scaled[:, 3])**2))
test_score = np.sqrt(np.mean((test_predict - test_scaled[:, 3])**2))



#train_score = np.sqrt(np.mean((train_predict - scaled_data[n+1:]['close'].values.reshape(-1, 1).astype('float')))**2)
#test_score = np.sqrt(np.mean((test_predict - scaled_data[:n]['close'].values.reshape(-1, 1).astype('float')))**2)

# train_score = np.sqrt(np.mean((train_predict - scaler.inverse_transform(scaled_data[n+1:, 3].reshape(-1, 1)))**2))
# test_score = np.sqrt(np.mean((test_predict - scaler.inverse_transform(scaled_data[:n, 3].reshape(-1, 1)))**2))

print(train_score)
print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')
