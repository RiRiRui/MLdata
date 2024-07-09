#!/usr/bin/env python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel(r"/nobackup/hnlv24/NEW/data June2023.xlsx", sheet_name = "Sheet1")
data_input = data.iloc[:, :5]
data_output = data.iloc[:, 5:7]

encoder = OneHotEncoder(handle_unknown='ignore')
data_input2 = data_input.iloc[:, :2]
encoder_df = pd.DataFrame(encoder.fit_transform(data_input[['M1', 'M2']]).toarray())
data_input3 = data_input.iloc[:, 2:6]
data_input3 = data_input3.join(encoder_df)
np.random.seed(1)
idx = np.random.permutation(data_input.shape[0])
X = data_input3.to_numpy()
X = X[idx, :]
Y = data_output.to_numpy()
Y = Y[idx, :]

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
ns = X.shape[0]
ntr = int(ns * 0.8)
X_tr = X[:ntr, :]
Y_tr = Y[:ntr, :]
X_val = X[ntr:, :]
Y_val = Y[ntr:, :]

#prep for accuracy
loss_train_list = []
loss_val_list = []
acc_list = []
acc_comsol = []
epc_list = []

comsol_data = np.array([[0.375793,0.362459],
[0.409038,0.381499],
[0.305534,0.219595],
[0.201953,0.292527],
[0.347149,0.228337],
[0.456014,0.424635],
[4.13E-01,3.86E-01],
[3.63E-01,3.49E-01],
[4.08E-01,3.91E-01],
[3.79E-01,2.89E-01],
[2.00E-01,2.41E-01],
[3.95E-01,3.41E-01],
[4.63E-01,4.06E-01],
[2.92E-01,2.17E-01],
[4.32E-01,3.81E-01],
[2.43E-01,3.36E-01],
[3.19E-01,3.54E-01],
[4.62E-01,4.09E-01],
[2.30E-01,2.13E-01],
[3.76E-01,2.82E-01]])
Test_dataset_1 = np.array([[132,108,21.42135624,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
[106,94,4.651803616,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[124,116,12.93607486,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
[132,108,23.54267658,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
[94,146,17.17871555,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
[162,78,24.24978336,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
[146,126,2.731493399,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[132,122,10.2081528,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
[130,120,10.2081528,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[102,108,13.64318164,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
[100,90,5.81475452,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
[96,130,21.62236636,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
[162,102,9.400540957,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
[140,116,11.5218613,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
[98,84,10.81475452,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
[104,128,2.731493399,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
[142,120,0.710678119,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
[174,100,6.974134086,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
[106,92,6.974134086,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
[104,114,13.13708499,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0]])
scaler = MinMaxScaler()
scaler.fit(Test_dataset_1)
Test_dataset_1 = scaler.transform(Test_dataset_1)

epc = 50000
patiance_num = 500 #argu of model.fit()

#i looped it to find the best accuracy model. 
nrd = 3
loss_train = []
loss_val = []
input_shape = X.shape[1]
output_shape = Y.shape[1]

# Network Architecture
input_tensor = keras.Input(shape = (input_shape))

x = keras.layers.Dense(input_shape, activation = None)(input_tensor)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)

x = keras.layers.Dense(nrd, activation = None)(x)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)

x = keras.layers.Dense(nrd, activation = None)(x)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)

x = keras.layers.Dense(nrd, activation = None)(x)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)

output_tensor = keras.layers.Dense(2)(x)   #output

model1 = keras.models.Model(input_tensor, output_tensor)
model1.compile(loss = keras.losses.MeanAbsoluteError(),
               optimizer = 'Adam', # adam optimiz https://machinelearningjourney.com/index.php/2021/01/09/adam-optimizer/#:~:text=Adam%2C%20derived%20from%20Adaptive%20Moment%20Estimation%2C%20is%20an,of%20an%20exponentially%20decaying%20average%20of%20past%20gradients.
               metrics = ['mean_absolute_error'])
callback_list = [keras.callbacks.ReduceLROnPlateau(monitor="mean_absolute_error", factor=0.1, patience=patiance_num, min_lr=1e-6), 
                 keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=patiance_num, restore_best_weights=True)]
history = model1.fit(X_tr, Y_tr, epochs = epc, callbacks = callback_list, validation_data = (X_val, Y_val),verbose = 0)
loss_train = history.history['mean_absolute_error']         # Produces a learning curve graph showing training and validation curves
loss_val = history.history['val_mean_absolute_error']

epc_list.append(len(loss_val)-patiance_num)

loss_train_list.append(loss_train)
loss_val_list.append(loss_val)
val_pred = model1.predict(X_val, verbose = 0)
acc_mean = []
acc_comsol_mean = []
comsol_pred = model1.predict(Test_dataset_1, verbose = 0)
for z in range(len(val_pred)):
    acc = 1 - np.sum(abs(val_pred[z, :] - Y_val[z, :])) / np.sum(Y_val[z, :])
    acc_mean.append(acc)
acc_list.append(np.mean(acc_mean))



for w in range(len(comsol_pred)):
    acc_comsol_mean.append(1 - np.sum(abs(comsol_pred[w, :] - comsol_data[w, :])) / np.sum(comsol_data[w, :]))
acc_comsol.append(np.mean(acc_comsol_mean))

print("model.l3.nrd%.f.h5" % (nrd))
model1.save("model.l3.nrd%.f.h5" % (nrd))

print(epc_list)
print(acc_list)
print(acc_comsol)