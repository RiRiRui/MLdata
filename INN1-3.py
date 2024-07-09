#!/usr/bin/env python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"/nobackup/hnlv24/NEW/FDNN_tuning.csv")
tune_input = data.iloc[:, 0:9]

tune_output = data.iloc[:, 9:19]

tune_input.columns =['M1', 'M2', 'D1', 'D2','G1','G2','G3','G4','G5',]
tune_output=pd.DataFrame(tune_output)

encoder = OneHotEncoder(handle_unknown='ignore')
data_input2 = tune_input.iloc[:, :2]
data_input3 = tune_input.iloc[:, 2:9]

encoder_df = pd.DataFrame(encoder.fit_transform(tune_input[['M1', 'M2']]).toarray())
data_input3 = data_input3.join(encoder_df)

np.random.seed(1)
idx = np.random.permutation(data_input3.shape[0])

X = data_input3.to_numpy()
X = X[idx, :]

Y = tune_output.to_numpy()
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


comsol_data = np.array([[0.3053521,0.2327784,0.3396666,0.2549892,0.359215,0.2823146,0.3668505,0.2957246,0.3661356,0.3115428],
[0.3915866,0.3703096,0.397361,0.4018752,0.3976159,0.4183231,0.3855821,0.4086282,0.3773715,0.403804],
[0.3519826,0.3608028,0.3473114,0.355884,0.3437361,0.3522185,0.340369,0.3487421,0.3373854,0.3457115],
[0.3713999,0.2926864,0.379599,0.3173017,0.3820836,0.3409757,0.3821801,0.3556083,0.3804561,0.366144],
[0.2924927,0.3438643,0.2753719,0.3009164,0.2911272,0.2863779,0.3065255,0.2813407,0.3199983,0.2974519],
[0.4256682,0.3795813,0.4273402,0.3834347,0.4265502,0.3886498,0.4244066,0.3889225,0.4214796,0.3904281],
[0.4108454,0.3719622,0.4211736,0.377365,0.4248126,0.3805885,0.4228973,0.3879639,0.4208339,0.3899879],
[0.3588031,0.2813648,0.3624564,0.2947076,0.3662703,0.314649,0.3658651,0.3249822,0.3655121,0.3374052],
[0.3505054,0.2632236,0.3696604,0.2973574,0.3711815,0.3079905,0.3723264,0.3233425,0.3755981,0.3376277],
[0.4219925,0.3796361,0.4249455,0.3827663,0.4228495,0.3866675,0.4211138,0.3877657,0.4175536,0.3902936],
[0.2391998,0.3526344,0.2194403,0.255299,0.2506351,0.2026121,0.2724555,0.1774958,0.2866425,0.1728771],
[0.2564407,0.2213241,0.2756677,0.1905176,0.2925339,0.1833776,0.3019195,0.1849736,0.3060488,0.186873],
[0.4154929,0.4281534,0.3971734,0.4103588,0.387477,0.3999433,0.3783481,0.3910374,0.3730099,0.3856848],
[0.3205504,0.2778236,0.3524941,0.3170092,0.3724035,0.3583525,0.3797217,0.3776613,0.3795321,0.3833794],
[0.208764,0.2957439,0.2098576,0.3084636,0.2129559,0.3303212,0.2140334,0.3345476,0.2145984,0.340833],
[0.3383847,0.3373593,0.3449962,0.3424924,0.3485262,0.3462361,0.349835,0.3483465,0.3499408,0.349394],
[0.4142198,0.363061,0.418087,0.3678009,0.4189144,0.3704769,0.4176054,0.3711945,0.4169095,0.3705114],
[0.3491656,0.2725096,0.3657162,0.2844988,0.3793145,0.310422,0.3856424,0.3332548,0.384161,0.334251],
[0.2206476,0.1932005,0.2480608,0.1852078,0.2610233,0.1842094,0.2627646,0.1818663,0.2624371,0.1798405]])

SNN_model1 = keras.models.load_model(r"/nobackup/hnlv24/NEW/model.l7.nrd1100.h5")      # load SNN model from saved location
SNN_model = keras.models.clone_model(SNN_model1)
SNN_model.set_weights(SNN_model1.get_weights())
for layer in SNN_model.layers:
  layer.trainable = False
  layer._name = layer.name + str("_2")
SNN_model._name = "SNN_newname"
loss_train_list = []
loss_val_list = []
acc_list = []
acc_comsol = []
epc_list = []
patiance_num = 500

epc = 50000
nrd = 3
loss_train = []
loss_val = []


input_shape = X.shape[1]
output_shape = Y.shape[1]


# Network Architecture
input_tensor = keras.Input(shape = (output_shape))

x = keras.layers.Dense(nrd, activation = None)(input_tensor)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)

x = keras.layers.Dense(nrd, activation = None)(x)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)

x = keras.layers.Dense(nrd, activation = None)(x)
x = keras.layers.LeakyReLU(alpha = 0.15)(x)
x = keras.layers.Dropout(0.01)(x)




output_tensor = keras.layers.Dense(input_shape, activation = None)(x)

inverse_model = keras.models.Model(input_tensor, output_tensor)

tandem_model_input = keras.Input(shape=(output_shape, ))
intermediate = inverse_model(tandem_model_input)
tandem_output = SNN_model(intermediate)
tandem_model = keras.models.Model(tandem_model_input, tandem_output, name = 'tandem_model')

tandem_model.compile(loss = keras.losses.MeanSquaredError(),
              optimizer = 'Adam',
              metrics = ['mean_absolute_error'])


callback_list = [keras.callbacks.ReduceLROnPlateau(monitor="mean_absolute_error", factor=0.1, patience=patiance_num, min_lr=1e-6), 
                 keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=patiance_num, restore_best_weights=True)]

history = tandem_model.fit(Y_tr, Y_tr, epochs = epc, callbacks = callback_list, validation_data = (Y_val, Y_val),verbose = 0)


loss_train = history.history['mean_absolute_error']         # Produces a learning curve graph showing training and validation curves
loss_val = history.history['val_mean_absolute_error']


epc_list=[]
acc_list=[]
epc_list.append(len(loss_val)-patiance_num)
loss_train_list.append(loss_train)
loss_val_list.append(loss_val)
val_pred = tandem_model.predict(Y_val, verbose = 0)
acc_mean = []

for z in range(len(val_pred)):
    acc = 1 - np.sum(abs(val_pred[z, :] - Y_val[z, :])) / np.sum(Y_val[z, :])
    acc_mean.append(acc)
acc_list.append(np.mean(acc_mean))

test_pred_2 = inverse_model.predict(comsol_data)
features = test_pred_2* (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
Diameter_1 = []
Diameter_2 = []
G1 = []
G2 = []
G3 = []
G4 = []
G5 = []
Material_1 = []
Material_2 = []

for i in range(len(comsol_data)):
    # Diameter 1
    unique_values = data_input3['D1'].unique()
    idx = np.argmin(np.abs(features[i, 0] - unique_values))
    Diameter_1.append(unique_values[idx])

    # Diameter 2
    unique_values = data_input3['D2'].unique()
    idx = np.argmin(np.abs(features[i, 1] - unique_values))
    Diameter_2.append(unique_values[idx])

    # G1
    unique_values = data_input3['G1'].unique()
    idx = np.argmin(np.abs(features[i, 2] - unique_values))
    G1.append(unique_values[idx])

    # G2
    unique_values = data_input3['G2'].unique()
    idx = np.argmin(np.abs(features[i, 3] - unique_values))
    G2.append(unique_values[idx])

    # G3
    unique_values = data_input3['G3'].unique()
    idx = np.argmin(np.abs(features[i, 4] - unique_values))
    G3.append(unique_values[idx])

    # G4
    unique_values = data_input3['G4'].unique()
    idx = np.argmin(np.abs(features[i, 5] - unique_values))
    G4.append(unique_values[idx])

    # G5
    unique_values = data_input3['G5'].unique()
    idx = np.argmin(np.abs(features[i, 6] - unique_values))
    G5.append(unique_values[idx])

    # Material 1
    names = ['Al2O3', 'GaAs', 'GaSb', 'Ge', 'ITO', 'Si', 'Si3N4', 'TiO2', 'ZnO']
    idx = np.argmax(features[i, 7:16])
    Material_1.append(names[idx])

    # Material 2
    names = ['Ag', 'Al', 'Au', 'Cu', 'Li', 'Ti']
    idx = np.argmax(features[i, 16:22])
    Material_2.append(names[idx])
    
INN_pred = pd.DataFrame({'Material_1': Material_1, 
                         'Material_2': Material_2,  
                         'G1':G1,
                         'G2':G2,
                         'G3':G3,
                         'G4':G4,
                         'G5':G5,
                         'Diameter_1': Diameter_1, 
                         'Diameter_2': Diameter_2} )



inverse_model.save("model.L3.nrd%.f.h5" % (nrd))
print(epc_list)
print(acc_list)
print(INN_pred)
print("model.INN.L3.nrd%.f.h5" % (nrd))

