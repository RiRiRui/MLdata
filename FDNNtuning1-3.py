#!/usr/bin/env python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"/nobackup/hnlv24/NEW/FDNN.csv")
tune_input = data.iloc[:, 0:9]
tune_output = data.iloc[:, 9:19]

tune_input.columns =['M1', 'M2', 'D1', 'D2', 'G1', 'G2', 'G3', 'G4', 'G5']
tune_output=pd.DataFrame(tune_output)

encoder = OneHotEncoder(handle_unknown='ignore')
data_input2 = tune_input.iloc[:, :2]
data_input3 = tune_input.iloc[:, 2:9]

encoder_df = pd.DataFrame(encoder.fit_transform(tune_input[['M1', 'M2']]).toarray())
data_input3 = data_input3.join(encoder_df)

np.random.seed(1)
idx = np.random.permutation(data_input3.shape[0])

Y = data_input3.to_numpy()
Y = Y[idx, :]

X = tune_output.to_numpy()
X = X[idx, :]

ns = X.shape[0]
ntr = int(ns * 0.89)
nv = int(ns * 0.99)
Y_tr = Y[:ntr, :]
X_tr = X[:ntr, :]
Y_val = Y[ntr:nv, :]
X_val = X[ntr:nv, :]

Y_com_val = Y[nv:, :]
X_com_val = np.array([[0.2024169,0.3006533,0.19252884,0.27134237,0.26572067,0.19199875,3.25E-01,2.14E-01,0.3504261,0.25088787],
[0.38895267,0.41819578,0.3734719,0.43595892,0.28797218,0.47998103,0.24920934,0.50181556,0.2586996,0.4178995],
[0.38732803,0.32263923,0.4185608,0.34785384,0.43917096,0.41550356,4.14E-01,4.12E-01,0.38676685,0.38985804],
[0.20465527,0.20298697,0.24398312,0.17347181,0.27597687,0.19184786,2.30E-01,0.33018437,0.200997,0.31825617],
[0.31637022,0.37264237,0.28789017,0.40172005,0.24275878,0.41083917,2.28E-01,0.4026772,0.2279731,0.3685098],
[0.30938676,0.34372008,0.31141457,0.34955776,0.20358396,0.30380163,2.40E-01,2.34E-01,0.26325879,0.24631464],
[0.37850016,0.3867515,0.32622102,0.41289553,0.33764678,0.39688402,4.24E-01,0.37547496,0.43637452,0.38846347],
[0.4502825,0.4419356,0.4418481,0.45254612,0.40030488,0.41575703,3.79E-01,3.90E-01,0.36826348,0.37798437],
[0.24078685,0.32405606,0.23507327,0.20269953,0.33197522,0.20904091,3.83E-01,0.32522926,0.39425454,0.38534728],
[0.30295068,0.16743106,0.32707155,0.18632886,0.41090274,0.34161344,0.43584597,0.4336181,0.3990394,0.41189018],
[0.4477158,0.4282174,0.44174471,0.44892547,0.397319,0.40942556,0.37314835,0.3830873,0.3688252,0.37344623],
[0.29960895,0.1879597,0.27749795,0.18309179,0.2627379,0.17835598,0.22924939,0.29150575,0.24354453,0.24352019],
[0.44283354,0.38653716,0.46636248,0.40296263,0.46174476,0.42262968,0.4300904,0.41984725,0.4052861,0.40603065],
[0.43602538,0.37844956,0.4543122,0.40151766,0.440638,0.4194634,0.41263402,0.40843573,0.3798503,0.38261425],
[0.3523828,0.33852005,0.365209,0.33797908,0.4198187,0.36851838,0.42419392,0.38087884,0.41057158,0.38043642],
[0.3905255,0.3507308,0.4211171,0.3648373,0.44671357,0.39313924,0.4379331,0.4004067,0.4218245,0.39330426],
[0.27945763,0.35122207,0.26783836,0.3388614,0.2457186,0.2661285,0.32745367,0.25251752,0.3923667,0.33492213],
[0.29867512,0.29639262,0.3026459,0.2871729,0.32255778,0.28528538,0.33271044,0.2941931,0.33500785,0.30002102],
[0.34990105,0.33661473,0.36294925,0.34283566,0.3863907,0.3674279,0.38702387,0.37709042,0.373846,0.37211394],
[0.33603257,0.35234794,0.33715796,0.33766118,0.35166922,0.37043735,0.34015587,0.36150256,0.3329949,0.35325643],
[0.36641502,0.3888616,0.34685186,0.40443832,0.3457021,0.3858127,0.39408812,0.37695986,0.4099696,0.37832484],
[0.3301937,0.36530405,0.32233685,0.36468583,0.21372718,0.27917415,0.29375738,0.22302854,0.3351888,0.25531158],
[0.32825604,0.31752515,0.3103063,0.29380882,0.35152012,0.2913116,0.38017985,0.31967956,0.38393438,0.3558025],
[0.22485648,0.23028482,0.30948293,0.18413094,0.3597579,0.2642787,0.38171056,0.3368206,0.38585487,0.38305774],
[0.3760514,0.3474953,0.20983082,0.18751407,0.23057014,0.17504546,0.317678,0.21410558,0.37461033,0.30895665],
[0.3331477,0.29066068,0.34891257,0.3221115,0.36372364,0.3685914,0.34751287,0.35382733,0.34709206,0.35628435],
[0.38404283,0.38622615,0.19754577,0.3045459,0.20536007,0.29732382,0.3050506,0.17411235,0.37176877,0.2785245],
[0.3514415,0.4245259,0.34420195,0.44474927,0.29782176,0.5078538,0.27877712,0.53273284,0.26926205,0.502789],
[0.39191496,0.38655105,0.38722894,0.39124236,0.32659975,0.3636578,0.3919689,0.3389524,0.43470043,0.35326365]])


input_shape = X.shape[1]
output_shape = Y.shape[1]

patiance_num = 500 #argu of model.fit()
epc = 50000

nrd = 3

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

output_tensor = keras.layers.Dense(22)(x)   #output


model1 = keras.models.Model(input_tensor, output_tensor)
model1.compile(loss = keras.losses.MeanAbsoluteError(),
               optimizer = 'Adam', 
               metrics = ['mean_absolute_error'])
callback_list = [keras.callbacks.ReduceLROnPlateau(monitor="mean_absolute_error", factor=0.1, patience=patiance_num, min_lr=1e-6), 
                 keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=patiance_num, restore_best_weights=True)]
history = model1.fit(X_tr, Y_tr, epochs = epc, callbacks = callback_list, validation_data = (X_val, Y_val),verbose = 0)


#prep for accuracy
loss_train_list = []
loss_val_list = []
acc_list = []
acc_comsol = []
epc_list = []
loss_train = history.history['mean_absolute_error']        
loss_val = history.history['val_mean_absolute_error']

epc_list.append(len(loss_val))
epochs =  range (1, int(epc_list[-1]))


plt.plot(epochs, loss_train[1:], 'g', label='Training MSE')
plt.plot(epochs, loss_val[1:], 'b', label='Validation MSE')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Absolute Error (%)',fontsize=16)
plt.legend()
plt.savefig("L3.%.f.png" % (nrd), transparent=True, dpi=1000)

loss_train_list.append(loss_train)
loss_val_list.append(loss_val)
val_pred = model1.predict(X_val, verbose = 0)
acc_mean = []

for z in range(len(val_pred)):
    acc = 1 - np.sum(abs(val_pred[z, :] - Y_val[z, :])) / np.sum(Y_val[z, :])
    acc_mean.append(acc)
acc_list.append(np.mean(acc_mean))
print(acc_list)

Y_comsol_pred = model1.predict(X_com_val, verbose = 0)

np.savetxt('fdnnonly.L3.%.f.csv' % (nrd), Y_comsol_pred, delimiter=',')

print(epc_list)
print("fdnnonly.l3.%.f.h5" % (nrd))
model1.save("model.fdnnonly.l3.%.f.h5" % (nrd))