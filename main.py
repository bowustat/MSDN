import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, LSTM,Conv1D,Dropout,Bidirectional,Multiply,Concatenate,Add,Flatten,Permute,Lambda,RepeatVector
from keras.models import Model
import tensorflow.python.keras.backend  as K
from thop import profile
#from attention_utils import get_activations
from keras.layers.core import *
from keras.layers import LSTM,GRU
from keras.models import *
tf.compat.v1.enable_eager_execution()
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn import preprocessing

city ="自贡"
data_wr = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\O3.xlsx")
data_city = data_wr[city]
data_O3 = data_city.values
data_O3=data_O3.reshape((len(data_O3),1))
data_wr_PM25 = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\PM25.xlsx")
data_city_PM25 = data_wr_PM25[city]
data_PM25 = data_city_PM25.values
data_PM25=data_PM25.reshape((len(data_PM25),1))
data_wr_NO2 = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\NO2.xlsx")
data_city_NO2 = data_wr_NO2[city]
data_NO2 = data_city_NO2.values
data_NO2=data_NO2.reshape((len(data_NO2),1))
data_wr_CO = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\CO.xlsx")
data_city_CO = data_wr_CO[city]
data_CO = data_city_CO.values
data_CO=data_CO.reshape((len(data_CO),1))
data_wr_SO2 = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\SO2.xlsx")
data_city_SO2 = data_wr_SO2[city]
data_SO2 = data_city_SO2.values
data_SO2=data_SO2.reshape((len(data_SO2),1))
data_wr_PM10 = pd.read_excel(r"D:\python_student\合作文章\HEZUO\DATA\整理污染数据\2016-2024\PM10.xlsx")
data_city_PM10 = data_wr_PM10[city]
data_PM10 = data_city_PM10.values
data_PM10 =data_PM10.reshape((len(data_PM10),1))

lj = "D:\python_student\合作文章\HEZUO\DATA\整理气象数据/2016-2024/"+city+"气象数据.xlsx"
data_qx = pd.read_excel(lj)
data_qx = data_qx.values
data_qx = data_qx[:,2:]

window_size = 4
wid = window_size
data_qx_x = data_qx
y_his = data_PM25
Y = y_his[wid:, :]

data_x1 = np.concatenate((data_qx_x,y_his,data_O3,data_NO2,data_CO,data_SO2,data_PM10),axis=1)
data_x1 = preprocessing.minmax_scale(data_x1)

re_data_x =[]
for i in range(0,len(y_his)-wid):
    wid_data = data_x1[i:i+wid,:]
    re_data_x.append(wid_data)
re_data_x = np.array(re_data_x)

data_x = re_data_x

# input_max = max(X)
# input_min = min(X)
# data_x  = X

output_max = np.max(Y)
output_min = np.min(Y)
data_y = preprocessing.minmax_scale(Y)  # 标准化
# data_y = Y
# 数据集分割
data_len = len(data_x)
t = np.linspace(0, data_len, data_len)

train_data_ratio = 0.8  # Choose 80% of the data for training
train_data_len = int(data_len * train_data_ratio)

train_x = data_x[0:train_data_len]
train_y = data_y[0:train_data_len]
t_for_training = t[0:train_data_len]

test_x = data_x[train_data_len:]
test_y = data_y[train_data_len:]
t_for_testing = t[train_data_len:]

INPUT_FEATURES_NUM = np.shape(train_x)[2]
OUTPUT_FEATURES_NUM = 1

# 改变输入形状
train_x_tensor = train_x.reshape(-1, INPUT_FEATURES_NUM, window_size)  # set batch size to 1
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1
test_x_tensor = test_x.reshape(-1, INPUT_FEATURES_NUM,window_size)

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

fea_num = INPUT_FEATURES_NUM
def attention_model():
    inputs = Input((fea_num,window_size))
    x1 = Conv1D(filters = 64, kernel_size = 2, strides=1, padding="same", activation = 'relu')(inputs)  #, padding = 'same'
    x12 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x1)  # , padding = 'same'
    GRU_out1 = GRU(20, return_sequences=True)(x12)
    x2 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x1)  # , padding = 'same'
    x22 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x2)  # , padding = 'same'
    GRU_out2 = GRU(20, return_sequences=True)(x22)
    x3 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x2)  # , padding = 'same'
    x32 = Conv1D(filters=64, kernel_size=2, strides=1, padding="same", activation='relu')(x3)  # , padding = 'same'
    GRU_out3 = GRU(20, return_sequences=True)(x32)
    x = Concatenate(axis=1)([GRU_out1, GRU_out2, GRU_out3])
    attention_mul = attention_3d_block2(x)
    attention_mul = Flatten()(attention_mul)
    output_Dense = Dense(100, activation="relu")(attention_mul)
    drop = Dropout(0.1)(output_Dense)
    output_Dense = Dense(20, activation="relu")(drop)
    drop2 = Dropout(0.1)(output_Dense)
    dense_3 = Dense(50, activation="relu")(drop2)
    output1 = Dense(name='output1',units=1)(dense_3)
    model = Model(inputs=inputs, outputs=output1)
    return model

m = attention_model()
m.compile(loss="mse", optimizer='adam')
m.summary()



m.fit(x=train_x_tensor, y=train_y_tensor, epochs=10,batch_size= 80 )


pred_y_for_test = m.predict(test_x_tensor)
train_y_for_test = m.predict(train_x_tensor)


pred_y_for_test = (output_max - output_min) * np.array(pred_y_for_test) + output_min
train_y_for_test = (output_max - output_min) * np.array(train_y_for_test) + output_min
train_y = (output_max - output_min) * np.array(train_y) + output_min
test_y = (output_max - output_min) * np.array(test_y) + output_min
pred_y_for_test=pd.DataFrame(pred_y_for_test)

MSE = mean_squared_error(test_y, pred_y_for_test)
MAE = mean_absolute_error(test_y, pred_y_for_test)
R2 = r2_score(test_y, pred_y_for_test)
RMSE = np.mean((test_y - pred_y_for_test.values) ** 2) ** 0.5
print("MSE1:",MSE, "MAE:",MAE,"R2",R2,"RMSE:",RMSE)

plt.figure(1)
t = t_for_testing
plt.plot(t,test_y,"r.", markersize=1)
plt.plot(t,pred_y_for_test, markersize=1)
# 散点图的标题
plt.title("24 Step")
# 设置坐标轴的标签
plt.xlabel("time")
plt.ylabel("PM2.5")
#plt.plot(t, ub, lw=1)
#plt.plot(t, lb, lw=1)
plt.show()
# test_y = pd.DataFrame(test_y)
# test_y.to_csv('LSTM_test_y.csv')  # 数据存入csv,存储位置及文件名称
# pred_y_for_test_data = pd.DataFrame(pred_y_for_test)  # 将验证集数据的预测值放进表格
# pred_y_for_test_data.to_csv('LSTM_pred_y_for_test.csv')  # 数据存入csv,存储位置及文件名称
