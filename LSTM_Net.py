from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import save_model, load_model

import sys


import numpy as np
import pandas as pd 
import os
import sys
import seaborn as sns
import funk


"""
print(experiments)
exp = path + experiments[69] + '/'
print(exp)
tables = sorted(os.listdir(path + experiments[0]))
acc = pd.read_csv(exp + tables[0], index_col=0)
gyro = pd.read_csv(exp + tables[1], index_col=0)
trajectory = pd.read_csv(exp + tables[2], index_col=0)
acc.columns = ['xa', 'ya','za','ta']
gyro.columns = ['xg', 'yg', 'zg', 'tg']
data = pd.concat([acc, gyro, trajectory], axis=1)
test_X=data[['xa','ya','za','xg', 'yg' ,'zg']]
test_y=data[['x','y']]
"""

path = './data/train/'
experiments = funk.sorts(os.listdir(path))



# Пытаюсь заменить на функцию из другой папки
test_X,test_y=funk.testdata(2)
print (test_X,test_y)
print(pd.__version__)
print(type(test_X))
test_X = test_X.to_numpy().reshape(test_X.shape[0], 1, test_X.shape[1])
print(test_X.shape)




# design network
model = Sequential()
model.add(LSTM(50, input_shape=(1, 6)))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss='mae', optimizer='adam')
model.save('my_model.h5')

for i in range (len(experiments)-10):
    exp = path + experiments[i] + '/'
    tables = sorted(os.listdir(path + experiments[0]))
    
    acc = pd.read_csv(exp + tables[0], index_col=0)
    gyro = pd.read_csv(exp + tables[1], index_col=0)
    trajectory = pd.read_csv(exp + tables[2], index_col=0)
    
    acc.columns = ['xa', 'ya','za','ta']
    gyro.columns = ['xg', 'yg', 'zg', 'tg']
    
    data = pd.concat([acc, gyro, trajectory], axis=1)
    del data['tmsp']
    
    train_X=data[['xa','ya','za','xg', 'yg' ,'zg']]
    
    # Нормализую столбцы ебать
    for i in range(6):
        train_X[::,i]=(train_X[::,i]-train_X[::,i].mean())/train_X[::,i].std()
    print (train_X)
    
    
    
    train_y=data[['x','y']]
    
    train_X=train_X.to_numpy().reshape(train_X.shape[0], 1, train_X.shape[1])
    
    train_X =train_X
    test_X =test_X


    model = load_model('my_model.h5')

    # fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=200, verbose=2, shuffle=False)
    model.save('my_model.h5')
    
    
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
    
    
    
    
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[1], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[ ]:




