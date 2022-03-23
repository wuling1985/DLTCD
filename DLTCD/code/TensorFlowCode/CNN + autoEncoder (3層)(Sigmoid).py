#!/usr/bin/env python
# coding: utf-8

# In[1]:


#引用tensorflow函式庫
import tensorflow
#引用keras函式庫
import keras
#引用csv函式庫
import csv
#引用numpy函式庫
import numpy
#引用pyplot函式庫
import matplotlib.pyplot as plot

from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

import datetime
start = datetime.datetime.now().time()

#輸入變數數量
num_inputs = 62

#讀取訓練資料
dataset = numpy.loadtxt("Data.csv", delimiter=",")
X = dataset[:,0:num_inputs]
Y = X
X = X.reshape(X.shape[0], num_inputs, 1, 1).astype('float32')
#dataset_y = dataset[:,num_inputs]
#Y = keras.utils.to_categorical(dataset_y, num_classes)


# In[2]:


#設定亂數種子
numpy.random.seed(0)

# this is the size of our encoded representations
encoding_dim = 2  # 2 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
inputs = Input(shape=(num_inputs, 1, 1))
# "encoded" is the encoded representation of the input
encoded1 = Conv2D(filters = 10, kernel_size = (10, 1), activation='sigmoid')(inputs)
encoded2 = Conv2D(filters = 10, kernel_size = (6, 1), activation='sigmoid')(encoded1)
encoded3 = Flatten()(encoded2)
decoded3 = Dense(2, activation='sigmoid')(encoded3)
# "decoded" is the lossy reconstruction of the input
decoded2 = Dense(5, activation='sigmoid')(decoded3)
decoded1 = Dense(10, activation='sigmoid')(decoded2)
decoded = Dense(num_inputs, activation='sigmoid')(decoded1)


# In[3]:


# this model maps an input to its reconstruction
autoencoder = Model(inputs, decoded)

# this model maps an input to its encoded representation
encoder = Model(inputs, decoded3)

# 
autoencoder.compile(loss='mean_squared_error', optimizer = 'sgd', metrics = ['mae'])

#訓練神經網路
train_history = autoencoder.fit(X, Y, epochs = 20000, batch_size = 3) #訓練回合數: 20000, 每3筆修正權重

def show_train_history(train_history, x1, x2):
    plot.plot(train_history.history[x1])
    plot.plot(train_history.history[x2])
    plot.title('Train History')
    plot.ylabel('train')
    plot.xlabel('Epoch')
    plot.legend([x1, x2], loc = 'upper right')
    plot.show()

#顯示訓練過程
show_train_history(train_history, 'loss', 'mean_absolute_error')

#儲存模型
autoencoder.save('CNN+AE(sigmoid).h5')
autoencoder.save_weights("CNN+AE(sigmoid)_weights.h5")


# In[4]:


#讀取訓練資料
dataset = numpy.loadtxt("Data.csv", delimiter=",")
X = dataset[:,0:num_inputs] 
X = X.reshape(X.shape[0], num_inputs, 1, 1).astype('float32')
predictions = encoder.predict(X)
print(predictions)
#將測試結果寫入predictions.csv
numpy.savetxt("Result(CNN+AE+sigmoid).csv", predictions, delimiter=",")

print(start)
print(datetime.datetime.now().time())

