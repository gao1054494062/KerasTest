from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32)


model1=Sequential()
model1.add(Dense(32,activation='relu',input_dim=100))
model1.add(Dense(10,activation='softmax'))
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

data1=np.random.random((1000,100))
labels1=np.random.randint(10,size=(1000,1))
one_hot_lables=np_utils.to_categorical(labels1,num_classes=10)
model1.fit(data1,one_hot_lables,epochs=10,batch_size=32)