import numpy as np
import pandas as pd
from pickle import dump
#from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model, load_model
#from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout, Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2

from sklearn.model_selection import KFold
import os
import shutil
from keras.callbacks import TensorBoard

bs = 1000
wdir = 'C:\\Users\\Asus\\Desktop\\EXP\\EXP\\Training Algorithms\\freshly_trained_nets\\'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return(res)

def make_resnet(num_blocks=1, num_filters=64, num_outputs=1, d1=512, d2=512, word_size=64, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,))
  rs = Reshape((2 * num_blocks, word_size))(inp)
  perm = Permute((2,1))(rs)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  conv0 = BatchNormalization()(conv0)
  conv0 = Activation('relu')(conv0)
  #add residual blocks
  shortcut = conv0
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    shortcut = Add()([shortcut, conv2])
    
    
  #add prediction head
  flat1 = Flatten()(shortcut)
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)#output array of shape d1and regularizer
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)#creates model using the specified inputs and output layer
  
  return(model)

def train_xtea_distinguisher(num_epochs, num_rounds=7, depth=1):
  #train and evaluate
  X = pd.read_csv("C:\\Users\\Asus\\Desktop\\EXP\\EXP\\xtea_R7.csv",header=None)
  Y = pd.read_csv("C:\\Users\\Asus\\Desktop\\EXP\\EXP\\label_2M.csv",header=None)
  X = np.array(X).reshape(X.shape[0],X.shape[1])          #      10^5 X 128
  Y = np.array(Y).reshape(Y.shape[0], Y.shape[1])         #      10^5 X 1
  print("X Shape : ",X.shape)
  print("Y Shape : ",Y.shape)

  kf = KFold(n_splits=5)
  
  best_val_accuracy=0
  i=1
  for train_index, test_index in kf.split(X):
    print("Fitting the model on Fold : ",i)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    #create learnrate schedule

    #lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))
    lr=0.0001
    #set up model checkpoint
    check = make_checkpoint(wdir+'KFoldTempModel.h5')
    log_dir = 'logs/R7/fold_{}'.format(i)
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    h = net.fit(X_train, Y_train, epochs=num_epochs, batch_size=bs,callbacks=[tensorboard_callback,check] , verbose=1, validation_data=(X_test, Y_test))

    print("======="*12, end="\n\n\n")
    kfold_val_accuracy = np.max(h.history['val_acc'])
    if best_val_accuracy<kfold_val_accuracy:
      shutil.copyfile(wdir+'KFoldTempModel.h5', wdir+'BestModel.h5')
      best_val_accuracy=kfold_val_accuracy
    
    del net
    i+=1
          
  del X,Y
  print("Best validation accuracy: ", best_val_accuracy)
  
  model = load_model(wdir+'BestModel.h5')
  model_json = model.to_json()
  with open("R7_conv1d.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("R7_conv1d.h5")
  print("Saved model to disk")
  return(model, h)
    
train_xtea_distinguisher(30,7,3)
