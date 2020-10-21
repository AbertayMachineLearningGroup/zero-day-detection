#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:10:44 2019

@author: hananhindy
"""
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam, Nadam
from keras.regularizers import l2, l1, l1_l2
import keras.backend as K

class autoencoder:
    def __init__(self, num_features, verbose=True, mse_threshold = 0.5, archi="U15,D,U9,D,U6,D,U9,D,U15", reg = 'l2', l1_value = 0.1, l2_value = 0.001, dropout = 0.05, loss = 'mse'):
        self.model = Sequential()
        self.mse_threshold = mse_threshold
        
        input_ = Input((num_features,))
        
        regularisation = l2(l2_value)
        if reg == 'l1':
            regularisation = l1(l1_value)
        elif reg == 'l1l2':
            regularisation = l1_l2(l1 = l1_value, l2 = l2_value)
        
        layers = archi.split(',')
        
        input_ = Input((num_features,))
        previous = input_
        for l in layers:
            if l[0] == 'U':
                layer_value = int(l[1:])
                current = Dense(units = layer_value, use_bias=True, kernel_regularizer=regularisation, kernel_initializer = 'uniform')(previous)
                previous = current
            elif l[0] == 'D':
                current = Dropout(dropout)(previous)
                previous = current
        
        layer6 = Dense(units = num_features)(previous)
            
        self.model = Model(input_, layer6)
        if loss == 'mae':
            self.model.compile(loss=self.loss, optimizer = 'rmsprop', metrics=[self.accuracy])
        else:
            self.model.compile(loss='mean_squared_error', optimizer = 'adadelta', metrics=[self.accuracy])        
        
        
        if verbose:
            self.model.summary()   
            
    
    def accuracy(self, y_true, y_pred):
        mse = K.mean(K.square((y_true - y_pred)), axis = 1)
        temp = K.ones(K.shape(mse))
        return K.mean(K.equal(temp, K.cast(mse < self.mse_threshold, temp.dtype)))
    
    def loss(self, y_true,y_pred):
        return K.mean(K.abs(y_pred - y_true), axis=1)
    
