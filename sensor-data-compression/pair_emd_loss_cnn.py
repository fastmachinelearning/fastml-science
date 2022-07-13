"""
Created on Thu Apr 15 08:02:19 2021

@author: Javier Duarte, Rohan Shenoy, UCSD
"""

import numpy as np
import pandas as pd
import math

import itertools

import os
import sys
sys.path.insert(0, "../")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.wafer import plot_wafer as plotWafer

from utils.metrics import emd

from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Activation, Average, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from qkeras import *

class pair_EMD_CNN:
    
    X1_train=[]
    X2_train=[]
    
    def ittrain(self,calQ_data,num_filt, kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d, num_epochs,Loss):
        
        current_directory='/ecoderemdvol/q_pair'
        
        #Arranging the hexagon
        arrange443 = np.array([0,16, 32,
                               1,17, 33,
                               2,18, 34,
                               3,19, 35,
                               4,20, 36,
                               5,21, 37,
                               6,22, 38,
                               7,23, 39,
                               8,24, 40,
                               9,25, 41,
                               10,26, 42,
                               11,27, 43,
                               12,28, 44,
                               13,29, 45,
                               14,30, 46,
                               15,31, 47])
        
        calQ     = calQ_data
        sumQ     = calQ.sum(axis=1)
        calQ     = calQ[sumQ>0]
        sumQ     = sumQ[sumQ>0]
        occ = (np.count_nonzero(calQ>1,axis=1))
    
        # reshape into 443 and normalize (as is usually done for autoencoder)
        calQ_443 = (calQ/np.expand_dims(sumQ,-1))[:,arrange443].reshape(-1,4,4,3)
    
        # split train and validation so there is no overlap in samples whatsoever
        train_indices = range(0, int(0.6*len(calQ)))
        val_indices = range(int(0.6*len(calQ)), len(calQ))
        
        idx1_train = np.array([i for i,j in itertools.product(train_indices,train_indices)])
        idx2_train = np.array([j for i,j in itertools.product(train_indices,train_indices)])
              
        X = calQ_443
                 
        X1_train = X[idx1_train]
        X2_train = X[idx2_train]
        y_train = np.array([emd(calQ[i],calQ[j]) for i, j in zip(idx1_train, idx2_train)])
    
        idx1_val = np.array([i for i,j in itertools.product(val_indices,val_indices)])
        idx2_val = np.array([j for i,j in itertools.product(val_indices,val_indices)])
    
        X1_val = X[idx1_val]
        X2_val = X[idx2_val]
        y_val = np.array([emd(calQ[i],calQ[j]) for i, j in zip(idx1_val, idx2_val)])
    
        print(X1_train.shape)
        print(X2_train.shape)
        print(y_train.shape)
    
        print(X1_val.shape)
        print(X2_val.shape)
        print(y_val.shape)        
    
        #Building CNN
        
        # make a convolutional model as a more advanced PoC
        input1 = Input(shape=(4, 4, 3,), name='input_1')
        input2 = Input(shape=(4, 4, 3,), name='input_2')
        x = Concatenate(name='concat')([input1, input2])
            
        #Number of Conv2D Layers
        for i in range(1,num_conv_2d+1):
            ind=str(i)
            x = QConv2D(num_filt, kernel_size, strides=(1, 1), kernel_quantizer="stochastic_ternary",bias_quantizer="ternary",name='conv2d_'+ind, padding='same', kernel_regularizer=l1_l2(l1=0,l2=1e-4))(x)
            x = QBatchNormalization(name='batchnorm_'+ind)(x)
            x = QActivation('relu', name='relu_'+ind)(x)
                
        x = Flatten(name='flatten')(x)
            
        #Number of Dense Layers
        for i in range(1,num_dens_layers+1):
            ind=str(i)
            jind=str(i+num_conv_2d)
            x = QDense(units=num_dens_neurons, kernel_quantizer=quantized_bits(3),bias_quantizer=quantized_bits(3),kernel_regularizer=l1_l2(l1=0,l2=1e-4), name='dense_'+ind)(x)
            x = QBatchNormalization(name='batchnorm'+jind)(x)
            x = QActivation('relu', name='relu_'+jind)(x)
        
        x = QActivation("quantized_bits(20, 5)")(x)
                
        output = QDense(1, name='output')(x)
        model = Model(inputs=[input1, input2], outputs=output, name='base_model')
        model.summary()
                
        # make a model that enforces the symmetry of the EMD function by averging the outputs for swapped inputs
        output = Average(name='average')([model((input1, input2)), model((input2, input1))])
        sym_model = Model(inputs=[input1, input2], outputs=output, name='sym_model')
        sym_model.summary()
        
        final_directory=os.path.join(current_directory,r'q_pair_emd_models')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        callbacks = [ModelCheckpoint('/ecoderemdvol/q_pair/q_pair_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'best.h5', monitor='val_loss', verbose=1, save_best_only=True),
                     ModelCheckpoint('/ecoderemdvol/q_pair/q_pair_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'last.h5', monitor='val_loss', verbose=1, save_last_only=True),
                    ]
            
        sym_model.compile(optimizer='adam', loss=Loss, metrics=['mse', 'mae', 'mape', 'msle'])
        history = sym_model.fit((X1_train, X2_train), y_train, 
                            validation_data=((X1_val, X2_val), y_val),
                            epochs=num_epochs, verbose=1, batch_size=32, callbacks=callbacks)
        
        #Making directory for graphs
        
        img_directory=os.path.join(current_directory,r'Q Pair EMD Plots')
        if not os.path.exists(img_directory):
            os.makedirs(img_directory)
        
        #Plot Validation loss and training loss
        
        plt.close()
        fig=plt.plot(history.history['loss'], label='Train')
        fig=plt.plot(history.history['val_loss'], label='Val.')
        fig=plt.xlabel('Epoch')
        fig=plt.ylabel(Loss+''+'loss')
        fig=plt.legend()
        plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Loss.png")
        plt.close()
        
        #Plots True EMD and Pred Emd Histogram
        
        plt.close()
        y_val_preds = sym_model.predict((X1_val, X2_val))
        fig=plt.figure()
        fig=plt.hist(y_val, alpha=0.5, bins=np.arange(0, 7.5, 0.01), label='TrueEMD')
        fig=plt.hist(y_val_preds, alpha=0.5, bins=np.arange(0, 7.5, 0.01), label='EMDCNN')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Hist.png")
        plt.close()
        
        #Plot Relative Difference
        
        plt.close()
        rel_diff = (y_val_preds[y_val>0].flatten()-y_val[y_val>0].flatten())/y_val[y_val>0].flatten()
        fig=plt.figure()
        fig=plt.hist(rel_diff, bins=np.arange(-1, 1, 0.01), color='green', label = 'mean = {:.3f}, std. = {:.3f}'.format(np.mean(rel_diff), np.std(rel_diff)))
        fig=plt.xlabel('EMD rel. diff.')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"RelD.png")
        plt.close()
        
        #Plot True EMD vs Pred Emd Graphic
        
        plt.close()
        fig, ax = plt.subplots(figsize =(5, 5)) 
        x_bins = np.arange(0, 7.5, 0.01)
        y_bins = np.arange(0, 7.5, 0.01)
        plt.hist2d(y_val.flatten(), y_val_preds.flatten(), bins=[x_bins,y_bins])
        plt.plot([0, 15], [0, 15], color='gray', alpha=0.5)
        ax.set_xlabel('True EMD [GeV]')
        ax.set_ylabel('Pred. EMD [GeV]')
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Graphic.png")
        plt.close()
        
        return(np.mean(rel_diff),np.std(rel_diff))
    
