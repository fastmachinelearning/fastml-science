import numpy as np
import pandas as pd
import math
import os

from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Activation, Average, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import backend as K

current_directory=os.getcwd()

remap_8x8 = [4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
             24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
             59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]

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

def get_emd_loss(model_number):
    
    model_path=os.getcwd()+'/best_emd/'+str(model_number)+'.h5'
    emd_model = tf.keras.models.load_model(model_path)
    emd_model.trainable = False
                 
    def map_881_to_443(x):
        y = tf.reshape(x, (-1, 64))
        y = tf.gather(y, remap_8x8, axis=1)
        y = tf.gather(y, arrange443, axis=1)
        y = tf.reshape(y, (-1, 4, 4, 3))
        return y
  
    def emd_loss(y_true, y_pred):
        y_pred_443 = map_881_to_443(y_pred)
        y_true_443 = map_881_to_443(y_true)
        return emd_model([y_true_443, y_pred_443])
  
    return emd_loss
