"""
Auxilliary  functions to calculate the telescope metric
C. Herwig

Masks correspond to groupings of 2x2 cells
They are weighted to account for edge TCs that enter fewer 2x2 supercells than those in the center of the sensor.

the idea is that we want to associate shape information to each TC, so each TC should contribute an equal amount to the information encoded in the loss associated to the collection of 2x2 super cells.
to derive the weights, we first ask how many 2x2 cells each TC enters and get values like

1 2 2 2 | 2 2 2 1
2 4 4 4 | 4 4 4 2
2 4 4 4 | 4 4 4 2
2 4 4 3 | 3 4 4 2
-------  --------
2 4 4 3 | #     ^
2 4 4 4 | #     |
2 4 4 4 | # <-- these edges
1 2 2 2 | #     are associated 

e.g. the very top left TC only enters one 2x2 tower (as top left TC)
while the one next to it can be the top left or top right TC of a 2x2 supercell

the 2x2 SC weights are derived to ensure that each TC contributes equally regardless of how many supercells it enters (she contributed shape info will just be less). This ensures that there are no charge-dependent biases.

the weights for a SC is the sum of the inverses of the # of times each constituent TC enters a TC
i.e. the weight for a SC combined from the upper left 2x2 is
W = 1/1 + 1/2 + 1/2 + 1/4 = 2.25
while for a SC shifted one TC to the right the weight is 2*1/4+2*1/2=1.5
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

# combine neighbor cells in 2x2 grids, record weights
# multilpy weights by 0.25 for now to account for effective increase in cells from 12 (sum weights now 48 not 12)
SCmask_48_36 = np.array([
    [ 0,  1,  4,  5, 0.25*1.5], # 2x2 supercells that perfectly tile the sensor
    [ 2,  3,  6,  7, 0.25*1.+1./12], #4 TC indices for 1 supercell (+) weight
    [ 8,  9, 12, 13, 0.25*2.25], 
    [10, 11, 14, 15, 0.25*1.5], 
    [16, 17, 20, 21, 0.25*1.5], 
    [18, 19, 22, 23, 0.25*1.+1./12], 
    [24, 25, 28, 29, 0.25*2.25], 
    [26, 27, 30, 31, 0.25*1.5], 
    [32, 33, 36, 37, 0.25*1.5], 
    [34, 35, 38, 39, 0.25*1.+1./12], 
    [40, 41, 44, 45, 0.25*2.25], 
    [42, 43, 46, 47, 0.25*1.5], 
    [ 4,  5,  8,  9, 0.25*1.5], # shift right by one TC (2/2x2)
    [ 6,  7, 10, 11, 0.25*1.],
    [20, 21, 24, 25, 0.25*1.5],
    [22, 23, 26, 27, 0.25*1.],
    [36, 37, 40, 41, 0.25*1.5],
    [38, 39, 42, 43, 0.25*1.],
    [ 1,  2,  5,  6, 0.25*1.], # shift down by one TC (2/2x2)
    [ 9, 10, 13, 14, 0.25*1.5],
    [17, 18, 21, 22, 0.25*1.],
    [25, 26, 29, 30, 0.25*1.5],
    [33, 34, 37, 38, 0.25*1.],
    [41, 42, 45, 46, 0.25*1.5],
    [ 5,  6,  9, 10, 0.25*1.], # shift down and right by one TC (1/2x2)
    [21, 22, 25, 26, 0.25*1.],
    [37, 38, 41, 42, 0.25*1.],
    [ 0,  1, 27, 31, 0.25*1.5], # inter-2x2 overlaps
    [ 1,  2, 23, 27, 0.25*1.],
    [ 2,  3, 19, 23, 0.25*1.+1./6],
    [ 3,  7, 34, 35, 0.25*1.+1./6],
    [ 7, 11, 33, 34, 0.25*1.],
    [11, 15, 32, 33, 0.25*1.5],
    [16, 17, 47, 43, 0.25*1.5],
    [17, 18, 43, 39, 0.25*1.],
    [18, 19, 39, 35, 0.25*1.+1./6],
])
Remap_48_36 = np.zeros((48,36))
for isc,sc in enumerate(SCmask_48_36): 
    for tc in sc[:4]:
        Remap_48_36[int(tc),isc]=1
tf_Remap_48_36 = tf.constant(Remap_48_36,dtype=tf.float32)
Weights_48_36 = SCmask_48_36[:,4]
tf_Weights_48_36 = tf.constant(Weights_48_36,dtype=tf.float32)

#
# keep simplified 12 x 3 mapping for now
SCmask_48_12 = np.array([
    [ 0,  1,  4,  5],
    [ 2,  3,  6,  7],
    [ 8,  9, 12, 13],
    [10, 11, 14, 15],
    [16, 17, 20, 21],
    [18, 19, 22, 23],
    [24, 25, 28, 29],
    [26, 27, 30, 31],
    [32, 33, 36, 37],
    [34, 35, 38, 39],
    [40, 41, 44, 45],
    [42, 43, 46, 47],
])
Remap_48_12 = np.zeros((48,12))
for isc,sc in enumerate(SCmask_48_12): 
    for tc in sc:
        Remap_48_12[int(tc),isc]=1
tf_Remap_48_12 = tf.constant(Remap_48_12,dtype=tf.float32)
Remap_12_3 = np.zeros((12,3))
for i in range(12): Remap_12_3[i,int(i/4)]=1
tf_Remap_12_3 = tf.constant(Remap_12_3,dtype=tf.float32)

def telescopeMSE2(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    # TC-level MSE
    y_pred_rs = K.reshape(y_pred, (-1,48))
    y_true_rs = K.reshape(y_true, (-1,48))
    # lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs), axis=(-1))
    lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs) * K.maximum(y_pred_rs, y_true_rs), axis=(-1))

    # map TCs to 2x2 supercells and compute MSE
    y_pred_36 = tf.matmul(y_pred_rs, tf_Remap_48_36)
    y_true_36 = tf.matmul(y_true_rs, tf_Remap_48_36)
    # lossTC2 = K.mean(K.square(y_true_12 - y_pred_12), axis=(-1))
    lossTC2 = K.mean(K.square(y_true_36 - y_pred_36) * K.maximum(y_pred_36, y_true_36) * tf_Weights_48_36, axis=(-1))
  
    # map 2x2 supercells to 4x4 supercells and compute MSE
    y_pred_12 = tf.matmul(y_pred_rs, tf_Remap_48_12)
    y_true_12 = tf.matmul(y_true_rs, tf_Remap_48_12)
    y_pred_3 = tf.matmul(y_pred_12, tf_Remap_12_3)
    y_true_3 = tf.matmul(y_true_12, tf_Remap_12_3)
    # lossTC3 = K.mean(K.square(y_true_3 - y_pred_3), axis=(-1))
    lossTC3 = K.mean(K.square(y_true_3 - y_pred_3) * K.maximum(y_pred_3, y_true_3), axis=(-1))

    # sum MSEs
    #return lossTC1 + lossTC2 + lossTC3
    return 4*lossTC1 + 2*lossTC2 + lossTC3



remap_443 = np.array([ 0,  3, 6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,  
                       1,  4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,  
                       2,  5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47])
remap_443_matrix = np.zeros(48*48,dtype=np.float32).reshape((48,48))
for i in range(48): 
    remap_443_matrix[remap_443[i],i] = 1

def telescopeMSE443(y_true,y_pred):
    return telescopeMSE2(tf.matmul(K.reshape(y_true,(-1,48)),remap_443_matrix),
                         tf.matmul(K.reshape(y_pred,(-1,48)),remap_443_matrix))

remap_663 = np.array([ 25, 26, 27,  28, 19, 20, 21, 22, 13, 14, 15, 16,  7,  8,  9, 10, 
                       61, 62, 63,  64, 55, 56, 57, 58, 49, 50, 51, 52, 43, 44, 45, 46, 
                       97, 98, 99, 100, 91, 92, 93, 94, 85, 86, 87, 88, 79, 80, 81, 82])
remap_663_matrix = np.zeros(48*108,dtype=np.float32).reshape((108,48))
for i in range(48): 
    remap_663_matrix[remap_663[i],i] = 1

def telescopeMSE663(y_true,y_pred):
    return telescopeMSE2(tf.matmul(K.reshape(y_true,(-1,108)),remap_663_matrix),
                         tf.matmul(K.reshape(y_pred,(-1,108)),remap_663_matrix))

remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]
remap_8x8_matrix = np.zeros(48*64,dtype=np.float32).reshape((64,48))

for i in range(48): 
    remap_8x8_matrix[remap_8x8[i],i] = 1

def telescopeMSE8x8(y_true,y_pred):
    return telescopeMSE2(tf.matmul(K.reshape(y_true,(-1,64)),remap_8x8_matrix),
                         tf.matmul(K.reshape(y_pred,(-1,64)),remap_8x8_matrix))

