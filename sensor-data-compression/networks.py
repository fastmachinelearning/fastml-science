import numpy as np
import tensorflow.keras.optimizers as opt

from telescope import telescopeMSE443,telescopeMSE663,telescopeMSE8x8

from emdloss import get_emd_loss

edim = 16

"""
8x8 note: for newer versions of CALQ.csv files where the arrangement is fixed for the AE block
we can swith to something like:
np.array([0,1,2,3,4,5,6,7,
         8,9,10,11,12,13,14,15,
         16,17,18,19,20,21,22,23,
         24,25,26,27,28,29,30,31,
         32,33,34,35,32,33,34,35,
         36,37,38,39,36,37,38,39,
         40,41,42,43,40,41,42,43,
         44,45,46,47,44,45,46,47])
"""
arrange_dict = {
    # 4x4x3 geometry
    # 3 4x4 images
    '4x4x3': {'arrange': np.array([0,16,32,
                                  1,17,33,
                                  2,18,34,
                                  3,19,35,
                                  4,20,36,
                                  5,21,37,
                                  6,22,38,
                                  7,23,39,
                                  8,24,40,
                                  9,25,41,
                                  10,26,42,
                                  11,27,43,
                                  12,28,44,
                                  13,29,45,
                                  14,30,46,
                                  15,31,47]),
              'arrMask': [],
              'calQMask': [],
              },
    # 8x8x4 geometry
    # 1 8x8 image
    '8x8': {'arrange': np.array([28,29,30,31,0,4,8,12,
                                 24,25,26,27,1,5,9,13,
                                 20,21,22,23,2,6,10,14,
                                 16,17,18,19,3,7,11,15,
                                 47,43,39,35,35,34,33,32,
                                 46,42,38,34,39,38,37,36,
                                 45,41,37,33,43,42,41,40,
                                 44,40,36,32,47,46,45,44]),
            'arrMask': np.array([1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,
                                 1,1,1,1,1,1,1,1,
                                 1,1,1,1,0,0,0,0,
                                 1,1,1,1,0,0,0,0,
                                 1,1,1,1,0,0,0,0,
                                 1,1,1,1,0,0,0,0,]),
            'calQMask': np.array([1,1,1,1,1,1,1,1,
                                  1,1,1,1,1,1,1,1,
                                  1,1,1,1,1,1,1,1,
                                  1,1,1,1,1,1,1,1,
                                  1,1,1,1,0,0,0,0,
                                  1,1,1,1,0,0,0,0,
                                  1,1,1,1,0,0,0,0,
                                  1,1,1,1,0,0,0,0,])
            },
    # 6x6x3 geometry
    # 3 6x6 images with edge padding in each
    '6x6x3': {'arrange': np.array([  0,0,0,0,0,0,
                                     0,12,13,14,15,32,
                                     0,8,9,10,11,33,
                                     0,4,5,6,7,34,
                                     0,0,1,2,3,35,
                                     0,31,27,23,19,0,
                                     0,0,0,0,0,0,
                                     0,28,29,30,31,0,
                                     0,24,25,26,27,1,
                                     0,20,21,22,23,2,
                                     0,16,17,18,19,3,
                                     0,47,43,39,35,0,
                                     0,0,0,0,0,0,
                                     0,44,45,46,47,16,
                                     0,40,41,42,43,17,
                                     0,36,37,38,39,18,
                                     0,32,33,34,35,19,
                                     0,15,11,7,3,0]),
              'arrMask': np.array([  0,0,0,0,0,0,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,0,
                                     0,0,0,0,0,0,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,0,
                                     0,0,0,0,0,0,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,1,
                                     0,1,1,1,1,0]),
              'calQMask': np.array([  0,0,0,0,0,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,0,0,0,0,0,
                                      0,0,0,0,0,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,0,0,0,0,0,
                                      0,0,0,0,0,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,1,1,1,1,0,
                                      0,0,0,0,0,0])
              },
    }
arrange_dict['6x6x3_tp'] = {'arrange': arrange_dict['6x6x3']['arrange'].reshape(3,36).transpose().flatten(),
                            'arrMask': arrange_dict['6x6x3']['arrMask'].reshape(3,36).transpose().flatten(),
                            'calQMask': arrange_dict['6x6x3']['calQMask'].reshape(3,36).transpose().flatten(),
                            }

networks_by_name = [
    
    {'name':'8x8_c8_S2_qK',
     'label':'8x8_c[8]_S2',
     'arr_key':'8x8',
     'isQK':True,
     'params':{
         'shape':(8,8,1),
         'loss':'weightedMSE',
         'CNN_layer_nodes':[8],
         'CNN_kernel_size':[3],
         'CNN_strides':[(2,2)],
         'nBits_input': {'total': 10, 'integer': 3, 'keep_negative':1},
         'nBits_encod': {'total':  9, 'integer': 1,'keep_negative':0}, # 0 to 2 range, 8 bit decimal 
         'nBits_accum': {'total': 11, 'integer': 3, 'keep_negative':1},
         'nBits_weight': {'total':  5, 'integer': 1, 'keep_negative':1}, # sign bit not included
        },
    },
    
    {'name':'8x8_c8_S2_tele',
     'label':'8x8_c[8]_S2(tele)',
     'arr_key':'8x8',
     'params':{
         'shape':(8,8,1),
         'loss':telescopeMSE8x8,
         'CNN_layer_nodes':[8],
         'CNN_kernel_size':[3],
         'CNN_strides':[(2,2)],
        },
    },
]

defaults = {'channels_first': False,
            'encoded_dim': 16,
            }
                
for m in networks_by_name:
    arrange = arrange_dict[m['arr_key']]
    m['params'].update({
        'arrange': arrange['arrange'],
        'arrMask': arrange['arrMask'],
        'calQMask': arrange['calQMask'],
    })
    
    if not 'isDense2D' in m.keys(): m.update({'isDense2D':False})
    if not 'isQK' in m.keys(): m.update({'isQK':False})
    if not 'ws' in m.keys(): m.update({'ws':''})
    for p,v in defaults.items():
        if not p in m['params'].keys():
            m['params'].update({p:v})
