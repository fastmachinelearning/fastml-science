#!/usr/bin/env python

from tensorflow.keras.models import model_from_json  
from argparse import ArgumentParser
from keras import backend as K
import os
import tensorflow as tf
import json
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def save_graph(tfsession,pred_node_names,tfoutpath,graphname):
    saver = tfv1.train.Saver()
    
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io

    constant_graph = graph_util.convert_variables_to_constants(
        tfsession, tfsession.graph.as_graph_def(), pred_node_names)
    #constant_graph = tfsession.graph.as_graph_def()

    f = graphname+'_constantgraph.pb.ascii'
    tfv1.train.write_graph(constant_graph, tfoutpath, f, as_text=True)
    print('saved the graph definition in ascii format at: ', os.path.join(tfoutpath, f))

    f = graphname+'_constantgraph.pb'
    tfv1.train.write_graph(constant_graph, tfoutpath, f, as_text=False)
    print('saved the graph definition in pb format at: ', os.path.join(tfoutpath, f))


    #graph_io.write_graph(constant_graph, args.outputDir, output_graph_name, as_text=False)
    #print('saved the constant graph (ready for inference) at: ', os.path.join(args.outputDir, output_graph_name))

    saver.save(tfsession, tfoutpath)

tfback._get_available_gpus = _get_available_gpus

## use tfv1 for conversion
if tf.__version__.startswith("2."):
    tfv1 = tf.compat.v1
tfv1.disable_eager_execution()

parser = ArgumentParser('')
parser.add_argument('-i','--inputModel',dest='inputModel',default='./models/no-weight.json')
parser.add_argument('-o','--outputDir',dest='outputDir',default='./')
parser.add_argument('--outputLayer',dest='outputLayer',default='encoded_vector/Relu')
parser.add_argument('--outputGraph',dest='outputGraph',default='encoder')

args = parser.parse_args()

print(args.outputDir)

f_model = args.inputModel
with open(f_model,'r') as f:
    if 'QActivation' in f.read():
        from qkeras import QDense, QConv2D, QActivation,quantized_bits,Clip
        f.seek(0)
        model = model_from_json(f.read(),
                                custom_objects={'QActivation':QActivation,
                                                'quantized_bits':quantized_bits,
                                                'QConv2D':QConv2D,
                                                'QDense':QDense,
                                                'Clip':Clip})
        hdf5  = f_model.replace('json','hdf5')
        model.load_weights(hdf5)
    else:
        f.seek(0)
        model = model_from_json(f.read())
        hdf5  = f_model.replace('json','hdf5')
        model.load_weights(hdf5)


print(model.summary())

## get_session is deprecated in tf2
tfsession = tfv1.keras.backend.get_session()

graph_node_names = [args.outputLayer]

save_graph(tfsession,graph_node_names,args.outputDir,args.outputGraph)
