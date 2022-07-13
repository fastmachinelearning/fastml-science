import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from denseCNN import MaskLayer
import argparse

def get_flops_from_pb_v2(model_json):
    with open(model_json,'r') as fjson:
        model = model_from_json(fjson.read(),custom_objects={'MaskLayer':MaskLayer})
        #hdf5  = model_json.replace('json','hdf5')
        #model.load_weights(hdf5)
        model.summary()
        print(model)
        inputs = [
            tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]
        full_model = tf.function(model).get_concrete_function(inputs)
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        for l in layers: print(l)
        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
        return flops.total_float_ops, model.count_params()
 
def get_flops_from_model(model):
        inputs = [
            tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]
        full_model = tf.function(model).get_concrete_function(inputs)
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        #for l in layers: print(l)
        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
        return flops.total_float_ops

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--flist", type=str, default='', dest="flist",
                        help="list of encoder .json files to calculate flops from")
    flist = flist.split(',')

    results = {}
    for f in flist:
        results[f.split('/')[-1]]={}
        flops, pams = get_flops_from_pb_v2(f)
        results[f.split('/')[-1]]['flops']= flops
        results[f.split('/')[-1]]['pams']= pams
    for k in results:
        print(k,results[k]['flops'],results[k]['pams'])
