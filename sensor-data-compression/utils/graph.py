import tensorflow as tf
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

## Load qkeras/Keras model from json file
def load_model(f_model):
    with open(f_model,'r') as f:
        if 'QActivation' in f.read():
            from qkeras import QDense, QConv2D, QActivation,quantized_relu,quantized_bits,Clip,QInitializer
            f.seek(0)
            model = model_from_json(f.read(),
                                    custom_objects={'QActivation':QActivation,
                                                    'quantized_bits':quantized_bits,
                                                    'quantized_relu':quantized_relu,
                                                    'QConv2D':QConv2D,
                                                    'QDense':QDense,
                                                    'Clip':Clip,
                                                    'QInitializer':QInitializer})
            hdf5  = f_model.replace('json','hdf5')
            model.load_weights(hdf5)
        else:
            f.seek(0)
            model = model_from_json(f.read())
            hdf5  = f_model.replace('json','hdf5')
        model.load_weights(hdf5)
    return model

def set_quantized_weights(model,f_pkl):
    with open(f_pkl, 'rb') as f:
        #weights as a dictionary
        ws = pickle.load(f)
        for layer_name in ws.keys():
            layer = model.get_layer(layer_name)
            layer.set_weights(ws[layer_name]['weights'])
    return model

## Write model to graph
def write_frozen_graph(model,outputName="frozen_graph.pb",logdir='./',asText=False):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=logdir,
                      name=outputName,
                      as_text=asText)

## Load frozen graph
def load_frozen_graph(graph,printGraph=False):
    with tf.io.gfile.GFile(graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    tf.compat.v1.import_graph_def(graph_def, name="")
    
    # Build the tensor from the first and last node of the graph
    #     if isQK:
    #         inputs=["x:0"],
    #         outputs=["Identity:0"]
    #     else:
    #         inputs=["input_1:0"]
    #         outputs=["encoded_vector/Relu:0"]
    #
    inputs = graph_def.node[0].name+":0"
    outputs= graph_def.node[-1].name+":0"

    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=inputs,
                                    outputs=outputs,
                                    print_graph=printGraph)
    return frozen_func

#load performance pickles with flist = [{'label','p'}]
def load_pickles(flist):
    perf_dict = {}
    for f in flist:
        f_path = f['p']
        with open(f_path,'rb') as f_pkl:
            d = pickle.load(f_pkl)
            if 'label' in f.keys():
                for k in d.keys():
                    perf_dict[f['label']] = d[k]
            else:
                perf_dict.update(d)
    return perf_dict

## Helper function to load graph
def wrap_frozen_graph(graph_def, inputs,outputs,print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        if print_graph == True:
            for layer in layers:
                print(layer)
        print("-" * 50)
    
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

## Get the output from layer_index of input x from a model
def get_layer_output(model,layer_index,x):
    m = tf.keras.models.Model(
        inputs =model.inputs,
        outputs=model.layers[layer_index].output
    )
    return m.predict(x)

## plotAll the weights from model
def plot_weights(model,nBins=50):
    plt.figure(figsize=(8,6))
    for ilayer in range(1,len(model.layers)):
        if len(model.layers[ilayer].get_weights())>0:
            label = model.layers[ilayer].name
            data = np.histogram(model.layers[ilayer].get_weights()[0])
            print(ilayer, label,'unique weights',len(np.unique(model.layers[ilayer].get_weights()[0])))
            hep.histplot(data[0],data[1],label=label)
        else:
            print(ilayer,'no weights')
    plt.xlabel('weights')
    plt.ylabel('Entries')
    plt.yscale('log')            
    plt.legend()
    plt.savefig("%s_weights.pdf"%model.name)
    plt.clf()
    
#plot outputs from each layers given an input
def plot_outputs(model,x,layer_indices=[],nBins=10):
    plt.figure(figsize=(8,6))
    if len(layer_indices)>0:
        layers = layer_indices
    else:
        layers = range(1,len(model.layers))
    for ilayer in layers:
        label = model.layers[ilayer].name
        output,bins = np.histogram(layerOutput(model,ilayer,x).flatten(),nBins)
        hep.histplot(output,bins,label=label)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.xlabel('Output values')
    plt.ylabel('Entries')
    str_layers = "_".join([str(l) for l in layer_indices])
    plt.savefig("hist_outputs_%s.pdf"%str_layers)
    plt.clf()
    return

def plot_history(hist_dict,diff=False,title=None):
    plt.figure(figsize=(8,6))
    linestyles  = ['-', '--', '-.', ':',',']
    for i,(label,data) in enumerate(hist_dict.items()):
        print(label,data.keys())
        if diff:
            plt.plot(np.abs(np.array(data['loss'])-np.array(data['val_loss'])),label=label)      
        else:
#            plt.plot(data['loss']    ,marker = ls=linestyles[i],c='tab:blue',label=label+"_train")
            line, = plt.plot(data['loss']    ,label=label+"_train")
            plt.plot(data['val_loss'],ls=linestyles[1],c=line.get_color(),label=label+"_test")     
    plt.xlabel('epochs')
    if diff:
        plt.ylabel('Abs. Loss difference(Train-Test)')        
    else:
        plt.ylabel('Loss')
    plt.legend(loc='upper right',title=title)        
    plt.yscale('log')
    return

def plot_EMD(flist):
    perf_dict = loadPickles(flist)
    eval_settings={
        # compression algorithms, autoencoder and more traditional benchmarks
        'algnames' : ['ae','stc','thr_lo','thr_hi','bc'],
        # metrics to compute on the validation dataset
        'metrics' : {
            'EMD'      :emd,
            #'dMean':d_weighted_mean,
            #'dRMS':d_abs_weighted_rms,
        },
        "occ_nbins"   :12,
        "occ_range"   :(0,24),
        "occ_bins"    : [0,2,5,10,15],
        "chg_nbins"   :20,
        "chg_range"   :(0,200),
        "chglog_nbins":10,
        "chglog_range":(0,2.5),
        "chg_bins"    :[0,2,5,10,50],
        "occTitle"    :r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       ,
        "logMaxTitle" :r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
        "logTotTitle" :r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
        'ylim'        :None,
    }
    metrics = eval_settings['metrics']

    for mname in metrics:
        chgs=[]
        occs=[]
        for model_name in perf_dict:
#            print(model_name)
            plots = perf_dict[model_name]
            occs += [(model_name, plots["occ_"+mname+"_ae"])]
            chgs += [(model_name, plots["chg_"+mname+"_ae"])]        
        ylim_occ = (0,4)
        ylim_chg = None        
        OverlayPlots(occs,"ae_comp_occ_"+mname,xtitle=eval_settings['occTitle'],ytitle=mname,ylim=ylim_occ)
        OverlayPlots(chgs,"ae_comp_chgs_"+mname,xtitle=eval_settings['logTotTitle'],ytitle=mname,ylim=ylim_chg)
