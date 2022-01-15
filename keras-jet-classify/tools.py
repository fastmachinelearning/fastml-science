import yaml
import csv
import math
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.utils import _add_supported_quantized_objects

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import itertools

# confusion matrix code from Maurizio
# /eos/user/m/mpierini/DeepLearning/ML4FPGA/jupyter/HbbTagger_Conv1D.ipynb
def plot_confusion_matrix(cm, classes,
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotRoc(fpr, tpr, auc, labels, linestyle, legend=True, save_dir=None):
    for i, label in enumerate(labels):
        plt.plot(tpr[label],fpr[label],label='%s tagger, AUC = %.1f%%'%(label.replace('j_',''),auc[label]*100.),linestyle=linestyle)
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    if legend: plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    if save_dir:
        plt.savefig(save_dir)

def rocData(y, predict_test, labels):

    df = pd.DataFrame()

    fpr = {}
    tpr = {}
    auc1 = {}

    for i, label in enumerate(labels):
        df[label] = y[:,i]
        df[label + '_pred'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
    return fpr, tpr, auc1

def makeRoc(y, predict_test, labels, linestyle='-', legend=True, save_dir=None):

    if 'j_index' in labels: labels.remove('j_index')
        
    fpr, tpr, auc1 = rocData(y, predict_test, labels)
    plotRoc(fpr, tpr, auc1, labels, linestyle, legend=legend, save_dir=save_dir)
    return predict_test

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def load_model(file_path):
    '''
    load keras model using keras.models.load_model
    '''
    co = {}
    _add_supported_quantized_objects(co)
    return keras.models.load_model(file_path, custom_objects=co)

def calc_BOPS(model, input_data_precision=32,):
    '''
    calculate number of bit operations during forward pass through through the MLP
    assuming:
         b_a is the output bitwidth of the last layer/input
         b_w is the current layer's bitwidth
         n - layer input nodes
         m - layer output nodes
    '''
    last_bit_width = input_data_precision
    total_BOPS = 0
    for layer in model.layers:
        if isinstance(layer, QDense) or isinstance(layer, Dense):
            b_a = last_bit_width
            b_w = layer.get_quantizers()[0].get_config()['bits'] if isinstance(layer, QDense)  else 32
            n = layer.input.get_shape()[1]
            m = layer.output.get_shape()[1]
            p = 1 # fraction of layer remaining after pruning
            module_BOPS = m * n * (p * b_a * b_w + b_a + b_w + math.log2(n))
            print("{} BOPS: {} = {}*{}({}*{}*{} + {} + {} + {})".format(layer.name,module_BOPS,m,n,p,b_a,b_w,b_a,b_w,math.log2(n)))
            last_bit_width = b_w
            total_BOPS += module_BOPS
    print("Total BOPS: {}".format(total_BOPS))
    return total_BOPS