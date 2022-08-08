import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric

import scipy

from scipy import stats, optimize, interpolate

import os

import pandas as pd

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)])
    data_values=data.values
            
    return data_values

def load_phy(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(48,49)])
    data_values=data.values
            
    return data_values


def plot_eta(current_dir,models,phys):
    
    fig,ax=plt.subplots()
    plt.figure(figsize=(6,4))
    
    for model in models.split(','):
        input_dir=os.path.join(current_dir,model,'verify_input_calQ.csv')
        output_dir=os.path.join(current_dir,model,'verify_decoded_calQ.csv')
        
        input_Q=load_data(input_dir)
        output_Q=load_data(output_dir)
        
        indices = range(0,(len(input_Q)))
        
        emd_values = np.array([emd(input_Q[i],output_Q[j]) for i, j in zip(indices,indices)])
    
        eta=[]
        for i in indices:
            eta=np.append(eta,phys[i][0])
            
        x=eta
        y=emd_values
        
        nbins=10
        lims=None
        stats=True
        if lims==None: lims = (x.min(),x.max())
        median_result = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5))
        lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.68/2))
        hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.68/2))
        median = np.nan_to_num(median_result.statistic)
        hi = np.nan_to_num(hi_result.statistic)
        lo = np.nan_to_num(lo_result.statistic)
        hie = hi-median
        loe = median-lo
        bin_edges = median_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

        
        plt.errorbar(x=bin_centers, y=median, yerr=[loe,hie], label=model)
        fig=plt.legend()
    
    plt.legend()
    plt.xlabel(r'$\eta$')
    plt.ylabel('EMD')
    plt.savefig(current_dir+'emd_v_eta.pdf',dpi=600)
 
        

    
    
 
