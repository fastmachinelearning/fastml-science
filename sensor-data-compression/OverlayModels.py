
from qDenseCNN import qDenseCNN
from denseCNN import denseCNN

import matplotlib.pyplot as plt
import matplotlib

from martinModels import models
import graphUtil
import numba
import pickle

from plotWafer import plotWafer

from train import OverlayPlots,emd,d_weighted_mean,d_abs_weighted_rms


def loadPickles(flist):
    perf_dict = {}
    for key,f in flist.items():
        with open(f,'rb') as f_pkl:
            old_dict = pickle.load(f_pkl)
            for old_key,it in old_dict:
                new_key = key+old_key
                old_dict[new_key] = it
                del old_dict[old_key]
            perf_dict.update( old_dict ) 
    return perf_dict

def makePlots(flist,eval_settings,odir='.',tag=''):

    perf_dict = loadPickles(flist)
    algnames    =eval_settings[  "algnames"  ] 
    metrics     =eval_settings[  "metrics"  ] 
    occ_nbins    =eval_settings[  "occ_nbins"  ] 
    occ_range    =eval_settings[  "occ_range"   ]
    occ_bins     =eval_settings[  "occ_bins"    ]
    chg_nbins    =eval_settings[  "chg_nbins"   ]
    chg_range    =eval_settings[  "chg_range"   ]
    chglog_nbins =eval_settings[  "chglog_nbins"]
    chglog_range =eval_settings[  "chglog_range"]
    chg_bins     =eval_settings[  "chg_bins"    ]
    occTitle    =eval_settings["occTitle"   ]
    logMaxTitle =eval_settings["logMaxTitle"]
    logTotTitle =eval_settings["logTotTitle"]

    for mname in metrics:
        chgs=[]
        occs=[]
        for model_name in perf_dict:
            plots = perf_dict[model_name]
            chgs += [(model_name, plots["chg_"+mname+"_ae"])]
            occs += [(model_name, plots["occ_"+mname+"_ae"])]
        xt =eval_settings['logTotTitle']
        OverlayPlots(occ ,"ae_comp_%s_occ_"%tag+mname,xtitle=xt,ytitle=mname,odir=odir,ylim=(0,4))
        OverlayPlots(chgs,"ae_comp_%s_chg_"%tag+mname,xtitle=xt,ytitle=mname,odir=odir,ylim=(0,5))

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
}

odir = 'perf_plots/'
tag = ''
flist = {
    "electron": "../V11/signal/nElinks_5/Sep1_CNN_keras_norm/performance_Sep1_CNN_keras_norm.pkl",
    "electron_sim0p01": "../V11/signal/nElinks_5/Sep26_663/performance_Sep26_663.pkl",
}
makePlots(flist,eval_settings,odir)
