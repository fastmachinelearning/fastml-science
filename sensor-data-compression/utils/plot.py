import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import scipy.stats

from .wafer import plot_wafer

def string_to_file(fname,s):
    with open(fname,'w') as f:
        f.write(s)
            
def plot_hist(vals,name,odir='.',xtitle="",ytitle="",nbins=40,lims=None,
             stats=True, logy=False, leg=None):
    plt.figure(figsize=(6,4))
    if leg:
        n, bins, patches = plt.hist(vals, nbins, range=lims, label=leg)
    else:
        n, bins, patches = plt.hist(vals, nbins, range=lims)
    ax = plt.gca()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    if stats:
        mu = np.mean(vals)
        std = np.std(vals)
        plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle if ytitle else 'Entries')
    if logy: plt.yscale('log')
    pname = odir+"/"+name+".pdf"
    tname = pname.replace('.pdf','.txt')
    string_to_file(tname, "{}, {}, {}, \t {}, {}, \n".format(np.quantile(vals,0.5), np.quantile(vals,0.5-0.68/2), np.quantile(vals,0.5+0.68/2), np.mean(vals), np.std(vals)))
    plt.savefig(pname)
    plt.close()
    return

def plot_loss(history,name):
    plt.figure(figsize=(8,6))
    plt.yscale('log')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss %s'%name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig("history_%s.pdf"%name)
    plt.close()
    plt.clf()
    return 
    
def visualize_displays(index,input_Q,input_calQ,decoded_Q,encoded_Q=np.array([]),conv2d=None,name='model_X'):
    Nevents = len(index)
    inputImg = input_Q[index]
    inputImgCalQ = input_calQ[index]
    outputImg = decoded_Q[index]

    fig, axs = plt.subplots(3, Nevents, figsize=(16, 10))
    for i in range(Nevents):
        if i==0:
            axs[0,i].set(xlabel='',ylabel='cell_y',title='Input_%i'%i)
        else:
            axs[0,i].set(xlabel='',title='Input_%i'%i)
        plot_wafer( inputImgCalQ[i], fig, axs[0,i])

    for i in range(Nevents):
        if i==0:
            axs[1,i].set(xlabel='cell_x',ylabel='cell_y',title='CNN Ouput_%i'%i)
        else:
            axs[1,i].set(xlabel='cell_x',title='CNN Ouput_%i'%i)
        plot_wafer( outputImg[i], fig, axs[1,i])

    if len(encoded_Q):
        encodedImg  = encoded_Q[index]
        for i in range(0,Nevents):
            if i==0:
                axs[2,i].set(xlabel='latent dim',ylabel='depth',title='Encoded_%i'%i)
            else:
                axs[2,i].set(xlabel='latent dim',title='Encoded_%i'%i)
            c1=axs[2,i].imshow(encodedImg[i])
            if i==Nevents:
                plt.colorbar(c1,ax=axs[2,i])
    plt.savefig("%s_examples.pdf"%name)
    plt.close()
    return

def visualize_metric(input_Q,decoded_Q,metric,name,odir,skipPlot=False):

    plot_hist(vals,name,options.odir,xtitle=longMetric[mname])
    
    plt.figure(figsize=(6,4))
    plt.hist([input_Q.flatten(),decoded_Q.flatten()],20,label=['input','output'])
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('Charge fraction')
    plt.savefig("hist_Qfr_%s.pdf"%name)
    plt.close()
    
    input_Q_abs = np.array([input_Q[i] * maxQ[i] for i in range(0,len(input_Q))])
    decoded_Q_abs = np.array([decoded_Q[i]*maxQ[i] for i in range(0,len(decoded_Q))])
    nonzeroQs = np.count_nonzero(input_Q_abs.reshape(len(input_Q_abs),48),axis=1)
    occbins = [0,5,10,20,48]
    
    fig, axes = plt.subplots(1,len(occbins)-1, figsize=(16, 4))
    for i,ax in enumerate(axes):
        selection = np.logical_and(nonzeroQs<occbins[i+1],nonzeroQs>occbins[i])
        label = '%i<occ<%i'%(occbins[i],occbins[i+1])
        mu = np.mean(cross_corr_arr[selection])
        std = np.std(cross_corr_arr[selection])
        plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
        ax.hist(cross_corr_arr[selection],40)
        ax.set(xlabel='corr',title=label)
        plt.savefig('corr_vs_occ_%s.pdf'%name)
        plt.close()
    return

def overlay_plots(results, name, xtitle="",ytitle="Entries",odir='.',text="",ylim=None):
    centers = results[0][1][0]
    wid = centers[1]-centers[0]
    offset = 0.33*wid
    plt.figure(figsize=(6,4))
    for ir,r in enumerate(results):
        lab = r[0]
        dat = r[1]
        off = offset * (ir-1)/2 * (-1. if ir%2 else 1.) # .1 left, .1 right, .2 left, ...
        plt.errorbar(x=dat[0]+off, y=dat[1], yerr=dat[2], label=lab)
    ax = plt.gca()
    plt.text(0.1, 0.9, name, transform=ax.transAxes)
    if text: plt.text(0.1, 0.82, text.replace('MAX','inf'), transform=ax.transAxes)
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc='upper right')
    pname = odir+"/"+name+".pdf"
    plt.savefig(pname)
    plt.close()
    return

def plot_profile(x,y,name,odir='.',xtitle="",ytitle="Entries",nbins=40,lims=None,
                 stats=True, logy=False, leg=None, text=""):
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

    plt.figure(figsize=(6,4))
    plt.errorbar(x=bin_centers, y=median, yerr=[loe,hie], linestyle='none', marker='.', label=leg)

    printstr=""
    for i,b in enumerate(bin_centers):
        printstr += "{} {} {} {} \n".format(b, median[i], loe[i], hie[i])

    ax = plt.gca()
    plt.text(0.1, 0.9, name,transform=ax.transAxes)
    if text: plt.text(0.1, 0.82, text.replace('MAX','inf'), transform=ax.transAxes)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if logy: plt.yscale('log')
    pname = odir+"/"+name+".pdf"
    tname = pname.replace('.pdf','.txt')
    string_to_file(tname, printstr)
    plt.savefig(pname)
    plt.close()
    return bin_centers, median, [loe,hie]
