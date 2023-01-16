import numpy as np
import pandas as pd
import argparse
import json
import pickle
import os
import numba
import tensorflow as tf
from tensorflow.keras import losses

from qkeras import get_quantizer,QActivation
from qkeras.utils import model_save_quantized_weights

from qDenseCNN import qDenseCNN
from denseCNN import denseCNN

from get_flops import get_flops_from_model

from utils.logger import _logger
from utils.plot import plot_loss, plot_hist, visualize_displays, plot_profile, overlay_plots
from emd_v_eta import plot_eta

parser = argparse.ArgumentParser()
parser.add_argument('-o',"--odir", type=str, default='output/', dest="odir",
                    help="output directory")
parser.add_argument('-i',"--inputFile", type=str, default='data/nElinks_5/', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--loss", type=str, default=None, dest="loss",
                    help="force loss function to use")
parser.add_argument("--quantize", action='store_true', default=False, dest="quantize",
                    help="quantize the model with qKeras. Default precision is 16,6 for all values.")
parser.add_argument("--epochs", type=int, default = 200, dest="epochs",
                    help="number of epochs to train")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")

parser.add_argument("--skipPlot", action='store_true', default=False, dest="skipPlot",
                    help="skip the plotting step")
parser.add_argument("--full", action='store_true', default = False,dest="full",
                    help="run all algorithms and metrics")

parser.add_argument("--quickTrain", action='store_true', default = False,dest="quickTrain",
                    help="train w only 5k events for testing purposes")
parser.add_argument("--retrain", action='store_true', default = False,dest="retrain",
                    help="retrain models even if weights are already present for testing purposes")
parser.add_argument("--evalOnly", action='store_true', default = False,dest="evalOnly",
                    help="only evaluate the NN on the input sample, no train")

parser.add_argument("--double", action='store_true', default = False,dest="double",
                    help="test PU400 by combining PU200 events")
parser.add_argument("--overrideInput", action='store_true', default = False,dest="overrideInput",
                    help="disable safety check on inputs")
parser.add_argument("--nCSV", type=int, default = 1, dest="nCSV",
                    help="n of validation events to write to csv")
parser.add_argument("--maxVal", type=int, default = -1, dest="maxVal",
                    help="clip outputs to maxVal")
parser.add_argument("--AEonly", type=int, default=1, dest="AEonly",
                    help="run only AE algo")
parser.add_argument("--rescaleInputToMax", action='store_true', default=False, dest="rescaleInputToMax",
                    help="rescale the input images so the maximum deposit is 1. Else normalize")
parser.add_argument("--rescaleOutputToMax", action='store_true', default=False, dest="rescaleOutputToMax",
                    help="rescale the output images to match the initial sum of charge")
parser.add_argument("--nrowsPerFile", type=int, default=500000, dest="nrowsPerFile",
                    help="load nrowsPerFile in a directory")
parser.add_argument("--occReweight", action='store_true', default = False,dest="occReweight",
                    help="train with per-event weight on TC occupancy")

parser.add_argument("--maskPartials", action='store_true', default = False,dest="maskPartials",
                    help="mask partial modules")
parser.add_argument("--maskEnergies", action='store_true', default = False,dest="maskEnergies",
                    help="Mask energy fractions <= 0.05")
parser.add_argument("--saveEnergy", action='store_true', default = False,dest="saveEnergy",
                    help="save SimEnergy from input data")
parser.add_argument("--noHeader", action='store_true', default = False,dest="noHeader",
                    help="input data has no header")

parser.add_argument("--models", type=str, default="8x8_c8_S2_tele", dest="models",
                    help="models to run, if empty string run all")

@numba.jit
def normalize(data,rescaleInputToMax=False, sumlog2=True):
    maxes =[]
    sums =[]
    sums_log2=[]
    for i in range(len(data)):
        maxes.append( data[i].max() )
        sums.append( data[i].sum() )
        sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
        if sumlog2:
            data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
        elif rescaleInputToMax:
            data[i] = 1.*data[i]/(data[i].max() if data[i].max() else 1.)
        else:
            data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
    if sumlog2:
        return  data,np.array(maxes),np.array(sums_log2)
    else:
        return data,np.array(maxes),np.array(sums)

@numba.jit
def unnormalize(norm_data,maxvals,rescaleOutputToMax=False, sumlog2=True):
    for i in range(len(norm_data)):
        if rescaleOutputToMax:
            norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].max() if norm_data[i].max() else 1.)
        else:
            if sumlog2:
                sumlog2 = 2**(np.floor(np.log2(norm_data[i].sum())))
                norm_data[i] =  norm_data[i] * maxvals[i] / (sumlog2 if sumlog2 else 1.)
            else:
                norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].sum() if norm_data[i].sum() else 1.)
    return norm_data

def load_data(args):
    # charge data headers of 48 Input Trigger Cells (TC) 
    CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]
    
    #Keep track of phys data
    COORD_COLS=['tc_eta','tc_phi']
    
    def mask_data(data,args):
        # mask rows where occupancy is zero
        mask_occupancy = (data[CALQ_COLS].astype('float64').sum(axis=1) != 0)
        data = data[mask_occupancy]
        
        if args.maskPartials:
            mask_isFullModule = np.isin(data.ModType.values,['FI','FM','FO'])
            _logger.info('Mask partial modules from input dataset')
            data = data[mask_isFull]
        if args.maskEnergies:
            try:
                mask_energy = data['SimEnergyFraction'].astype('float64') > 0.05
                data = data[mask_energy]
            except:
                _logger.warning('No SimEnergyFraction array in input data')
        return data
    
    if os.path.isdir(args.inputFile):
        df_arr = []
        phy_arr=[]
        for infile in os.listdir(args.inputFile):
            if os.path.isdir(args.inputFile+infile): continue
            infile = os.path.join(args.inputFile,infile)
            if args.noHeader:
                df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = args.nrowsPerFile, usecols=[*range(0,48)], names=CALQ_COLS))
                phy_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = args.nrowsPerFile, usecols=[*range(55,57)], names=COORD_COLS))
            else:
                df_arr.append(pd.read_csv(infile, nrows=args.nrowsPerFile))
        data = pd.concat(df_arr)
        phys = pd.concat(phy_arr)
    else:
        data = pd.read_csv(args.inputFile, nrows=args.nrowsPerFile)
    data = mask_data(data,args)

    if args.saveEnergy:
        try:
            simEnergyFraction = data['SimEnergyFraction'].astype('float64') # module simEnergyFraction w. respect to total event's energy
            simEnergy = data['SimEnergyTotal'].astype('float64') # module simEnergy
            simEnergyEvent = data['EventSimEnergyTotal'].astype('float64') # event simEnergy
        except:
            simEnergyFraction = None
            simEnergy = None
            simEnergyEvent = None
            _logger.warning('No SimEnergyFraction or SimEnergyTotal or EventSimEnergyTotal arrays in input data')

    data = data[CALQ_COLS].astype('float64')
    phys = phys[COORD_COLS]
    data_values = data.values
    phys_values = phys.values
    _logger.info('Input data shape')
    print(data.shape)
    data.describe()

    # duplicate data (e.g. for PU400?)
    if args.double:
        def double_data(data):
            doubled=[]
            i=0
            while i<= len(data)-2:
                doubled.append( data[i] + data[i+1] )
                i+=2
            return np.array(doubled)
        doubled_data = double_data(data_values.copy())
        _logger.info('Duplicated the data, the new shape is:')
        print(doubled_data.shape)
        data_values = doubled_data

    return (data_values,phys_values)

def build_model(args):
    # import network architecture and loss function
    from networks import networks_by_name

    # select models to run
    if args.models != "":
        m_to_run = args.models.split(',')
        models = [n for n in networks_by_name if n['name'] in m_to_run]
    else:
        models = networks_by_name
        
    nBits_encod = dict()
    nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0} # 0 to 2 range, 8 bit decimal 
        
    for m in models:
        if not 'nBits_encod' in m['params'].keys():
            m['params'].update({'nBits_encod':nBits_encod})
            
    nBits_input  = {'total': 10, 'integer': 3, 'keep_negative':1}
    nBits_accum  = {'total': 11, 'integer': 3, 'keep_negative':1}
    nBits_weight = {'total':  5, 'integer': 1, 'keep_negative':1} # sign bit not included

    for m in models:
        # print nbits for qkeras
        if m['isQK']:
             _logger.info('qKeras model weight {total}, {integer}, {keep_negative}'.format(**m['params']['nBits_weight']))
             _logger.info('qKeras model input {total}, {integer}, {keep_negative}'.format(**m['params']['nBits_input']))
             _logger.info('qKeras model accum {total}, {integer}, {keep_negative}'.format(**m['params']['nBits_accum']))
             _logger.info('qKeras model encod {total}, {integer}, {keep_negative}'.format(**m['params']['nBits_encod']))
             
        # re-use trained weights 
        if m['ws']=="":
            if os.path.exists(args.odir+m['name']+"/"+m['name']+".hdf5"):
                if args.retrain:
                    _logger.info('Found weights, but going to re-train as told.')
                    m['ws'] = ""
                else:
                    _logger.info('Found weights, using it by default')
                    m['ws'] = m['name']+".hdf5"
            else:
                _logger.info('Have not found trained weights in dir: %s'%(args.odir+m['name']+"/"+m['name']+".hdf5"))
        else:
            _logger.info('Found user input weights, using %s'%m['ws'])
            
        if args.loss:
            m['params']['loss'] = args.loss

    return models

def split(shaped_data, validation_frac=0.2,randomize=False):
    N = round(len(shaped_data)*validation_frac)
    if randomize:
        val_index = np.random.choice(shaped_data.shape[0], N, replace=False) # randomly select 25% entries
        full_index = np.array(range(0,len(shaped_data))) # select the indices of the other 75%
        train_index = np.logical_not(np.in1d(full_index,val_index))

        val_input = shaped_data[val_index]
        train_input = shaped_data[train_index]
    else:
        val_input = shaped_data[:N]
        train_input = shaped_data[N:]
        val_index = np.arange(N)
        train_index = np.arange(len(shaped_data))[N:]

    _logger.info('Training shape')
    print(train_input.shape)
    _logger.info('Validation shape')
    print(val_input.shape)
    return val_input,train_input,val_index,train_index

def train(autoencoder,encoder,train_input,train_target,val_input,name,n_epochs=100, train_weights=None):
    from tensorflow.keras import callbacks
    from tensorflow import keras as kr
    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

    if train_weights != None:
        history = autoencoder.fit(train_input,train_target,sample_weight=train_weights,epochs=n_epochs,batch_size=500,shuffle=True,validation_data=(val_input,val_input),callbacks=[es])
    else:
        history = autoencoder.fit(train_input,train_target,epochs=n_epochs,batch_size=500,shuffle=True,validation_data=(val_input,val_input),callbacks=[es])

    plot_loss(history,name)

    with open('./history_%s.pkl'%name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    isQK = False
    for layer in autoencoder.layers[1].layers:
        if QActivation == type(layer): isQK = True

    def save_models(autoencoder, name, isQK=False):
        from utils import graph
        
        json_string = autoencoder.to_json()
        encoder = autoencoder.get_layer("encoder")
        decoder = autoencoder.get_layer("decoder")
        with open('./%s.json'%name,'w') as f:        f.write(autoencoder.to_json())
        with open('./%s.json'%("encoder_"+name),'w') as f:            f.write(encoder.to_json())
        with open('./%s.json'%("decoder_"+name),'w') as f:            f.write(decoder.to_json())
        autoencoder.save_weights('%s.hdf5'%name)
        encoder.save_weights('%s.hdf5'%("encoder_"+name))
        decoder.save_weights('%s.hdf5'%("decoder_"+name))
        if isQK:
            encoder_qWeight = model_save_quantized_weights(encoder)
            with open('encoder_'+name+'.pkl','wb') as f:
                pickle.dump(encoder_qWeight,f)
            encoder = graph.set_quantized_weights(encoder,'encoder_'+name+'.pkl')
        graph.write_frozen_graph(encoder,'encoder_'+name+'.pb')
        graph.write_frozen_graph(encoder,'encoder_'+name+'.pb.ascii','./',True)
        graph.write_frozen_graph(decoder,'decoder_'+name+'.pb')
        graph.write_frozen_graph(decoder,'decoder_'+name+'.pb.ascii','./',True)
        
        graph.plot_weights(autoencoder)
        graph.plot_weights(encoder)
        graph.plot_weights(decoder)
    
    save_models(autoencoder,name,isQK)

    return history

def evaluate_model(model,charges,aux_arrs,eval_dict,args):
    # input arrays
    input_Q         = charges['input_Q']
    input_Q_abs     = charges['input_Q_abs']
    input_calQ      = charges['input_calQ']
    output_calQ     = charges['output_calQ']
    output_calQ_fr  = charges['output_calQ_fr']
    cnn_deQ         = charges['cnn_deQ']
    cnn_enQ         = charges['cnn_enQ']
    val_sum         = charges['val_sum']
    val_max         = charges['val_max']

    ae_out      = output_calQ
    ae_out_frac = normalize(output_calQ.copy())

    occupancy_1MT = aux_arrs['occupancy_1MT']

    # visualize 2D activations
    if not model['isQK']:
        conv2d  = None
    else:
        conv2d = kr.models.Model(
            inputs =model['m_autoCNNen'].inputs,
            outputs=model['m_autoCNNen'].get_layer("conv2d_0_m").output
        )

    occ_nbins = eval_dict['occ_nbins']
    occ_range = eval_dict['occ_range']
    occ_bins = eval_dict['occ_bins']
    
    chg_nbins = eval_dict['chg_nbins']
    chg_range = eval_dict['chg_range']
    chglog_nbins = eval_dict['chglog_nbins']
    chglog_range = eval_dict['chglog_range']
    chg_bins = eval_dict['chg_bins']
    
    occTitle = eval_dict['occTitle']
    logMaxTitle = eval_dict['logMaxTitle']
    logTotTitle = eval_dict['logTotTitle']
    
    longMetric = {'cross_corr':'cross correlation',
                  'SSD':'sum of squared differences',
                  'EMD':'earth movers distance',
                  'dMean':'difference in energy-weighted mean',
                  'dRMS':'difference in energy-weighted RMS',
                  'zero_frac':'zero fraction',}

    _logger.info("Running non-AE algorithms")
    if args.AEonly:
        alg_outs = {'ae' : ae_out}
    else:
        thr_lo_Q = np.where(input_Q_abs>1.35,input_Q_abs,0) # 1.35 transverse MIPs
        stc_Q = make_supercells(input_Q_abs, stc16=True)
        nBC={2:4, 3:6, 4:9, 5:14} #4, 6, 9, 14 (for 2,3,4,5 e-links)
        bc_Q = best_choice(input_Q_abs, nBC[args.nElinks])
        alg_outs = {
            'ae' : ae_out,
            'stc': stc_Q,
            #'bc': bc_Q,
            #'thr_lo': thr_lo_Q, 
        }

    model_name = model['name']
    plots={}
    summary_by_model = {
        'name':model_name,
        'en_pams' : model['m_autoCNNen'].count_params(),
        'en_flops' : get_flops_from_model(model['m_autoCNNen']),
        'tot_pams': model['m_autoCNN'].count_params(),
    }

    if (not args.skipPlot): plot_hist(np.log10(val_sum.flatten()),
                                      "sumQ_validation",xtitle=logTotTitle,ytitle="Entries",
                                      stats=True,logy=True,nbins=chglog_nbins,lims = chglog_range)
    if (not args.skipPlot): plot_hist([np.log10(val_max.flatten())],
                                      "maxQ_validation",xtitle=logMaxTitle,ytitle="Entries",
                                      stats=True,logy=True,nbins=chglog_nbins,lims = chglog_range)

    if (not args.skipPlot):
        from utils import graph
        for ilayer in range(0,len(model['m_autoCNNen'].layers)):
            label = model['m_autoCNNen'].layers[ilayer].name
            output,bins = np.histogram(graph.get_layer_output(model['m_autoCNNen'],ilayer,input_Q).flatten(),50)
            plots['hist_output_%s'%ilayer] = output,bins,label

    # compute metric for each algorithm
    for algname, alg_out in alg_outs.items():
        # event displays
        if(not args.skipPlot):
            Nevents = 8
            index = np.random.choice(input_Q.shape[0], Nevents, replace=False)
            visualize_displays(index, input_Q, input_calQ, alg_out, (cnn_enQ if algname=='ae' else np.array([])),(conv2d if algname=='ae' else None), name=algname)

        for mname, metric in eval_dict['metrics'].items():
            name = mname+"_"+algname
            vals = np.array([metric(input_calQ[i],alg_out[i]) for i in range(0,len(input_Q_abs))])

            model[name] = np.round(np.mean(vals), 3)
            model[name+'_err'] = np.round(np.std(vals), 3)
            summary_by_model[name] = model[name]
            summary_by_model[name+'_err'] = model[name+'_err']
            
            if(not args.skipPlot) and (not('zero_frac' in mname)):
                plot_hist(vals,"hist_"+name,xtitle=longMetric[mname])
                plot_hist(vals[vals>-1e-9],"hist_nonzero_"+name,xtitle=longMetric[mname])
                plot_hist(np.where(vals>-1e-9,1,0),"hist_iszero_"+name,xtitle=longMetric[mname])

                # 1d profiles
                plots["occ_"+name] = plot_profile(occupancy_1MT, vals,"profile_occ_"+name,
                                                  nbins=occ_nbins, lims=occ_range,
                                                  xtitle=occTitle,ytitle=longMetric[mname])
                plots["chg_"+name] = plot_profile(np.log10(val_max), vals,"profile_maxQ_"+name,ytitle=longMetric[mname],
                                                  nbins=chglog_nbins, lims=chglog_range,
                                                  xtitle=logMaxTitle if args.rescaleInputToMax else logTotTitle)

                # binned profiles in occupancy
                for iocc, occ_lo in enumerate(occ_bins):
                    occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
                    occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
                    indices = (occupancy_1MT >= occ_lo) & (occupancy_1MT < occ_hi)
                    pname = "chg_{}occ{}_{}".format(occ_lo,occ_hi_s,name)
                    plots[pname] = plot_profile(np.log10(val_max[indices]), vals[indices],"profile_"+pname,
                                                xtitle=logMaxTitle,
                                                nbins=chglog_nbins, lims=chglog_range,
                                                ytitle=longMetric[mname],
                                                text="{} <= occupancy < {}".format(occ_lo,occ_hi_s,name))

                # binned profiles in charge
                for ichg, chg_lo in enumerate(chg_bins):
                    chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
                    chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
                    indices = (val_max >= chg_lo) & (val_max < chg_hi)
                    pname = "occ_{}chg{}_{}".format(chg_lo,chg_hi_s,name)
                    plots[pname] = plot_profile(occupancy_1MT[indices], vals[indices],"profile_"+pname,
                                                xtitle=occTitle,
                                                ytitle=longMetric[mname],
                                                nbins=occ_nbins, lims=occ_range,
                                                text="{} <= Max Q < {}".format(chg_lo,chg_hi_s,name))
                    
    # overlay different metrics
    for mname in eval_dict['metrics']:
        chgs=[]
        occs=[]
        if(not args.skipPlot):
            for algname in alg_outs:
                name = mname+"_"+algname
                chgs += [(algname, plots["chg_"+mname+"_"+algname])]
                occs += [(algname, plots["occ_"+mname+"_"+algname])]
            xt = logMaxTitle if args.rescaleInputToMax else logTotTitle
            overlay_plots(chgs,"overlay_chg_"+mname,xtitle=xt,ytitle=mname)
            overlay_plots(occs,"overlay_occ_"+mname,xtitle=occTitle,ytitle=mname)

            # binned comparison
            for iocc, occ_lo in enumerate(occ_bins):
                occ_hi = 9e99 if iocc+1==len(occ_bins) else occ_bins[iocc+1]
                occ_hi_s = 'MAX' if iocc+1==len(occ_bins) else str(occ_hi)
                pname = "chg_{}occ{}_{}".format(occ_lo,occ_hi_s,name)
                pname = "chg_{}occ{}".format(occ_lo,occ_hi_s)
                chgs=[(algname, plots[pname+"_"+mname+"_"+algname]) for algname in alg_outs]
                overlay_plots(chgs,"overlay_chg_{}_{}occ{}".format(mname,occ_lo,occ_hi_s),
                              xtitle=logMaxTitle,ytitle=mname,
                              text="{} <= occupancy < {}".format(occ_lo,occ_hi_s,name))

            for ichg, chg_lo in enumerate(chg_bins):
                chg_hi = 9e99 if ichg+1==len(chg_bins) else chg_bins[ichg+1]
                chg_hi_s = 'MAX' if ichg+1==len(chg_bins) else str(chg_hi)
                pname = "occ_{}chg{}".format(chg_lo,chg_hi_s)
                occs=[(algname, plots[pname+"_"+mname+"_"+algname]) for algname in alg_outs]
                overlay_plots(occs,"overlay_occ_{}_{}chg{}".format(mname,chg_lo,chg_hi_s),
                             xtitle=occTitle, ytitle=mname,
                             text="{} <= Max Q < {}".format(chg_lo,chg_hi_s,name))

    return plots, summary_by_model

def compare_models(models,perf_dict,eval_dict,args):
    algnames = eval_dict['algnames']
    metrics = eval_dict['metrics']
    occ_nbins = eval_dict['occ_nbins']
    occ_range = eval_dict['occ_range']
    occ_bins = eval_dict['occ_bins']
    chg_nbins = eval_dict['chg_nbins']
    chg_range = eval_dict['chg_range']
    chglog_nbins = eval_dict['chglog_nbins']
    chglog_range = eval_dict['chglog_range']
    chg_bins = eval_dict['chg_bins']
    occTitle = eval_dict['occTitle']
    logMaxTitle = eval_dict['logMaxTitle']
    logTotTitle = eval_dict['logTotTitle']

    summary_entries=['name','en_pams','tot_pams','en_flops']
    for algname in algnames:
        for mname in metrics:
            name = mname+"_"+algname
            summary_entries.append(mname+"_"+algname)
            summary_entries.append(mname+"_"+algname+"_err")
    summary = pd.DataFrame(columns=summary_entries)

    with open('./performance.pkl', 'wb') as file_pi:
        pickle.dump(perf_dict, file_pi)

    if(not args.skipPlot):
        for mname in metrics:
            chgs=[]
            occs=[]
            for model_name in perf_dict:
                plots = perf_dict[model_name]
                short_model = model_name
                chgs += [(short_model, plots["chg_"+mname+"_ae"])]
                occs += [(short_model, plots["occ_"+mname+"_ae"])]
            xt = logMaxTitle if args.rescaleInputToMax else logTotTitle
            overlay_plots(chgs,"ae_comp_chg_"+mname,xtitle=xt,ytitle=mname)
            overlay_plots(occs,"ae_comp_occ_"+mname,xtitle=occTitle,ytitle=mname)

    for model in models:
        _logger.info('Summary_dict')
        print(model['summary_dict'])
        summary = summary.append(model['summary_dict'], ignore_index=True)
        
    print(summary)
    return

def main(args):
    _logger.info(args)

    if ("nElinks_%s"%args.nElinks not in args.inputFile):
        if not args.overrideInput:
            _logger.warning("nElinks={0} while 'nElinks_{0}' isn't in '{1}', this will cause wrong BC and STC settings - Exiting!".format(args.nElinks,args.inputFile))
            exit(0)

    # load data
    data_values,phys_values = load_data(args)
        
    # measure TC occupancy
    occupancy_all = np.count_nonzero(data_values,axis=1) # measure non-zero TCs (should be all)
    occupancy_all_1MT = np.count_nonzero(data_values>35,axis=1) # measure TCs with charge > 35

    # normalize input charge data
    # rescaleInputToMax: normalizes charges to maximum charge in module
    # sumlog2 (default): normalizes charges to 2**floor(log2(sum of charge in module)) where floor is the largest scalar integer: i.e. normalizes to MSB of the sum of charges (MSB here is the most significant bit)
    # rescaleSum: normalizes charges to sum of charge in module
    normdata,maxdata,sumdata = normalize(data_values.copy(),rescaleInputToMax=args.rescaleInputToMax,sumlog2=True)
    maxdata = maxdata / 35. # normalize to units of transverse MIPs
    sumdata = sumdata / 35. # normalize to units of transverse MIPs

    if args.occReweight:
        # reweight by occupancy (number of bins, range up, range down)
        def get_weights(vals, n=None, a=None, b=None):
            if a==None: a=min(vals)
            if b==None: b=max(vals)
            if n==None: b=20
            # weight histogram
            contents, bins, patches = plt.hist(vals, n, range=(a,b))
            
            def _get_bin(x,bins):
                if x < bins[0]: return 0
                if x >= bins[-1]: return len(bins)-2
                for i in range(len(bins)-1):
                    if x>= bins[i] and x<bins[i+1]: return i
                return 0

            _bins = np.array([_get_bin(x,bins)for x in vals])
            return np.array([1./contents[b] for b in _bins]) # must be filled by construction
        weights_occ = get_weights(occupancy_all_1MT,50,0,50)
        weights_maxQ = get_weights(maxdata,50,0,50)

    # build default AE models
    models = build_model(args)
    
    # evaluate performance
    from utils.metrics import emd,d_weighted_mean,d_abs_weighted_rms,zero_frac,ssd
    
    eval_dict={
        # compare to other algorithms
        'algnames'    :['ae','stc','thr_lo','thr_hi','bc'],
        'metrics'     :{'EMD':emd},
        "occ_nbins"   :12,
        "occ_range"   :(0,24),
	"occ_bins"    : [0,2,5,10,15],
	"chg_nbins"   :20,
        "chg_range"   :(0,200),
        "chglog_nbins":20,
        "chglog_range":(0,2.5),
        "chg_bins"    :[0,2,5,10,50],
        "occTitle"    :r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       ,
        "logMaxTitle" :r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
	"logTotTitle" :r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
    }
    if args.full:
        eval_dict['metrics'].update({'EMD':emd,
                                     'dMean':d_weighted_mean,
                                     'dRMS':d_abs_weighted_rms,
                                     'zero_frac':(lambda x,y: np.all(y==0)),
                                     'SSD':ssd,
                                     })

    orig_dir = os.getcwd()
    if not os.path.exists(args.odir): os.makedirs(args.odir)
    os.chdir(args.odir)

    if(not args.skipPlot):
        # plot occupancy
        plot_hist(occupancy_all.flatten(),"occ_all",xtitle="occupancy (all cells)",ytitle="evts",
                  stats=False,logy=True,nbins=50,lims=[0,50])
        plot_hist(occupancy_all_1MT.flatten(),"occ_1MT",xtitle=r"occupancy (1 MIP$_{\mathrm{T}}$ cells)",ytitle="evts",
                  stats=False,logy=True,nbins=50,lims=[0,50])
        plot_hist(np.log10(maxdata.flatten()),"maxQ_all",xtitle=eval_dict['logMaxTitle'],ytitle="evts",
                  stats=False,logy=True,nbins=50,lims=[0,2.5])
        plot_hist(np.log10(sumdata.flatten()),"sumQ_all",xtitle=eval_dict['logTotTitle'],ytitle="evts",
                  stats=False,logy=True,nbins=50,lims=[0,2.5])
        
    # performance dictionary
    perf_dict={}
    
    #Putting back physics columns below once training is done
    Nphys = round(len(phys_values)*0.2)
    phys_val_input = phys_values[:Nphys]
    phys_val_input=phys_val_input
    
    # train each model
    for model in models:
        model_name = model['name']
        if not os.path.exists(model_name): os.mkdir(model_name)
        os.chdir(model_name)
        
        if model['isQK']:
            _logger.info("Model is a qDenseCNN")
            m = qDenseCNN(weights_f=model['ws'])
        else:
            _logger.info("Model is a denseCNN")
            m = denseCNN(weights_f=model['ws'])
        m.setpams(model['params'])
        m.init()

        shaped_data = m.prepInput(normdata)

        # split in training/validation datasets
        if args.evalOnly:
            _logger.info("Eval only")
            val_input = shaped_data
            val_ind = np.array(range(len(shaped_data)))
            train_input = val_input[:0] #empty with correct shape                                                                                                                                           
            train_ind = val_ind[:0]
        else:
            val_input, train_input, val_ind, train_ind = split(shaped_data)
            
        m_autoCNN , m_autoCNNen = m.get_models()
        model['m_autoCNN'] = m_autoCNN
        model['m_autoCNNen'] = m_autoCNNen

        val_max = maxdata[val_ind]
        val_sum = sumdata[val_ind]
        if args.occReweight:
            train_weights = np.multiply(weights_maxQ[train_ind], weights_occ[train_ind])
        else:
            train_weights = np.ones(len([train_input]))

        if args.maxVal>0:
            _logger.info('Clipping outputs')
            val_input = val_input[:args.maxVal]
            val_max = val_max[:args.maxVal]
            val_sum = val_sum[:args.maxVal]

        if model['ws']=='':
            if args.quickTrain:
                train_input = train_input[:5000]
                train_weights = train_weights[:5000]
            if args.occReweight:
                history = train(m_autoCNN,m_autoCNNen,
                                train_input,train_input,val_input,
                                name=model_name,
                                n_epochs = args.epochs,
                                train_weights=train_weights)
            else:
                history = train(m_autoCNN,m_autoCNNen,
                                train_input,train_input,val_input,
                                name=model_name,
                                n_epochs = args.epochs,
                                )
        else:
            if args.retrain: # retrain w input weights
                history = train(m_autoCNN,m_autoCNNen,
                                train_input,train_input,val_input,
                                name=model_name,
                                n_epochs = args.epochs,
                                )
                pass

        # evaluate model
        _logger.info('Evaluate AutoEncoder, model %s'%model_name)
        input_Q, cnn_deQ, cnn_enQ = m.predict(val_input)
        
        input_calQ  = m.mapToCalQ(input_Q)   # shape = (N,48) in CALQ order
        output_calQ_fr = m.mapToCalQ(cnn_deQ)   # shape = (N,48) in CALQ order
        _logger.info('inputQ shape')
        print(input_Q.shape)
        _logger.info('inputcalQ shape')
        print(input_calQ.shape)

        _logger.info('Restore normalization')
        input_Q_abs = np.array([input_Q[i]*(val_max[i] if args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_Q))]) * 35.   # restore abs input in CALQ unit                              
        input_calQ  = np.array([input_calQ[i]*(val_max[i] if args.rescaleInputToMax else val_sum[i]) for i in range(0,len(input_calQ)) ])  # shape = (N,48) in CALQ order                                
        output_calQ =  unnormalize(output_calQ_fr.copy(), val_max if args.rescaleOutputToMax else val_sum, rescaleOutputToMax=args.rescaleOutputToMax)

        isRTL = False
        if isRTL:
            _logger.info('Save CSV for RTL verification')
            N_csv= (args.nCSV if args.nCSV>=0 else input_Q.shape[0]) # about 80k                                                                                                                          
            AEvol = m.pams['shape'][0]* m.pams['shape'][1] *  m.pams['shape'][2]
            np.savetxt("verify_input_ae.csv", input_Q[0:N_csv].reshape(N_csv,AEvol), delimiter=",",fmt='%.12f')
            np.savetxt("verify_input_ae_abs.csv", input_Q_abs[0:N_csv].reshape(N_csv,AEvol), delimiter=",",fmt='%.12f')
            np.savetxt("verify_input_calQ.csv", np.hstack((input_calQ[0:N_csv].reshape(N_csv,48),phys_val_input)), delimiter=",",fmt='%.12f')
            np.savetxt("verify_output.csv",cnn_enQ[0:N_csv].reshape(N_csv,m.pams['encoded_dim']), delimiter=",",fmt='%.12f')
            np.savetxt("verify_decoded.csv",cnn_deQ[0:N_csv].reshape(N_csv,AEvol), delimiter=",",fmt='%.12f')
            np.savetxt("verify_decoded_calQ.csv",np.hstack((output_calQ_fr[0:N_csv].reshape(N_csv,48),phys_val_input)), delimiter=",",fmt='%.12f')
            
            #plot_eta(input_calQ[0:N_csv].reshape(N_csv,48), output_calQ_fr[0:N_csv].reshape(N_csv,48), phys_val_input)

        _logger.info('Renormalize inputs of AE for comparisons')
        occupancy_0MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48),axis=1)
        occupancy_1MT = np.count_nonzero(input_calQ.reshape(len(input_Q),48)>1.,axis=1)

        charges = {
            'input_Q'    : input_Q,               
            'input_Q_abs': input_Q_abs,           
            'input_calQ' : input_calQ,            # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
            'output_calQ': output_calQ,           # shape = (N,48) (in abs Q)   (in CALQ 1-48 order)
            'output_calQ_fr': output_calQ_fr,     # shape = (N,48) (in Q fr)   (in CALQ 1-48 order)
            'cnn_deQ'    : cnn_deQ,
            'cnn_enQ'    : cnn_enQ,
            'val_sum'    : val_sum,
            'val_max'    : val_max,
        }
        
        aux_arrs = {
           'occupancy_1MT':occupancy_1MT
        }
        
        perf_dict[model['label']] , model['summary_dict'] = evaluate_model(model,charges,aux_arrs,eval_dict,args)

        os.chdir('../')

    # compare the relative performance of each model
    compare_models(models,perf_dict,eval_dict,args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
