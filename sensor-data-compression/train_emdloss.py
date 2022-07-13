"""
For training EMD_CNN with different hyperparameters, loss functions, datasets
"""
import pair_emd_loss_cnn #Script for training the CNN to approximate EMD using pairs of real inputs
from ae_emd_cnn import ae_EMD_CNN #Approximating EMD using [input,AE] pairs
from app_emd_cnn import app_EMD_CNN #EMD using both of the above datasets
import pandas as pd
import os
import numpy as np
import argparse
from utils.logger import _logger

from train import load_data

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputFile", type=str, default='/ecoderemdvol/test_ae/', dest="inputFile",
                    help="previous AE training")
parser.add_argument("--epochs", type=int, default = 64, dest="num_epochs",
                    help="number of epochs to train")
parser.add_argument("--pairEMD", action='store_true', default = False,dest="pairEMD",
                    help="train EMD_CNN on pairs of real data")
parser.add_argument("--aeEMD", action='store_true', default = False,dest="aeEMD",
                    help="train EMD_CNN on [input,AE(input)] data")
parser.add_argument("--appEMD", action='store_true', default = False,dest="appEMD",
                    help="train EMD_CNN on pair+[input,AE(input)] data")

parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")
parser.add_argument("--nrowsPerFile", type=int, default=1000, dest="nrowsPerFile",
                    help="load nrowsPerFile in a directory")
parser.add_argument("--noHeader", action='store_true', default = False,dest="noHeader",
                    help="input data has no header")

parser.add_argument("--double", action='store_true', default = False,dest="double",
                    help="test PU400 by combining PU200 events")
parser.add_argument("--maskPartials", action='store_true', default = False,dest="maskPartials",
                    help="mask partial modules")
parser.add_argument("--maskEnergies", action='store_true', default = False,dest="maskEnergies",
                    help="Mask energy fractions <= 0.05")
parser.add_argument("--saveEnergy", action='store_true', default = False,dest="saveEnergy",
                    help="save SimEnergy from input data")


def main(args):
  
    if(not args.aeEMD):
        data=load_data(args)
    
    current_directory=os.getcwd()

    #Data to track the performance of various EMD_CNN models

    df=[]
    mean_data=[]
    std_data=[]
    nfilt_data=[]
    ksize_data=[]
    neuron_data=[]
    numlayer_data=[]
    convlayer_data=[]
    epoch_data=[]
    loss_data=[]

    #List of lists of Hyperparamters <- currently initialized from previous training
    hyp_list=[[32,5,256,1,3],
              [32,5,32,1,4],
              [64,5,32,1,4],
              [128,5,32,1,4],
              [128,5,64,1,3],
              [32,5,128,1,3],
              [128,3,256,1,4],
              [128,5,256,1,4]]
    
    loss_list=['huber_loss','msle','mse']
    
    num_epochs=args.num_epochs
    
    for hyp in hyp_list:
        num_filt=hyp[0]
        kernel_size=hyp[1]
        num_dens_neurons=hyp[2]
        num_dens_layers=hyp[3]
        num_conv_2d=hyp[4]
        
        #Each model per set of hyperparamters is trained thrice to avoid bad initialitazion discarding a good model. (We vary num_epochs by 1 to differentiate between these 3 trainings)
        
        for Loss in loss_list:
            for i in [0,1,2]:
                mean ,sd=0, 0
                if(args.aeEMD):
                    mean,sd=ae_EMD_CNN.ittrain(args.inputFile,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i,Loss)
                elif(args.appEMD):
                    mean,sd=app_EMD_CNN.ittrain(data,args.inputFile,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i,Loss)    
                elif(args.pairEMD):
                    obj=pair_emd_loss_cnn.pair_EMD_CNN()
                    mean, sd = obj.ittrain(data,num_filt,kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d,num_epochs+i,Loss)
                else:
                    print("Input which dataset(s) to train EMD_CNN on")
                mean_data.append(mean)
                std_data.append(sd)
                nfilt_data.append(num_filt)
                ksize_data.append(kernel_size)
                neuron_data.append(num_dens_neurons)
                numlayer_data.append(num_dens_layers)
                convlayer_data.append(num_conv_2d)
                epoch_data.append(num_epochs+i)
                loss_data.append(Loss)

    for_pdata=[mean_data,std_data,nfilt_data,ksize_data,neuron_data,numlayer_data,convlayer_data,epoch_data,loss_data]

    #Saving data from the entire optimization training 
    
    if(args.aeEMD):
        opt_data_directory=os.path.join(current_directory,'ae','EMD CNN Optimization Data.xlsx')
    if(args.appEMD):
        opt_data_directory=os.path.join(current_directory,'app','EMD CNN Optimization Data.xlsx')
    if(args.pairEMD):
        opt_data_directory=os.path.join(current_directory,'pair','EMD CNN Optimization Data.xlsx') 
    df=pd.DataFrame(for_pdata)
    df.to_excel(opt_data_directory)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

