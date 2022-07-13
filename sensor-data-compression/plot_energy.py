import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputFile", type=str, default='nElinks_5/', dest="inputFile",
                    help="input TSG files")
parser.add_argument("--models", type=str, default="8x8_c8_S2_tele", dest="models",
                    help="models to run, if empty string run all")
parser.add_argument("--nELinks", type=int, default = 5, dest="nElinks",
                    help="n of active transceiver e-links eTX")

# extra args needed to import model
parser.add_argument("--retrain", action='store_true', default = False,dest="retrain",
                    help="retrain models even if weights are already present for testing purposes")
parser.add_argument('-o',"--odir", type=str, default='CNN/PU/', dest="odir",
                    help="output directory")
parser.add_argument("--loss", type=str, default=None, dest="loss",
                    help="force loss function to use")

args = parser.parse_args()
    
data = pd.read_csv(args.inputFile, nrows=500000)

# charge data headers of 48 Input Trigger Cells (TC)
CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]
    
# mask rows where occupancy is zero
mask_occupancy = (data[CALQ_COLS].astype('float64').sum(axis=1) != 0)
data = data[mask_occupancy]

simEnergyFraction = data['SimEnergyFraction'].astype('float64') # module simEnergyFraction w. respect to total event's energy
simEnergy = data['SimEnergyTotal'].astype('float64') # module simEnergy
simEnergyEvent = data['EventSimEnergy'].astype('float64') # event simEnergy

# get data and normalize
data = data[CALQ_COLS].astype('float64')
data_values = data.values
from train import normalize,build_model
normdata,maxdata,sumdata = normalize(data_values.copy(),rescaleInputToMax=0,sumlog2=False)

# build our default model
models = build_model(args)
    
# set model
for model in models:
    # load model
    from denseCNN import denseCNN
    m = denseCNN(weights_f=model['ws'])
    print('load ')
    m.setpams(model['params'])
    print(model['params'])
    m.init()
    print('init ')

    # re shape the data
    print('norm data ',normdata)
    shaped_data = m.prepInput(normdata)
    print(shaped_data.shape)
    
    # map to calQ
    input_calQ  = m.mapToCalQ(shaped_data)
    
    # pick 2 random events
    Nevents = 2
    index = np.random.choice(shaped_data.shape[0], Nevents, replace=False)

    from utils.plot import plot_wafer
    fig, axs = plt.subplots(1, Nevents)
    for i in range(Nevents):
        if i==0:
            axs[i].set(xlabel='cell_x',ylabel='cell_y',title='Input_%i'%i)
        else:
            axs[i].set(xlabel='cell_x',title='Input_%i'%i)
        plot_wafer( input_calQ[i], fig, axs[i])
    plt.savefig("input.pdf")
    plt.close()
