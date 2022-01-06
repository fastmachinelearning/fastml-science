import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from brevitas.core.quant import QuantType


class three_layer_model_batnorm(nn.Module): # No "Masks" for the sake of exporting the model for HLS4ML
    def __init__(self, bn_affine = True, bn_stats = True ):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_batnorm, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64, affine=bn_affine, track_running_stats=bn_stats)
        self.bn2 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.bn3 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(self.bn1(test))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out

class three_layer_model_batnorm(nn.Module): # No "Masks" for the sake of exporting the model for HLS4ML
    def __init__(self, bn_affine = True, bn_stats = True ):
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_batnorm, self).__init__()
        self.quantized_model = False
        self.input_shape = 16  # (16,)
        self.fc1 = nn.Linear(self.input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64, affine=bn_affine, track_running_stats=bn_stats)
        self.bn2 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.bn3 = nn.BatchNorm1d(32, affine=bn_affine, track_running_stats=bn_stats)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(self.bn1(test))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        softmax_out = self.softmax(self.fc4(x))

        return softmax_out