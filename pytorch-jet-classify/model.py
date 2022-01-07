import torch
import torch.nn as nn
import numpy as np
import brevitas.nn as qnn
from brevitas.core.quant import QuantType


class three_layer_model_batchnorm(nn.Module): # No "Masks" for the sake of exporting the model for HLS4ML
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

class three_layer_model_batchnorm_quantized(nn.Module):
    def __init__(self, precision = 8):
        self.weight_precision = precision
        # Model with <16,64,32,32,5> Behavior
        super(three_layer_model_batchnorm_quantized, self).__init__()
        self.input_shape = int(16)  # (16,)
        self.quantized_model = True #variable to inform some of our plotting functions this is quantized
        self.fc1 = qnn.QuantLinear(self.input_shape, int(64),
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc2 = qnn.QuantLinear(64, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc3 = qnn.QuantLinear(32, 32,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.fc4 = qnn.QuantLinear(32, 5,
                                   bias=True,
                                   weight_quant_type=QuantType.INT,
                                   weight_bit_width=self.weight_precision)
        self.act1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6) #TODO Check/Change this away from 6, do we have to set a max value here? Can we not?
        self.act2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.act3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=self.weight_precision, max_val=6)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        test = self.fc1(x)
        x = self.act1(test)
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        softmax_out = self.softmax(self.fc4(x))
        return softmax_out
