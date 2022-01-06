from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import sys

def get_model(name='float', shape, **kwargs):
    '''
    retrieve a floating point model or quantized keras model
    depending on the input name string
    '''
    if name == 'float':
        return get_float_model(shape=shape)
    elif name == 'quantized':
        return get_quantized_model(shape=shape, **kwargs)
    else:
        print('** error, could not create model! **\n')
        sys.exit(-1)


def get_float_model(shape = [64,32,32]):

    # declare model
    model = Sequential()
    model.add(Dense(shape[0], input_shape=(16,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu', name='relu1')

    # iterate through shape list and create a dense block for each layer width
    for index, fc_width in enumerate(shape[1:]):
        model.add(Dense(fc_width, name=f'fc{index+2}', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
        model.add(Activation(activation='relu', name=f'relu{index+2}')
        model.add(BatchNormalization())
    
    # declare softmax/output layer
    model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))
    model.add(BatchNormalization())

def get_quantized_model(shape = [64,32,32], fc_bits=6, fc_int_bits=0, relu_bits=0):

    model = Sequential()
    model.add(QDense(shape[0], input_shape=(16,), name='fc1',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                    model.add(BatchNormalization())
    model.add(QActivation(activation=quantized_relu(relu_bits), name='relu1'))

    # iterate through shape list and create a dense block for each layer width
    for index, fc_width in enumerate(shape[1:]):
        model.add(QDense(shape[index], name=f'fc{index+2}',
                        kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
                        model.add(BatchNormalization())
        model.add(QActivation(activation=quantized_relu(6), name='relu2'))

    # declare softmax/output layer
    model.add(QDense(5, name='output',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))