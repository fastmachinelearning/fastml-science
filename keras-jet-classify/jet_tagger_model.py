from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import sys

def get_model(name, shape, **kwargs):
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


def get_float_model(shape = [64,32,32], **kwargs):

    # declare model
    model = Sequential()

    # create a dense blocks
    model.add(Dense(64, input_shape=(16,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu', name='relu1'))

    model.add(Dense(32, name=f'fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu', name=f'relu2'))

    model.add(Dense(32, name=f'fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu', name=f'relu3'))

    # declare softmax/output layer
    model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))

    return model

def get_quantized_model(shape = [64,32,32], fc_bits=6, fc_int_bits=0, relu_bits=0):

    # declare models
    model = Sequential()

    # create a dense blocks
    model.add(QDense(64, input_shape=(16,), name='fc1',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(QActivation(activation=quantized_relu(relu_bits), name='relu1'))


    model.add(QDense(32, name=f'fc2',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(QActivation(activation=quantized_relu(6), name=f'relu2'))

    model.add(QDense(32, name=f'fc3',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(BatchNormalization())
    model.add(QActivation(activation=quantized_relu(6), name=f'relu3'))

    # declare softmax/output layer
    model.add(QDense(5, name='output',
                    kernel_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1), bias_quantizer=quantized_bits(fc_bits,fc_int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))

    return model