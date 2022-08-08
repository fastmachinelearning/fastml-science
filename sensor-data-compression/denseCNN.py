from tensorflow.keras.layers import Layer,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import json
from telescope import telescopeMSE2

import tensorflow as tf
import inspect

class MaskLayer(Layer):
    def __init__(self,nFilter,arrMask):
        super(MaskLayer, self).__init__()
        self.nFilter = tf.constant(nFilter)
        self.arrayMask = np.array([arrMask])
        self.mask = tf.reshape(tf.stack(
                        tf.repeat(self.arrayMask,repeats=[nFilter],axis=0),axis=1),
                        shape=[-1])      
    def call(self, inputs):
        return tf.reshape(tf.boolean_mask(inputs,self.mask,axis=1),
                          shape=(tf.shape(inputs)[0],48*self.nFilter))
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nFilter': self.nFilter.numpy(),
            'arrMask': self.arrayMask.tolist(),
        })
        return config


class denseCNN:
    def __init__(self,name='',weights_f=''):
        self.name=name
        self.pams ={
            'CNN_layer_nodes'  : [8],  #n_filters
            'CNN_kernel_size'  : [3],
            'CNN_pool'         : [False],
            'CNN_padding'      : ['same'],
            'CNN_strides'      : [(1,1)],
            'Dense_layer_nodes': [], #does not include encoded layer
            'encoded_dim'      : 16,
            'shape'            : (4,4,3),
            'channels_first'   : False,
            'arrange'          : [],
            'arrMask'          : [],
            'calQMask'         : [],
            'maskConvOutput'   : [],
            'n_copy'           : 0,      # no. of copy for hi occ datasets
            'loss'             : '',
            'activation'       : 'relu',
            'optimizer'       : 'adam',
        }

        self.weights_f =weights_f
        

    def setpams(self,in_pams):
        for k,v in in_pams.items():
            self.pams[k] = v

    def shuffle(self,arr):
        order = np.arange(48)
        np.random.shuffle(order)
        return arr[:,order]
    
    def cloneInput(self,input_q,n_copy,occ_low,occ_hi):
        shape = self.pams['shape']
        nonzeroQs = np.count_nonzero(input_q.reshape(len(input_q),48),axis=1)
        selection = np.logical_and(nonzeroQs<=occ_hi,nonzeroQs>occ_low)
        occ_q     = input_q[selection]
        occ_q_flat= occ_q.reshape(len(occ_q),48)
        self.pams['cloned_fraction'] = len(occ_q)/len(input_q)
        for i in range(0,n_copy):
            clone   = self.shuffle(occ_q_flat)
            clone   = clone.reshape(len(clone),shape[0],shape[1],shape[2])
            input_q = np.concatenate([input_q,clone])
        return input_q
            
    def prepInput(self,normData):
      shape = self.pams['shape']

      if len(self.pams['arrange'])>0:
          arrange = self.pams['arrange']
          inputdata = normData[:,arrange]
      else:
          inputdata = normData
      if len(self.pams['arrMask'])>0:
          arrMask = self.pams['arrMask']
          inputdata[:,arrMask==0]=0  #zeros out repeated entries

      shaped_data = inputdata.reshape(len(inputdata),shape[0],shape[1],shape[2])

      if self.pams['n_copy']>0:
        n_copy  = self.pams['n_copy']
        occ_low = self.pams['occ_low']
        occ_hi = self.pams['occ_hi']
        shaped_data = self.cloneInput(shaped_data,n_copy,occ_low,occ_hi)

      return shaped_data

    def weightedMSE(self, y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        loss   = K.mean(K.square(y_true - y_pred)*K.maximum(y_pred,y_true),axis=(-1))
        return loss
            
    def init(self,printSummary=True):
        encoded_dim = self.pams['encoded_dim']

        CNN_layer_nodes   = self.pams['CNN_layer_nodes']
        CNN_kernel_size   = self.pams['CNN_kernel_size']
        CNN_padding       = self.pams['CNN_padding']
        CNN_strides       = self.pams['CNN_strides']
        CNN_pool          = self.pams['CNN_pool']
        Dense_layer_nodes = self.pams['Dense_layer_nodes'] #does not include encoded layer
        channels_first    = self.pams['channels_first']

        inputs = Input(shape=self.pams['shape'])  # adapt this if using `channels_first` image data format
        x = inputs

        for i,n_nodes in enumerate(CNN_layer_nodes):
            if channels_first:
              x = Conv2D(n_nodes, CNN_kernel_size[i], activation='relu', strides=CNN_strides[i], padding=CNN_padding[i],data_format='channels_first')(x)
            else:
              x = Conv2D(n_nodes, CNN_kernel_size[i], activation='relu', strides=CNN_strides[i], padding=CNN_padding[i])(x)
            if CNN_pool[i]:
              if channels_first:
                x = MaxPooling2D((2, 2), padding='same',data_format='channels_first')(x)
              else:
                x = MaxPooling2D((2, 2), padding='same')(x)

        shape = K.int_shape(x)

        x = Flatten()(x)

        if len(self.pams['maskConvOutput'])>0:
            if np.count_nonzero(self.pams['maskConvOutput'])!=48:
                raise ValueError("Trying to mask conv output with an array mask that does not contain exactly 48 calQ location. maskConvOutput = ",self.pams['maskConvOutput'])
            x = MaskLayer( nFilter = CNN_layer_nodes[-1] , arrMask = self.pams['maskConvOutput'] )(x)

        #encoder dense nodes
        for n_nodes in Dense_layer_nodes:
            x = Dense(n_nodes,activation='relu')(x)

        encodedLayer = Dense(encoded_dim, activation=self.pams['activation'],name='encoded_vector')(x)

        # Instantiate Encoder Model
        self.encoder = Model(inputs, encodedLayer, name='encoder')
        if printSummary:
          self.encoder.summary()

        encoded_inputs = Input(shape=(encoded_dim,), name='decoder_input')
        x = encoded_inputs

        #decoder dense nodes
        for n_nodes in Dense_layer_nodes:
             x = Dense(n_nodes, activation='relu')(x)

        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i,n_nodes in enumerate(CNN_layer_nodes):
            
            if CNN_pool[i]:
              if channels_first:
                  x = UpSampling2D((2, 2),data_format='channels_first')(x)
              else:
                  x = UpSampling2D((2, 2))(x)
            
            if channels_first:
              x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', strides=CNN_strides[i],padding=CNN_padding[i],data_format='channels_first')(x)
            else:
              x = Conv2DTranspose(n_nodes, CNN_kernel_size[i], activation='relu', strides=CNN_strides[i],padding=CNN_padding[i])(x)

        if channels_first:
          #shape[0] will be # of channel
          x = Conv2DTranspose(filters=self.pams['shape'][0],kernel_size=CNN_kernel_size[0],padding='same',data_format='channels_first')(x)
        else:
          x = Conv2DTranspose(filters=self.pams['shape'][2],kernel_size=CNN_kernel_size[0],padding='same')(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)


        self.decoder = Model(encoded_inputs, outputs, name='decoder')
        if printSummary:
          self.decoder.summary()

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        if printSummary:
          self.autoencoder.summary()
        
        self.compileModels()

        CNN_layers=''
        if len(CNN_layer_nodes)>0:
            CNN_layers += '_Conv'
            for i,n in enumerate(CNN_layer_nodes):
                CNN_layers += f'_{n}x{CNN_kernel_size[i]}'
                if CNN_pool[i]:
                    CNN_layers += 'pooled'
        Dense_layers = ''
        if len(Dense_layer_nodes)>0:
            Dense_layers += '_Dense'
            for n in Dense_layer_nodes:
                Dense_layers += f'_{n}'

        self.name = f'Autoencoded{CNN_layers}{Dense_layers}_Encoded_{encoded_dim}'
        
        if not self.weights_f=='':
            self.autoencoder.load_weights(self.weights_f)
        return


    def compileModels(self):
        opt = self.pams['optimizer']

        print('Using optimizer', opt)
        if self.pams['loss']=="weightedMSE":
            self.autoencoder.compile(loss=self.weightedMSE, optimizer=opt)
            self.encoder.compile(loss=self.weightedMSE, optimizer=opt)
        elif self.pams['loss'] == 'telescopeMSE':
            self.autoencoder.compile(loss=telescopeMSE2, optimizer=opt)
            self.encoder.compile(loss=telescopeMSE2, optimizer=opt)
        elif self.pams['loss']!='':
            self.autoencoder.compile(loss=self.pams['loss'], optimizer=opt)
            self.encoder.compile(loss=self.pams['loss'], optimizer=opt)
        else:
            self.autoencoder.compile(loss='mse', optimizer=opt)
            self.encoder.compile(loss='mse', optimizer=opt)
        return


    def get_models(self):
       return self.autoencoder,self.encoder

    def invertArrange(self,arrange,arrMask=[],calQMask=[]):
        remap =[]
        hashmap = {}  ## cell:index mapping
        ##Valid arrange check
        if not np.all(np.unique(arrange)==np.arange(48)):
            raise ValueError("Found cell location with number > 48. Please check your arrange:",arrange)
        foundDuplicateCharge = False
        if len(arrMask)==0:
            if len(arrange)>len(np.unique(arrange)):
                foundDuplicateCharge=True
        else:
            if len(arrange[arrMask==1])>len(np.unique(arrange[arrMask==1])):
                foundDuplicateCharge=True
    
        if foundDuplicateCharge and len(calQMask)==0:
            raise ValueError("Found duplicated charge arrangement, but did not specify calQmask")  
        if len(calQMask)>0 and np.count_nonzero(calQMask)!=48:
            raise ValueError("calQmask must indicate 48 calQ ")  
            
        for i in range(len(arrange)):
            if len(arrMask)>0 :
                ## fill hashmap only if arrMask allows it
                if arrMask[i]==1:   
                    if(foundDuplicateCharge):
                        ## fill hashmap only if calQMask allows it
                        if calQMask[i]==1: hashmap[arrange[i]]=i                    
                    else:
                        hashmap[arrange[i]]=i                    
            else:
                hashmap[arrange[i]]=i
        ## Always map to 48 calQ orders
        for i in range(len(np.unique(arrange))):
            remap.append(hashmap[i])
        return np.array(remap)

    ## remap input/output of autoencoder into CALQs orders
    def mapToCalQ(self,x):
        if len(self.pams['arrange']) > 0:
            arrange = self.pams['arrange']
            remap   = self.invertArrange(arrange,self.pams['arrMask'],self.pams['calQMask'])
            if len(self.pams['arrMask'])>0:
                imgSize =self.pams['shape'][0] *self.pams['shape'][1]* self.pams['shape'][2]
                x = x.reshape(len(x),imgSize)
                x[:,self.pams['arrMask']==0]=0 ## apply arrMask
                return x[:,remap]             ## map to calQ 
            else:
                return x.reshape(len(x),48)[:,remap]
        else:
            return x.reshape(len(x),48)

           
    def predict(self,x):
        decoded_Q = self.autoencoder.predict(x)
        encoded_Q = self.encoder.predict(x)
        encoded_Q = np.reshape(encoded_Q, (len(encoded_Q), self.pams['encoded_dim'], 1))
        return x,decoded_Q, encoded_Q

    def summary(self):
      self.encoder.summary()
      self.decoder.summary()
      self.autoencoder.summary()

    ##get params for writing json
    def get_pams(self):
      jsonpams={}      
      opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers,inspect.isclass))
      for k,v in self.pams.items():
          if type(v)==type(np.array([])):
              jsonpams[k] = v.tolist()
          elif  isinstance(v,opt_classes):
              config = {}
              for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
              jsonpams[k] = config
          elif  type(v)==type(telescopeMSE2):
              jsonpams[k] =str(v) 
          else:
              jsonpams[k] = v 
      return jsonpams
  
