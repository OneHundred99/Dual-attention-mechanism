"""
Networks for voxelwarp model
"""
#################################################################################################
#################################################################################################
from keras import layers, models,Model
from keras import backend as K
import keras.layers as KL
from keras.initializers import RandomNormal,Constant
from keras.layers import Conv3D,Reshape,Add,LeakyReLU,Multiply,Lambda,GlobalAveragePooling3D,Average,Dense,multiply,Maximum,Subtract,UpSampling3D,AveragePooling3D,concatenate,Permute,Input,BatchNormalization,add,Conv3DTranspose,Activation,MaxPooling3D,Dropout
K.set_image_data_format('channels_last')
#################################################################################################
###################  attention  #################################################################
from attention import PAM
###################  attention  #################################################################
#################################################################################################
# third party
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate,Dense,Flatten,BatchNormalization,add, pooling
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
from keras.optimizers import Adam

import keras
import keras.backend as K
from keras.layers.core import Dropout
import numpy as np
from structure import *
import losses
# local
from dense_3D_spatial_transformer import Dense3DSpatialTransformer
import losses
import os
import sys
sys.path.append('../ext')
from neuron.layers import  *


#################################################################################################
#################################################################################################

"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, Add,Dropout,BatchNormalization,MaxPooling3D,concatenate,add
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
import sys
sys.path.append('../ext/pynd-lib/pynd/')
import ndutils as nd
import sys
sys.path.append('../ext/neuron/')
from layers import *
from models import *
from utils import *
from keras.utils import plot_model

#################################################################################################
#################################################################################################


"""
U-Net+Res_block+Attention
"""

def BatchActivate(x):
    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = LeakyReLU(0.2)(x)
    return x
def res_block(x, nb_filters, strides):
	res_path = BatchActivate(x)
	res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0], kernel_initializer='he_normal')(res_path)

	res_path = BatchActivate(res_path)
    
	res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3 ,3), padding='same', strides=strides[1],kernel_initializer='he_normal')(res_path)
#    print('res_path_0:',main_path.shape)
	shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0],kernel_initializer='he_normal')(x)

	res_path = add([shortcut, res_path])
	return res_path

def encoderAttention(x):
    to_decoder = []

    main_path = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(x) #x0
    #print('main_path_0:',main_path.shape)
    main_path = BatchActivate(main_path)
    #print('main_path_00:',main_path.shape)
    
    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(main_path)#x1
   # print('main_path_1:',main_path.shape)
    main_path = BatchActivate(main_path)
   # print('main_path_11:',main_path.shape)
    
    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(main_path)#x2
    #print('main_path_2:',main_path.shape)
    
    shortcut = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(1, 1, 1),kernel_initializer='he_normal')(x)
   # print('shortcut:',shortcut.shape)
    main_path = add([shortcut, main_path])
     # first branching to decoder
    to_decoder.append(main_path)
    
    main_path = res_block(main_path, [32, 32], [(2, 2,2), (1,1, 1)])
    #print('main_path_3:',main_path.shape)
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2,2), (1, 1,1)])
    #print('main_path_4:',main_path.shape)
    to_decoder.append(main_path)


    main_path = res_block(main_path, [128, 128], [(2, 2, 2), (1, 1, 1)])
    #print('main_path_5:',main_path.shape)
    to_decoder.append(main_path)

    return to_decoder
def decoderAttention(x, from_encoder):  
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    #print('main_path_0:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[3]], axis=-1)
    #print('main_path_1:',main_path.shape)
    
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1,1, 1)])  
    #print('main_path2:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    #print('main_Path3:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[2]], axis=-1)
    #print('main_Path4:',main_path.shape)
    
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1, 1,1)])
    #print('main_Path5:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    #print('main_Path6:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[1]], axis=-1)
    #print('main_Path7:',main_path.shape)
    
    main_path = res_block(main_path, [4, 4], [(1, 1,1), (1, 1,1)])
    print('main_Path8:',main_path.shape)


    return main_path
def se_block(x, filters, ratio=16):
    """
    creates a squeeze and excitation block
    https://arxiv.org/abs/1709.01507

    Parameters
    ----------
    x : tensor
        Input keras tensor
    ratio : int
        The reduction ratio. The default is 16.
    Returns
    -------
    x : tensor
        A keras tensor
    """

    se_shape = (1, 1, filters)

    se = GlobalAveragePooling3D()(x)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([x, se])
    return x

def mutil_scale_decoder(x, from_encoder):  
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    #print('main_path_0:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[3]], axis=-1)
    #print('main_path_1:',main_path.shape)
    
    main_path11 = se_block(main_path,144)   
    #print('main_path_11:',main_path11.shape)
    main_path12 = PAM()(main_path)
    #print('main_path_12:',main_path12.shape)
    main_path = concatenate([main_path,main_path11],axis=-1)
    #print('main_path_3:',main_path.shape)
        
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1,1, 1)]) 
    #print('main_path4:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    #print('main_Path5:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[2]], axis=-1)
    #print('main_Path6:',main_path.shape)
    
    main_path = se_block(main_path,80)
    #print('main_Path7:',main_path.shape)
    
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1, 1,1)])
    #print('main_Path8:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    #print('main_Path9:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[1]], axis=-1)
    #print('main_Path10:',main_path.shape)
    
    main_path = se_block(main_path,48)
    #print('main_Path11:',main_path.shape)
    
    main_path = res_block(main_path, [4, 4], [(1, 1,1), (1, 1,1)])
    #print('main_Path12:',main_path.shape)


    return main_path


def unetAttention(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
    
    # inputs
    src = Input(shape=vol_size + (1,))  
    #print('src.shape:',src.shape)
    tgt = Input(shape=vol_size + (1,))
    #print('tgt.shape:',tgt.shape)      
    x_in = concatenate([src, tgt])  
    #print('x_in:',x_in.shape)
    # down-sample path.
#    x0 = myConv(x_in, enc_nf[0], 2)  
    to_decoder1 = encoderAttention(x_in)  
    
    path1 = res_block(to_decoder1[3], [16, 16], [(2, 2, 2), (1, 1, 1)])   
    #print('path1 :',path1.shape)
    
#    path1 = decoder(path1, from_encoder=to_decoder1)   
    path1 = mutil_scale_decoder(path1, from_encoder=to_decoder1)
    #print('path1_decoder :',path1.shape)
#    x = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax', name='output1',kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path1)
    
    path1 = UpSampling3D(size=(2, 2, 2))(path1)
    #print('path1_UP:',path1.shape)
    
    path1 = concatenate([path1,to_decoder1[0]], axis=-1)
    #print('path1_concatenate:',path1.shape)
    
    x = Conv3D(2, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=1)(path1)
    #print('x_Conv3D:',x.shape)
    if full_size:
        x = concatenate([x, x_in])
        x = myConvAttention(x, dec_nf[5])
        #print('x_myConv:',x.shape)
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConvAttention(x, dec_nf[6])
            print('x_myConv:',x.shape)
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
    # warp the source with the flow
    #y = Dense3DSpatialTransformer()([src, flow])
    y = SpatialTransformer(interp_method='linear', indexing='ij')([src, flow])
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])   # ??
    return model    

def myConvAttention(x_in, nf, strides=1):

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    
    return x_out


################################################################################################################################################
################################################################################################################################################

"""
"""

"""
U-Net+Res_block
"""

def BatchActivate(x):
    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = LeakyReLU(0.2)(x)
    return x
def res_block(x, nb_filters, strides):
	res_path = BatchActivate(x)
	res_path = Conv3D(filters=nb_filters[0], kernel_size=(3, 3, 3), padding='same', strides=strides[0], kernel_initializer='he_normal')(res_path)

	res_path = BatchActivate(res_path)
    
	res_path = Conv3D(filters=nb_filters[1], kernel_size=(3, 3 ,3), padding='same', strides=strides[1],kernel_initializer='he_normal')(res_path)
#    print('res_path_0:',main_path.shape)
	shortcut = Conv3D(nb_filters[1], kernel_size=(1, 1, 1), strides=strides[0],kernel_initializer='he_normal')(x)

	res_path = add([shortcut, res_path])
	return res_path

def encoder(x):
    to_decoder = []

    main_path = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(x) #x0
    print('main_path_0:',main_path.shape)
    main_path = BatchActivate(main_path)
    print('main_path_00:',main_path.shape)
    
    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(main_path)#x1
    print('main_path_1:',main_path.shape)
    main_path = BatchActivate(main_path)
    print('main_path_11:',main_path.shape)
    
    main_path = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),kernel_initializer='he_normal')(main_path)#x2
    print('main_path_2:',main_path.shape)
    
    shortcut = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(1, 1, 1),kernel_initializer='he_normal')(x)
    print('shortcut:',shortcut.shape)
    main_path = add([shortcut, main_path])
     # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [32, 32], [(2, 2,2), (1,1, 1)])
    print('main_path_3:',main_path.shape)
    to_decoder.append(main_path)

    main_path = res_block(main_path, [64, 64], [(2, 2,2), (1, 1,1)])
    print('main_path_4:',main_path.shape)
    to_decoder.append(main_path)


    main_path = res_block(main_path, [128, 128], [(2, 2, 2), (1, 1, 1)])
    print('main_path_5:',main_path.shape)
    to_decoder.append(main_path)

    return to_decoder
def decoder(x, from_encoder):   # dec_nf = [16,16,16,16,4,4,3]
    main_path = UpSampling3D(size=(2, 2, 2))(x)
    print('main_path_0:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[3]], axis=-1)
    print('main_path_1:',main_path.shape)
    
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1,1, 1)])  # 通道数修改了16
    print('main_path2:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    print('main_Path3:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[2]], axis=-1)
    print('main_Path4:',main_path.shape)
    
    main_path = res_block(main_path, [16, 16], [(1, 1,1), (1, 1,1)])
    print('main_Path5:',main_path.shape)
    
    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
    print('main_Path6:',main_path.shape)
    
    main_path = concatenate([main_path, from_encoder[1]], axis=-1)
    print('main_Path7:',main_path.shape)
    
    main_path = res_block(main_path, [4, 4], [(1, 1,1), (1, 1,1)])
    print('main_Path8:',main_path.shape)
#    main_path = UpSampling3D(size=(2, 2, 2))(main_path)
#    main_path = concatenate([main_path,from_encoder[0]], axis=-1)
#    main_path = res_block(main_path, [16, 16], [(1, 1, 1), (1, 1, 1)])

    return main_path
def unetBlock(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
        
    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    print('src.shape:',src.shape)
    tgt = Input(shape=vol_size + (1,))
    print('tgt.shape:',tgt.shape)      
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.
    print('x_in:',x_in.shape)
    # down-sample path.
#    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112     总的输入
    to_decoder1 = encoder(x_in)   #编码器 [8,16,16,16]
    
    path1 = res_block(to_decoder1[3], [16, 16], [(2, 2, 2), (1, 1, 1)])    #滤波器的个数需要修改,应该设置为多少
    print('path1 :',path1.shape)
    
    path1 = decoder(path1, from_encoder=to_decoder1)   #解码器[16,16,16,16,4,4,3]
    print('path1_decoder :',path1.shape)
#    x = Conv3D(4, kernel_size=(1, 1, 1), activation='softmax', name='output1',kernel_initializer='he_normal',bias_initializer=Constant(value=-10))(path1)
    path1 = UpSampling3D(size=(2, 2, 2))(path1)
    print('path1_UP:',path1.shape)
    
    path1 = concatenate([path1,to_decoder1[0]], axis=-1)
    print('path1_concatenate:',path1.shape)
    
    x = Conv3D(2, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=1)(path1)
    print('x_Conv3D:',x.shape)
    if full_size:
#        x = UpSampling3D()(x)
#        print('x_UP:',x.shape)
        x = concatenate([x, x_in]) #**********************报错*******************************************
        print('x_conca:',x.shape)
        x = myConvDemonsNew(x, dec_nf[5])
        print('x_myConv:',x.shape)
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConvDemonsNew(x, dec_nf[6])
            print('x_myConv:',x.shape)
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])   # ??
    return model    

def myConvDemonsNew(x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    
    return x_out


def GNet(vol_size, enc_nf, dec_nf):

    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))

    x_in = concatenate([src, tgt])
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14

    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    x = UpSampling3D()(x)
    x = concatenate([x, x_in])
    x = myConv(x, dec_nf[5])
    if(len(dec_nf) == 8):
        x = myConv(x, dec_nf[6])

    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)

    y = Dense3DSpatialTransformer()([src, flow])

    model = Model(inputs=[src, tgt], outputs=[y, flow])

    return model


def myConv(x_in, nf, strides=1):
    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
"""
"""
"""
def GNet(vol_size, enc_nf, dec_nf, full_size=True):


    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
     
    tgt = Input(shape=vol_size + (1,))
    
  
   
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.
    #add one dim
   
     
    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
     
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14
     

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    
    x = UpSampling3D()(x)
     
    x = concatenate([x, x2])
     
    x = myConv(x, dec_nf[1])
     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])
     
    x = myConv(x, dec_nf[2])
     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])
     
    x = myConv(x, dec_nf[3])
     
    x = myConv(x, dec_nf[4])
     
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])
         
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
     
    
    model = Model(inputs= [src, tgt], outputs =[y,flow])   # ??
 
    return model


def myConv(x_in, nf, strides=1):

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = BatchNormalization(x_out)
    
    x_out = LeakyReLU(0.3)(x_out)
   
    return x_out
"""
def Dnet_New(vol_size, enc_nf, dec_nf, full_size=True):
    src = Input(shape=vol_size + (1,))
    x = Conv3D(32, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=2)(src)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv3D(64, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv3D(128, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv3D(256, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=1)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
        
    model = Model(src,output)

    return model
    
    
    
def DNetCnn(vol_size, enc_nf, dec_nf, full_size=True):
    
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    
    tgt = Input(shape=vol_size + (1,))
   
   
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.
    
    
    # down-sample path.
    x0 = myConv(x_in, 10*enc_nf[0], 2)  # 80x96x112
    #x00 = Flattern()(x0)
    #x0_out = Dense(1,activation='sigmoid')(x00)
    
  
    x1 = myConv(x0, 10*enc_nf[1], 2)  # 40x48x56
    
    #x1= keras.layers.MaxPooling3D()(x1)
    
    x2 = myConv(x1, 10*enc_nf[2], 2)  # 20x24x28
    
    x3 = myConv(x2,10* enc_nf[3], 2)  # 10x12x14
 #   x3= keras.layers.MaxPooling3D()(x3)
    x  = LeakyReLU(0.05)(x3)
    x= Flatten()(x)
  #  x = Dense(64)(x)
   # x = Dense(32)(x)
  #  x= Dense(16)(x)
 #   x = Dense(8)(x)
    x =  Dropout(0.4)(x)
    output=  Dense(1,activation='sigmoid')(x)
    
    ##out = mean(x0_out,output)   
    # prepare model
    model = Model([src, tgt],output) 
    return model

def DNet(vol_size, enc_nf, dec_nf, full_size=True):
    src = Input(shape=vol_size + (1,))
    tgt = Input(shape=vol_size + (1,))

    x_in = concatenate([src, tgt])
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14

    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x2])
    x = myConv(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x1])
    x = myConv(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x0])
    x = myConv(x, dec_nf[3])
    x = myConv(x, dec_nf[4])

    x = UpSampling3D()(x)
    x = concatenate([x, x_in])
    x = myConv(x, dec_nf[5])
    if(len(dec_nf) == 8):
        x = myConv(x, dec_nf[6])
    x= Flatten()(x)    
    output=  Dense(1,activation='sigmoid')(x)
    model = Model([src, tgt],output) 
    return model
def GNet2(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """

    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
     
    tgt = Input(shape=vol_size + (1,))
    
  
   
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.
    #add one dim
   
     
    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 80x96x112
    
    x1 = myConv(x0, enc_nf[1], 2)  # 40x48x56
    
    x2 = myConv(x1, enc_nf[2], 2)  # 20x24x28
     
    x3 = myConv(x2, enc_nf[3], 2)  # 10x12x14
     

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    
    x = UpSampling3D()(x)
     
    x = concatenate([x, x2])
     
    x = myConv(x, dec_nf[1])
     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])
     
    x = myConv(x, dec_nf[2])
     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])
     
    x = myConv(x, dec_nf[3])
     
    x = myConv(x, dec_nf[4])
     
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])
         
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
     
    
    model = Model(inputs= [src, tgt], outputs =[y,flow])   # ??
 
    return model
def GNetMloss(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256),in experiment, the patch (64,64,64)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.

    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 32*32*32
    
    x1 = myConv(x0, enc_nf[1], 2)  # 16*16*16
    
    x2 = myConv(x1, enc_nf[2], 2)  # 8*8*8
     
    x3 = myConv(x2, enc_nf[3], 2)  # 4*4*4
     

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
     
    x = concatenate([x, x2])#8*8*8   
    
    x = myConv(x, dec_nf[1])
    flow1 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow1')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])#16*16*16
     
    x = myConv(x, dec_nf[2])
    
    flow2 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow2')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])#32*32*32
     
    x = myConv(x, dec_nf[3])
     
    x = myConv(x, dec_nf[4])
     
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])#64*64*64
         
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
     
    
  #  model = Model(inputs= [src, tgt], outputs =[y,flow])   # only one flow corresponding one loss
    model = Model(inputs= [src, tgt], outputs =[y,flow,flow2,flow1])
 
    return model

def GNetMloss_block(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256),in experiment, the patch (64,64,64)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.
    block_64_in= x_in
    block_64_out = identity_Block(block_64_in,6,3,with_conv_shortcut=True)
    
    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 32*32*32
    
    x1 = myConv(x0, enc_nf[1], 2)  # 16*16*16
    
    block_16_in = x1
    block_16_out= identity_Block(block_16_in,32,3,with_conv_shortcut=True)
    
    x2 = myConv(x1, enc_nf[2], 2)  # 8*8*8
     
    x3 = myConv(x2, enc_nf[3], 2)  # 4*4*4
     
    block_4_in = x3
    block_4_out = identity_Block(block_4_in,16,3,with_conv_shortcut=True)

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = add([x,block_4_out])    
    x = UpSampling3D()(x)
    
   
    
    x = concatenate([x, x2])#8*8*8   
    
    x = myConv(x, dec_nf[1])
    flow1 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow1')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])#16*16*16
     
    x = add([x, block_16_out])
    x = myConv(x, dec_nf[2])
    
    flow2 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow2')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])#32*32*32
     
    
    
    x = myConv(x, dec_nf[3])
    
     
    x = myConv(x, dec_nf[4])
     
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])#64*64*64
        x = add([x,block_64_out])
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
     
    
  #  model = Model(inputs= [src, tgt], outputs =[y,flow])   # only one flow corresponding one loss
    model = Model(inputs= [src, tgt], outputs =[y,flow,flow2,flow1])
 
    return model


def DNetMloss(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256),in experiment, the patch (64,64,64)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.

    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 32*32*32
    
    x1 = myConv(x0, enc_nf[1], 2)  # 16*16*16
    
    x2 = myConv(x1, enc_nf[2], 2)  # 8*8*8
     
    x3 = myConv(x2, enc_nf[3], 2)  # 4*4*4
     

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
     
    x = concatenate([x, x2])#8*8*8   
    
    x = myConv(x, dec_nf[1])
    #generate a similarity about 8*8*8
    output1_x= Flatten()(x)
    output1_x = Dense(4, activation='relu')(output1_x)
    output1_x = Dense(2, activation='relu')(output1_x)
    output1 = Dense(1, activation='sigmoid')(output1_x)
     #######
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])#16*16*16
     
    x = myConv(x, dec_nf[2])
    #generate a similarity about 16*16*16
    output2_x= Flatten()(x)
    output2_x = Dense(12, activation='relu')(output2_x)
    output2_x = Dense(8, activation='relu')(output2_x)
    output2 = Dense(1, activation='sigmoid')(output2_x)
     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])#32*32*32
     
    x = myConv(x, dec_nf[3])
     
    x = myConv(x, dec_nf[4])
     
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])#64*64*64
         
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
      ##add the dense layer, activation layer
    print('x='+ str(x.shape))
    x= Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
   
   
    # prepare model
    model = Model([src, tgt],[output,output2,output1])
    
  #  model = Model(inputs= [src, tgt], outputs =[y,flow])   # only one flow corresponding one loss
    
 
    return model
def GNetMlossDropout(vol_size, enc_nf, dec_nf, full_size=True):
    """
    unet network for voxelmorph 

    Args:
        vol_size: volume size. e.g. (256, 256, 256),in experiment, the patch (64,64,64)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]， [32,32,32,32,32,16,16,3]
            TODO: make this flexible.
        full_size

    """
    # inputs
    src = Input(shape=vol_size + (1,))  # Input()` is used to instantiate a Keras tensor.
    tgt = Input(shape=vol_size + (1,))
    x_in = concatenate([src, tgt])  # Functional interface to the `Concatenate` layer.

    # down-sample path.
    x0 = myConv(x_in, enc_nf[0], 2)  # 32*32*32
    
    x1 = myConv(x0, enc_nf[1], 2)  # 16*16*16
    
    x2 = myConv(x1, enc_nf[2], 2)  # 8*8*8
     
    x3 = myConv(x2, enc_nf[3], 2)  # 4*4*4
     

    # up-sample path.
    x = myConv(x3, dec_nf[0])
    x = UpSampling3D()(x)
     
    x = concatenate([x, x2])#8*8*8   
    
    x = myConv(x, dec_nf[1])
    flow1 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow1')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x1])#16*16*16
     
    x = myConv(x, dec_nf[2])
    
    flow2 = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow2')(x)

     
    x = UpSampling3D()(x)
     
    x = concatenate([x, x0])#32*32*32
     
    x = myConv(x, dec_nf[3])
     
    x = myConv(x, dec_nf[4])
    x = Dropout(0.5)(x) 
    if full_size:
        x = UpSampling3D()(x)
         
        x = concatenate([x, x_in])#64*64*64
         
        x = myConv(x, dec_nf[5])
         
        # optional convolution
        if (len(dec_nf) == 8):
            x = myConv(x, dec_nf[6])
             
    
    # transform the results into a flow.
    flow = Conv3D(dec_nf[-1], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
 ##add the dense layer, activation layer
    
    
    # warp the source with the flow
    y = Dense3DSpatialTransformer()([src, flow])
    
     
    
  #  model = Model(inputs= [src, tgt], outputs =[y,flow])   # only one flow corresponding one loss
    model = Model(inputs= [src, tgt], outputs =[y,flow,flow2,flow1])
 
    return model

def myConv(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """

    x_out = Conv3D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    
    x_out = BatchNormalization( )(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


