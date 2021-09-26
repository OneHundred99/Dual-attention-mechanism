
"""
Created on Thu Sep  9 10:23:07 2021

@author: LiMeng
"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Average

from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD,RMSprop
import sys
import gc
import matplotlib.pyplot as plt
import sys
import numpy as np
import gan_network1 as networks
#import gan_networks_My as networks
import losses
import random
import glob
from pathlib import Path
import data_gen
import os
import keras
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
#tensorboardX
from tensorboardX import SummaryWriter
writer = SummaryWriter("tempsAttention3")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pfromD_size = (1,1)

base_data_dir = '../data/OASIS3/'


class GAN():
    def __init__(self,vol_size,lr,vmmodel,gpu_id,reg_param):
        self.vol_size = vol_size;
        self.latent_dim  = 100
        self.nf_enc = [16,32,32,32]
        self.nf_dec = [32,32,32,32,32,16,16,3]

        self.generator = self.build_generator(lr,vmmodel,gpu_id)
        
        src = Input(shape=self.vol_size + (1,))
        atlas = Input(shape = self.vol_size +(1,))
        gen_img,flow= self.generator([src,atlas])
               
        self.discriminator1 = self.build_discriminator()
        self.discriminator1.compile(loss='binary_crossentropy',optimizer=SGD(lr),metrics=['accuracy'])
        self.discriminator1.trainable = False
        validity = self.discriminator1([gen_img,atlas])
        self.combined1 = Model([src,atlas], [validity,gen_img,flow])
        self.combined1.compile(loss=['binary_crossentropy',losses.cc3D(),losses.gradientLoss('l2')],loss_weights=[0.2,0.6,0.2], optimizer=Adam(0.000001),metrics=['accuracy'])

    def build_generator(self, vmmodel='1',gpu_id=1,lr=0.000001):
        
        gpu = '/gpu:' + str(gpu_id)
        if(vmmodel =='1'):
            nf_dec  = [32,32,32,32,8,8,3]
        else:
            nf_dec = [32,32,32,32,32,16,16,3]
        nf_dec = self.nf_dec
        
        model= networks.unetAttention(vol_size,self.nf_enc, self.nf_dec)
        
        return model
    def build_discriminator(self,vmmodel='1',gpu_id=1,lr=0.00001):
         gpu = '/gpu:' + str(gpu_id)
          
         if(vmmodel =='1'):
            nf_dec  = [32,32,32,32,8,8,3]
         else:
            nf_dec = [32,32,32,32,32,16,16,3]
         nf_dec = self.nf_dec
         
         model= networks.DNetCnn(vol_size,self.nf_enc, self.nf_dec)
         
         return model

    def printLoss(self,step, training, train_loss):
         s = str(step) + "," + str(training)

         if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
            for i in range(len(train_loss)):
               s += "," + str(train_loss[i])
         else:
            s += "," + str(train_loss)

         file_handle=open('My_unet_cc3D.txt',mode='a')
         file_handle.write(s+'\n')
         file_handle.close()
         sys.stdout.flush()
        
    def train(self,batch_size=1,model_save_iter = 2,n_iterations=100):
        model_dir = '../models/oneGtwoDattention'
        gpu = '/gpu:' + str(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
        # Adversarial ground truths
        valid =np.ones((batch_size, 1))
        fake =np.zeros((batch_size, 1))
        patch_size = [64,64,64]
        stride = 32
        number = 1
        atlas_data =  glob.glob(base_data_dir + 'atlas_0129' + '/*.nii.gz')
        training_data =  glob.glob(base_data_dir + 'train' + '/*.nii.gz')
        print(training_data)

        num_training =len(training_data)
        print(num_training)
        train_patch2 = data_gen.single_vols_generator_patch(vol_name = atlas_data[0], num_data = number ,patch_size = patch_size,stride_patch = stride)
         
        patch_len = len(train_patch2)
        print('patch_len', patch_len)
        zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))
        alternation = 0
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for step in range(n_iterations):
            k = 0             
            while k < num_training:
                train_patch = data_gen.single_vols_generator_patch(vol_name = training_data[k], num_data = number ,patch_size = patch_size,stride_patch = stride)
             
                print('training image number: \n', k)
                k = k + 1
                for counter in range(patch_len):
                    X = train_patch[counter]
                    X = np.reshape(X, (1,) + X.shape)
                     
                    atlas_vol = train_patch2[counter]
                    atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape)
                     
                     
                    gen_imgs,_, = self.generator.predict([X, atlas_vol])

                    get_positive_img = train_patch[counter]*0.05+ train_patch2[counter] *0.95
                    get_positive_img = np.reshape(get_positive_img, (1,) + get_positive_img.shape)

                    d_loss_real= self.discriminator1.train_on_batch([get_positive_img,atlas_vol],valid)
                    #d_loss_real= self.discriminator1.train_on_batch([atlas_vol,atlas_vol],valid)
                         
                    d_loss_fake= self.discriminator1.train_on_batch([gen_imgs,atlas_vol],fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                     
                    g_loss = self.combined1.train_on_batch([X, atlas_vol],[valid,atlas_vol,zero_flow])

                    print ("%d [D loss: %f, acc.: %.2f%%]" % (step, d_loss[0], 100*d_loss[1]))
                    print(" [g_loss] : \n",g_loss)


                    alternation = alternation + 1

                    if not isinstance(g_loss, list):
                         g_loss = [g_loss]

                
                del train_patch
                gc.collect()

            if(step %model_save_iter == 0 ):
              self.generator.save(model_dir + '/' + str(step)+"_" + '.h5')
            writer.add_scalar('loss0', g_loss[0] , step)
            writer.add_scalar('loss1', g_loss[1] , step)
            writer.add_scalar('loss2', g_loss[2] , step)

                
if __name__ == '__main__':
    #vol_size=(144, 176, 144)
    vol_size=(64, 64, 64)
    lr=1e-4
    vmmodel='1'
    gpu_id=1
    reg_param =1.0
    gan = GAN(vol_size,lr,vmmodel,gpu_id,reg_param)
    
    gan.train()         
    print("end")
            
            
             
         
        
        
        
       
       
     
        