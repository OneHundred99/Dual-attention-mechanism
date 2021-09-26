# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 21:35:20 2021

@author: Lenovo
"""

import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio
import gan_network1 as networks

import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam
from dense_3D_spatial_transformer import Dense3DSpatialTransformer

import matplotlib.pyplot as plt
from keras import backend as K
import data_gen
from scipy.interpolate import interpn

sys.path.append('../ext/medipy-lib')
from medipy.metrics import dice
sys.path.append('../ext/neuron')
import models
import apps as app
import nibabel as nib
import os
import gc
import commonutils as com
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

base_data_dir = '../data/OASIS3/'
model_dir = '../models/oneGtwoDattention/' 
postname = "nii.gz"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../ext/pynd-lib/pynd/')
import ndutils as nd

config = com.Config()
labels = config.read_labecsv(base_data_dir + 'label.csv')

print(labels)
def printTotalDice(mean,std,i,filename='../print_image/oneGtwoDattention/diceTotal.txt'):
    file_handle=open(filename,mode='a')
    s = 'mean: '+str(mean) + "  , std: "+ str(std)
    file_handle.write(s+'\n')
    file_handle.close()
    sys.stdout.flush()
def test(model_name, iters, gpu_id):
    patch_size = (64,64,64)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    stride = 32

    vol_size=(64, 64, 64)

    nf_enc = [16, 32, 32, 32]

    nf_dec = [32,32,32,32,8,8,3]

    validation_vols_data = glob.glob(base_data_dir + 'validation/test/*.'+postname)
    validation_labels_data = glob.glob(base_data_dir + 'validation/testlabel/*.'+postname)
    atlas_vols_data = glob.glob(base_data_dir + 'validation/atlas_0129/*.'+postname)
    atlasseg_data= glob.glob(base_data_dir + 'validation/atlaslabel_0129/*.'+postname)           
  
    model = networks.unetAttention(vol_size,nf_enc,nf_dec)

    model.summary()
    atlas_nii = nib.load(atlas_vols_data[0])
    data_nii = nib.load(validation_vols_data[0])

    ref_affine = data_nii.affine.copy()
    ref_aff = atlas_nii.affine.copy()
    data_nii = data_nii.get_data()
    data_temp = nib.load(validation_vols_data[0]).get_data()
    model_num = len(model_name)
    for i in range(model_num):
        current_model =model_name[i]
        print('current model', current_model)
        model.load_weights(current_model)
        
        length = len(validation_vols_data)
        print("length: ",length)
        for j in range(length):# the number of the images        
            validation_vols, vol_patch_loc = data_gen.single_vols_generator_patch(validation_vols_data[j],len(validation_vols_data),patch_size,stride_patch=stride,out=2,datatype=postname)
            atlas_vols, atlas_patch_loc = data_gen.single_vols_generator_patch(atlas_vols_data[0],len(atlas_vols_data),patch_size, stride_patch = stride, out =2,datatype=postname)

            dice_scores_mean = []
            dice_scores_std=[]
            mask1=np.empty(data_temp.shape+(3,)).astype('float32')
            mask2=np.empty(data_temp.shape+(3,)).astype('float32')
            patch_number = len(validation_vols)
            print('patch_number:',patch_number)
            for a in range(patch_number):

                _, pred_temp = model.predict([validation_vols[a],atlas_vols[a]])
                
                mask1[vol_patch_loc[a][0].start:vol_patch_loc[a][0].stop,
                vol_patch_loc[a][1].start:vol_patch_loc[a][1].stop,vol_patch_loc[a][2].start:vol_patch_loc[a][2].stop,:] += pred_temp[0,:,:,:,:]
                mask2[vol_patch_loc[a][0].start:vol_patch_loc[a][0].stop,
                vol_patch_loc[a][1].start:vol_patch_loc[a][1].stop,vol_patch_loc[a][2].start:vol_patch_loc[a][2].stop,:] += np.ones(pred_temp.shape[1:]).astype('float32')
               
            total_ddf_j=mask1/(mask2 +np.finfo(float).eps) 

            if postname=='npz':
               testlabel = np.load(validation_labels_data[j])['vol_data']
               atlas_seg = np.load(atlasseg_data[0])['vol_data']
            else:
               testlabel = nib.load(validation_labels_data[j]).get_fdata()
               atlas_seg = nib.load(atlasseg_data[0]).get_fdata()
             
            testlabel = np.reshape(testlabel, testlabel.shape + (1,))
            
            flow = np.reshape(total_ddf_j, (1,)+ total_ddf_j.shape)            
            
            warp_seg=app.warp_volumes_by_ddf_AD(testlabel, flow)

            moving= nib.load(validation_vols_data[j]).get_fdata() 
            moving = np.reshape(moving, moving.shape + (1,)).astype('float32')
            warp_mov = app.warp_volumes_by_ddf_(moving,flow)  
                    
            vals,_ = dice(warp_seg[0, :, :, :, 0], atlas_seg, labels, nargout=2)
            print("vals = ",vals)   ##CSF,GM,WM
            print("\n") 

            vals_new = np.delete(vals,[0])
            print("vals_new = ",vals_new)   ##CSF,GM,WM
            print("\n")
            dice_mean = np.mean(vals_new)
            dice_std = np.std(vals_new)

            print('Dice mean over structures: {:.16f} ({:.16f})\n'.format(dice_mean, dice_std))
         
            dice_scores_mean.append(np.mean(vals_new))
            dice_scores_std.append(np.std(vals_new))
            
            gc.collect()
            print('total mean dice', np.mean(dice_scores_mean), np.std(dice_scores_mean),j,i)
            del validation_vols,vol_patch_loc, atlas_vols, atlas_patch_loc,mask1,mask2,vals

            printTotalDice(np.mean(dice_scores_mean), np.std(dice_scores_mean),i)
  
    
if __name__ == "__main__":
        get_models = glob.glob(model_dir + '*.h5')             
        test(get_models,1000,0)
