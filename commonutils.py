# -*- coding: utf-8 -*-

###deal the csv and png
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import csv
import sys
import tensorflow as tf
import nibabel as nib
class Config(object):
    base_dir = '../data/OASIS3_100/'
    model_dir ='../models/'
    vol_size = (64,64,64)
    lr = 1e-4
    #nf_enc = [16,32,32,32]
    #nf_dec = [32,32,32,32,8,8,3]
    nf_enc = [8,16,16,16]
    nf_dec = [16,16,16,16,4,4,3]
    patch_size = [64,64,64]
    stride = 32
    number  = 1
    training_dir = base_dir +'train/'+'*.nii.gz'
    positive_dir = base_dir + 'positive/'
    register_dir = base_dir +'atlas/*.nii.gz'
    loss_dir = base_dir+'loss/loss.txt'
    tensorboard_dir = '../tensorboard/'
    
    log_dir = '../logs/'
    fileParams={'base_dir': base_dir,'loss_dir': loss_dir,'model_dir': model_dir,'train_dir': training_dir,
                'positive_dir':positive_dir, 'log_dir': log_dir,'register_dir': register_dir}
    
    trainModel={'vol_size': vol_size,'lr':lr,'nf_enc':nf_enc,'nf_dec': nf_dec}
    trainPatch={'patch_size': patch_size, 'stride': stride,'number': number}
    labelcsv = base_dir+'label.csv'
    testModel={}
    testPatch={}
    saveModel={}
    csvFile={}
    niiOperate={}
    imgFile={}
    
    
    
    
    @staticmethod
    def readFilesDir():
        pass
    @staticmethod
    def save(step, training, train_loss, loss_dir):
         s = str(step) + "," + str(training)
         if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
            for i in range(len(train_loss)):
               s += "," + str(train_loss[i])
         else:
            s += "," + str(train_loss)
         file_handle=open(loss_dir,mode='a')
         file_handle.write(s+'\n')
         file_handle.close()
         sys.stdout.flush()
    @staticmethod
    def saveFlow():
        pass
    @staticmethod
    def saveWrappedImage():
        pass
    @staticmethod
    def saveResultCSV():
        pass
    @staticmethod
    def createResultCSV():
        pass
    @staticmethod
    def draw_pic(path,name):
        pass
    @staticmethod
    def saveModel(path,name):
        pass
    @staticmethod
    def niitoNormalized():
        pass
    @staticmethod
    def printLoss(step,training, train_loss,savename):
         s = str(step) + "," + str(training)

         if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
            for i in range(len(train_loss)):
               s += "," + str(train_loss[i])
         else:
            s += "," + str(train_loss)
        
         file_handle=open(savename,mode='a')
         file_handle.write(s+'\n')
         file_handle.close()
         sys.stdout.flush()
         
    @staticmethod
    def named_logs(model, logs):
       result = {}
       for l in zip(model.metrics_names, logs):
          result[l[0]] = l[1]
       return result
    @staticmethod
    def write_log(callback, names, logs, batch_no):
      for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
    @staticmethod
    def savePositive(image,name):
       flow_new  = np.squeeze(np.array(image ))
       ref_affine = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
       flow_new_nii = nib.Nifti1Image(flow_new, ref_affine)
       nib.save(flow_new_nii, Config.base_dir + 'positive/'+ name)
    @staticmethod
    def read_labecsv(file):
       with open(file, 'r') as f:
         reader = csv.reader(f)
     
         lis = []
       
         for row in reader:
           for v in row:
             lis.append(int(v))    
         
     
      
       return lis
        