#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy
import datetime
import numpy as np
import tensorflow as tf
import hickle as hkl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
import copy
import csv
import __main__ as main
import argparse
import pandas as pd
from tabulate import tabulate
import time
from tensorboard.plugins.hparams import api as hp

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


# In[2]:


# Library scripts
import utils 
import training
import data2vec_model
import mae_model
import simclr_model
import gridsearch_trainer


# In[3]:


experimentSetting = 'LODO'
# 'LOGO','LODO'
testingDataset = 'UCI'
# 'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
evaluationType = 'group'
# 'subject','group'

method = 'MAE'
# Data2vec, MAE, SimCLR

architecture = 'HART'
# ISPL,HART,HART_BASE

finetune_epoch = 50

finetune_batch_size = 64

SSL_batch_size = 128

loss = 'Adam'
# 'LARS', 'Adam', 'SGD'
SSL_LR = 3e-4

FT_LR = 3e-4

input_shape = (128,6)
frame_length = 16
SSL_epochs = 200
masking_ratio = 75e-2

instance_number = 20

GpuSpecifyIndex = 0


# In[4]:


gpus = tf.config.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GpuSpecifyIndex)
if gpus:
  try:
    tf.config.set_visible_devices(gpus[GpuSpecifyIndex], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GpuSpecifyIndex], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# In[5]:


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--experimentSetting', type=str, default=experimentSetting, 
        help='Leave one dataset out or Leave one group out')  
    parser.add_argument('--testingDataset', type=str, default=testingDataset, 
        help='Left out dataset')  
    parser.add_argument('--evaluationType', type=str, default=evaluationType, 
        help='Dataset group evaluation or subject by subject evaluation')  
    parser.add_argument('--SSL_epochs', type=int, default=SSL_epochs, 
        help='SSL Epochs')  
    parser.add_argument('--SSL_batch_size', type=int, default=SSL_batch_size, 
        help='SSL batch_size')  
    parser.add_argument('--finetune_epoch', type=int, default=finetune_epoch, 
        help='Fine_tune Epochs')  
    parser.add_argument('--loss', type=str, default=loss, 
        help='Specify the loss') 
    parser.add_argument('--SSL_LR', type=float, default=SSL_LR, 
        help='Specify the learning rate for the SSL techniques') 
    parser.add_argument('--masking_ratio', type=float, default=masking_ratio, 
        help='Specify the masking ratio') 
    parser.add_argument('--frame_length', type=int, default=frame_length, 
        help='Specify the masking ratio') 
    parser.add_argument('--architecture', type=str, default=architecture, 
        help='Specify the architecture of the model to train with') 
    parser.add_argument('--method', type=str, default=method, 
        help='Specify the SSL method') 
    parser.add_argument('--instance_number', type=int, default=instance_number, 
        help='Specify the SSL method') 
    args = parser.parse_args()
    return args
def is_interactive():
    return not hasattr(main, '__file__')


# In[6]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[7]:


# tf.keras.backend.set_floatx('float32')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# In[8]:


if(input_shape[0] % frame_length != 0 ):
    raise Exception("Invalid segment size")
else:
    patch_count = input_shape[0] // frame_length
print("Number of segments : "+str(patch_count))


# In[9]:


rootdir = './'
if not is_interactive():
    args = add_fit_args(argparse.ArgumentParser(description='SSL Pretraining Pipeline'))
    experimentSetting = args.experimentSetting
    testingDataset = args.testingDataset
    evaluationType = args.evaluationType
    SSL_epochs = args.SSL_epochs
    frame_length = args.frame_length
    SSL_batch_size = args.SSL_batch_size
    finetune_epoch = args.finetune_epoch
    loss = args.loss
    SSL_LR = args.SSL_LR
    masking_ratio = args.masking_ratio
    architecture = args.architecture
    method = args.method
    instance_number = args.instance_number


# In[10]:


hype1Index = instance_number % 3
hype2Index = (instance_number//3) % 3 
hype3Index = (instance_number//9) % 3  
print(str(hype3Index)+str(hype2Index)+str(hype1Index))
print("------------")


# In[11]:


dataDir = rootdir+'Datasets/SSL_PipelineUnionV2/'+experimentSetting+'/'
# projectName = str(architecture)+'_Data2Vec_LayerNorm_mask_'+str(masking_ratio)+'_frameLength_'+str(frame_length)+'_SSL_epochs_'+str(SSL_epochs)
projectName = str(method) +"_"+str (architecture) 
testMode = False
if(finetune_epoch < 10):
    testMode = True
    projectName= projectName + '/tests'
    
dataSetting = testingDataset

project_directory = rootdir+'results/'+projectName+'/'
initWeightDir_pretrain = project_directory+'ini_'+str(method)+'_'+str(architecture)+'_Pretraining_Weights.h5'

os.makedirs(project_directory, exist_ok=True)


# In[12]:


datasetList = ["HHAR","MobiAct","MotionSense","RealWorld_Waist","UCI","PAMAP"] 


# In[13]:


SSLdatasetList = copy.deepcopy(datasetList)
# SSLdatasetList.remove(testingDataset)
SSL_data = []
SSL_label = []

SSL_val_data = []
SSL_val_label = []

for datasetName in SSLdatasetList:
    SSL_data.append(hkl.load(dataDir + 'testData/'+str(datasetName)+'_data.hkl'))
    SSL_data.append(hkl.load(dataDir + 'fineTuneData/'+str(datasetName)+'_all_samples_data.hkl'))
    SSL_val_data.append(hkl.load(dataDir + 'valData/'+str(datasetName)+'_data.hkl'))

SSL_data = np.vstack((np.hstack((SSL_data))))
SSL_val_data = np.vstack((np.hstack((SSL_val_data))))

testData = hkl.load(dataDir + 'testData/'+testingDataset+'_data.hkl')
testLabel = hkl.load(dataDir + 'testData/'+testingDataset+'_label.hkl')

valData = hkl.load(dataDir + 'valData/'+testingDataset+'_data.hkl')
valLabel = hkl.load(dataDir + 'valData/'+testingDataset+'_label.hkl')

testData = np.vstack((testData))
testLabel = np.vstack((testLabel))
valData = np.vstack((valData))
valLabel = np.vstack((valLabel))


# In[14]:


# Here we are getting the labels presented only in the target dataset and calculating the suitable output shape.
ALL_ACTIVITY_LABEL = np.asarray(['Downstairs', 'Upstairs','Running','Sitting','Standing','Walking','Lying','Cycling','Nordic_Walking','Jumping'])
uniqueClassIDs = np.unique(np.argmax(testLabel,axis = -1))
ACTIVITY_LABEL = ALL_ACTIVITY_LABEL[uniqueClassIDs]
output_shape = len(ACTIVITY_LABEL)
METRIC_ACCURACY = 'loss'


# In[15]:


gridLogsDir = project_directory +'logs/'
os.makedirs(gridLogsDir+'hparam_tuning', exist_ok=True)


# In[16]:


if(method == 'Data2vec'):
    HP_0 = hp.HParam('Mask_Ratio', hp.Discrete([0.50,0.60, 0.75]))
    HP_1 = hp.HParam('Tau', hp.Discrete([0.998,0.9998, 0.9999]))
    HP_2 = hp.HParam('Beta', hp.Discrete([0.5,1.0,2.0]))
elif(method == 'MAE'):
    if(architecture == 'HART'):
        HP_0 = hp.HParam('Mask_Ratio', hp.Discrete([0.50,0.60, 0.75]))
        HP_1 = hp.HParam('Decoder_Depth',  hp.Discrete([2,4,6]))
        HP_2 = hp.HParam('Decoder_Width', hp.Discrete([128,192, 256]))
    else:
        HP_0 = hp.HParam('Mask_Ratio', hp.Discrete([0.50,0.60, 0.75]))
        HP_1 = hp.HParam('Decoder_Depth',  hp.Discrete([2,3,4]))
        HP_2 = hp.HParam('Decoder_Filter_Count', hp.Discrete([64,128, 192]))
elif(method == 'SimCLR'):
    HP_0= hp.HParam('Batch_Size', hp.Discrete([128,256, 512]))
    HP_1= hp.HParam('Temperature', hp.Discrete([0.1,0.5, 1.0]))
    HP_2 = hp.HParam('Transformations', hp.Discrete(['Rotation', 'Noise','Scaled']))

hpramsList = [HP_0,HP_1,HP_2]
with tf.summary.create_file_writer(gridLogsDir+'hparam_tuning').as_default():
  hp.hparams_config(
    hparams= hpramsList,
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Loss')],)


# In[17]:


instanceDir = project_directory+'GridSearch/'+str(HP_0.name)+'_'+str(HP_0.domain.values[hype1Index])+'_'+str(HP_1.name)+'_'+str(HP_1.domain.values[hype2Index])+'_'+str(HP_2.name)+'_'+str(HP_2.domain.values[hype3Index])+'/'
os.makedirs(instanceDir, exist_ok=True)


# In[18]:


def run(run_dir, hparams, method):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial

    if(method == 'Data2vec'):
        if(architecture == 'HART'):
            accuracy = gridsearch_trainer.data2vec_HART_train(hparams,
                                           frame_length = frame_length,
                                           HP_0 = HP_0,
                                           HP_1 = HP_1,
                                           HP_2 = HP_2,
                                           initWeightDir_pretrain = initWeightDir_pretrain,
                                           instanceDir = instanceDir,
                                           SSL_LR = SSL_LR,
                                           SSL_epochs = SSL_epochs,
                                           SSL_data = SSL_data,
                                           SSL_val_data = SSL_val_data,
                                           SSL_batch_size = SSL_batch_size)
        else:
            accuracy = gridsearch_trainer.data2vec_ISPL_train(hparams,
                                           frame_length = frame_length,
                                           HP_0 = HP_0,
                                           HP_1 = HP_1,
                                           HP_2 = HP_2,
                                           initWeightDir_pretrain = initWeightDir_pretrain,
                                           instanceDir = instanceDir,
                                           SSL_LR = SSL_LR,
                                           SSL_epochs = SSL_epochs,
                                           SSL_data = SSL_data,
                                           SSL_val_data = SSL_val_data,
                                           SSL_batch_size = SSL_batch_size)
    elif(method == 'MAE'):
        if(architecture == 'HART'):
            accuracy = gridsearch_trainer.MAE_HART_train(hparams,
                               frame_length = frame_length,
                               HP_0 = HP_0,
                               HP_1 = HP_1,
                               HP_2 = HP_2,
                               initWeightDir_pretrain = initWeightDir_pretrain,
                               instanceDir = instanceDir,
                               SSL_LR = SSL_LR,
                               SSL_epochs = SSL_epochs,
                               SSL_data = SSL_data,
                               SSL_val_data = SSL_val_data,
                               SSL_batch_size = SSL_batch_size,
                               input_shape = input_shape,
                               patch_count = patch_count)
        else:
            accuracy = gridsearch_trainer.MAE_ISPL_train(hparams,
                   frame_length = frame_length,
                   HP_0 = HP_0,
                   HP_1 = HP_1,
                   HP_2 = HP_2,
                   initWeightDir_pretrain = initWeightDir_pretrain,
                   instanceDir = instanceDir,
                   SSL_LR = SSL_LR,
                   SSL_epochs = SSL_epochs,
                   SSL_data = SSL_data,
                   SSL_val_data = SSL_val_data,
                   SSL_batch_size = SSL_batch_size,
                   input_shape = input_shape,
                   patch_count = patch_count)
    elif(method == 'SimCLR'):
        if(architecture == 'HART'):
            accuracy = gridsearch_trainer.SimCLR_HART_train(hparams,
                   frame_length = frame_length,
                   HP_0 = HP_0,
                   HP_1 = HP_1,
                   HP_2 = HP_2,
                   initWeightDir_pretrain = initWeightDir_pretrain,
                   instanceDir = instanceDir,
                   SSL_LR = SSL_LR,
                   SSL_epochs = SSL_epochs,
                   SSL_data = SSL_data,
                   SSL_val_data = SSL_val_data,
                   SSL_batch_size = SSL_batch_size,
                   input_shape = input_shape,)

        else:
            accuracy = gridsearch_trainer.SimCLR_ISPL_train(hparams,
                   frame_length = frame_length,
                   HP_0 = HP_0,
                   HP_1 = HP_1,
                   HP_2 = HP_2,
                   initWeightDir_pretrain = initWeightDir_pretrain,
                   instanceDir = instanceDir,
                   SSL_LR = SSL_LR,
                   SSL_epochs = SSL_epochs,
                   SSL_data = SSL_data,
                   SSL_val_data = SSL_val_data,
                   SSL_batch_size = SSL_batch_size,
                   input_shape = input_shape,)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    return accuracy


# In[19]:


# session_num = hype1Index + (hype2Index*3) + (hype3Index * 9)


# In[20]:


hparams = {
  HP_0: HP_0.domain.values[hype1Index],
  HP_1: HP_1.domain.values[hype2Index],
  HP_2: HP_2.domain.values[hype3Index],
}
run_name = "run-%d" % instance_number
print('--- Starting trial: %s' % run_name)
print({h.name: hparams[h] for h in hparams})
bestVal_Loss = run(gridLogsDir+'hparam_tuning/' + run_name, hparams, method = method)
# session_num += 1


# In[21]:


modelStatistics = {
    "val_loss:": str(bestVal_Loss),
}
with open(instanceDir +'valLoss.csv','w') as f:
    w = csv.writer(f)
    w.writerows(modelStatistics.items())


# In[ ]:




