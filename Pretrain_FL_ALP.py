#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
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

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


# In[ ]:


# Library scripts
import utils 
import training
import data2vec_model
import mae_model
import simclr_model
import model_alp


# In[ ]:


experimentSetting = 'LODO'
# 'LOGO','LODO'
testingDataset = 'HHAR'
# 'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
evaluationType = 'group'
# 'subject','group'

method = 'MAE_ALP'
# Data2vec, MAE, SimCLR

architecture = 'HART'

finetune_epoch = 1

finetune_batch_size = 64

SSL_batch_size = 128

loss = 'Adam'
# 'LARS', 'Adam', 'SGD'

SSL_LR = 3e-4

FT_LR = 3e-4

input_shape = (128,6)

frame_length = 16

SSL_epochs = 300

masking_ratio = 75e-2

instance_number = 0

randomRuns = 5

warmUpEpoch = 50

memoryCount = 1024


# In[ ]:


datasets = ['HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP']


# In[ ]:


datasets = ['MobiAct','MotionSense','UCI','PAMAP','SHL']


# In[ ]:


architectures = ['HART','ISPL']


# In[ ]:


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
    parser.add_argument('--memoryCount', type=int, default=memoryCount, 
        help='Specify the SSL method') 
    args = parser.parse_args()
    return args
def is_interactive():
    return not hasattr(main, '__file__')


# In[ ]:


tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


if(input_shape[0] % frame_length != 0 ):
    raise Exception("Invalid segment size")
else:
    patch_count = input_shape[0] // frame_length
print("Number of segments : "+str(patch_count))


# In[ ]:


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
    memoryCount = args.memoryCount


# In[ ]:


dataDir = rootdir+'Datasets/SSL_PipelineUnionV2/'+experimentSetting+'/'
projectName = str(method) +"_"+str (architecture) + "_SSL_batch_size_" + str(SSL_batch_size) +'_memoryCount_'+str(memoryCount)
testMode = False
if(finetune_epoch < 10):
    testMode = True
    projectName= projectName + '/tests'
    
dataSetting = testingDataset

project_directory = rootdir+'results/'+projectName+'/'
working_directory = project_directory+dataSetting+'/'
pretrained_dir = working_directory + evaluationType + '/'
    
initWeightDir_pretrain = project_directory+'ini_'+str(method)+'_'+str(architecture)+'_Pretraining_Weights.h5'
val_checkpoint_pipeline_weights = working_directory+"best_val_"+str(method)+"_pretrain.h5"
trained_pipeline_weights = working_directory+"trained_"+str(method)+"_pretrain.h5"
random_FT_weights = working_directory+"ini_"+str(method)+"_HART_Classification_Weights.h5"    
trained_FT_weights = working_directory+"trained_"+str(method)+"_dowmstream.h5"
trained_FE_dir = working_directory+"trained_"+str(method)+"_feature_extractor.h5"
os.makedirs(pretrained_dir, exist_ok=True)


# In[ ]:


# datasetList = ["HHAR","MobiAct","MotionSense","RealWorld_Waist","UCI","PAMAP"] 


# In[ ]:


datasetList = ["SHL","MobiAct","MotionSense","UCI","PAMAP"] 


# In[ ]:


# datasetList = ["UCI"] 


# In[ ]:


SSLdatasetList = copy.deepcopy(datasetList)
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


testData = np.vstack((testData))
testLabel = np.vstack((testLabel))


# In[ ]:


# Here we are getting the labels presented only in the target dataset and calculating the suitable output shape.
ALL_ACTIVITY_LABEL = np.asarray(['Downstairs', 'Upstairs','Running','Sitting','Standing','Walking','Lying','Cycling','Nordic_Walking','Jumping'])


# In[ ]:


pretrain_callbacks = []


# In[ ]:


enc_embedding_size = 192
convKernels = [3, 7, 15, 31, 31, 31]
numberOfMemoryBlocks = len(convKernels)
patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,0.6)    
mae_encoder = model_alp.HART_ALP_encoder(enc_embedding_size,                                                     
                                     num_heads = 3,
                                     filterAttentionHead = 4, 
                                     memoryBankSize = 1024,
                                     convKernels = convKernels)
mae_decoder = mae_model.HART_decoder(enc_embedding_size = enc_embedding_size,
                                     projection_dim = 256,
                                     patch_count = patch_count,
                                     num_heads = 3,
                                     filterAttentionHead = 4, 
                                     convKernels = [3, 7, 15, 31, 31, 31])
pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
                        patch_encoder,
                        mae_encoder,
                        mae_decoder)
SSL_loss = tf.keras.losses.MeanSquaredError()


# In[ ]:


def getLayerIndexByName(model, layername):
    layerIndex = []
    for idx, layer in enumerate(model.layers):
        # print(layer.name)
        if layername in layer.name:
            layerIndex.append(idx)
    return layerIndex
        # if layer.name == layername:
            # return idx


# In[ ]:


layersID = getLayerIndexByName(mae_encoder,"mem_node")


# In[ ]:


optimizer = tf.keras.optimizers.Adam(SSL_LR)

pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])

# Forcing a build to the model 
pretrain_pipeline.build(input_shape = (None,128,6))

if(not os.path.exists(initWeightDir_pretrain)):
    print("Initialized model weights not found, generating one")
    pretrain_pipeline.save_weights(initWeightDir_pretrain)
else:
    pretrain_pipeline.load_weights(initWeightDir_pretrain)
    print("Initialized model weights loaded")


# In[ ]:


pretrained_FE = pretrain_pipeline.return_feature_extrator()
FE_Layers = len(pretrained_FE.layers) + 1


# In[ ]:


historyWarmUp = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = warmUpEpoch,
                                    verbose=2)


# In[ ]:


class trackMemoryStability(tf.keras.callbacks.Callback):
    def __init__(self,layersID,**kwargs):
        super(trackMemoryStability, self).__init__(**kwargs)  
        self.memoryStabilityEpoch = {i: [] for i in range(numberOfMemoryBlocks)}
        self.layersID = layersID
        self.previousMemory = np.empty(numberOfMemoryBlocks, dtype=object)  

    def on_epoch_begin(self, epoch, logs=None):
        if(epoch != 0):
            for index,layerID in enumerate(layersID):
                self.previousMemory[index] = tf.identity(self.model.encoder.layers[layerID].memoryPlaceHolder)
    def on_epoch_end(self, epoch, logs=None):
        if(epoch != 0):
            for index,layerID in enumerate(layersID):
                self.memoryStabilityEpoch[index].append(tf.reduce_mean(tf.math.abs(tf.identity(self.model.encoder.layers[layerID].memoryPlaceHolder) - self.previousMemory[index])))


memoryChangeTrack = trackMemoryStability(layersID)


# In[ ]:


for index,layerID in enumerate(layersID):
    mae_encoder.layers[layerID].coldStart = True
pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])


# In[ ]:


best_val_model_callback = tf.keras.callbacks.ModelCheckpoint(val_checkpoint_pipeline_weights,
monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=2)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
pretrain_callbacks.append(best_val_model_callback)
pretrain_callbacks.append(stop_early)
pretrain_callbacks.append(memoryChangeTrack)
historyAdapt = pretrain_pipeline.fit(SSL_data,
                                validation_data = (SSL_val_data,SSL_val_data), 
                                batch_size = SSL_batch_size, 
                                epochs = SSL_epochs,
                                callbacks=pretrain_callbacks,
                                verbose=2)


# In[ ]:


if(stop_early.stopped_epoch == 0):
    earlyStopEpoch = SSL_epochs 
else:
    earlyStopEpoch = stop_early.stopped_epoch + 1


# In[ ]:


memoryStabilityPath = pretrained_dir+"memoryImages/"
os.makedirs(memoryStabilityPath, exist_ok=True)
memoryStabilityEpoch = [memoryChangeTrack.memoryStabilityEpoch[key] for key in memoryChangeTrack.memoryStabilityEpoch]
for index, layerMemoryStability in enumerate(memoryStabilityEpoch):
    memoryStabilityEpoch = [memoryChangeTrack.memoryStabilityEpoch[key] for key in memoryChangeTrack.memoryStabilityEpoch]
    epoch_range = range(1, earlyStopEpoch)
    plt.plot(epoch_range, layerMemoryStability)
    plt.title('Prototype Displacements For Block '+str(index))
    plt.ylabel('L1 Distance')
    plt.xlabel('Epoch')
    plt.savefig(memoryStabilityPath+"B_"+str(index+1)+"_memoryDisplacement.png", bbox_inches="tight")
    plt.show()
    plt.clf()

# memoryStabilityEpoch = [memoryChangeTrack.memoryStabilityEpoch[key] for key in memoryChangeTrack.memoryStabilityEpoch]
epoch_range = range(1, earlyStopEpoch)
meanMemory = np.mean(memoryStabilityEpoch,axis = 0)
stdMemory = np.std(memoryStabilityEpoch,axis = 0)
plt.errorbar(epoch_range, meanMemory, yerr=stdMemory)
plt.title('Mean Prototype Displacements')
plt.ylabel('L1 Distance')
plt.xlabel('Epoch')
plt.savefig(memoryStabilityPath+"meanMemoryDisplacement.png", bbox_inches="tight")
plt.show()
plt.clf()


# In[ ]:


history = {}
history['loss'] = historyWarmUp.history['loss'] + historyAdapt.history['loss']
# Optionally, combine validation loss and metrics if available
if 'val_loss' in historyWarmUp.history and 'val_loss' in historyAdapt.history:
    history['val_loss'] = historyWarmUp.history['val_loss'] + historyAdapt.history['val_loss']


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(history['loss'], label = 'Train Loss')
plt.plot(history['val_loss'], label = 'Val Loss')
plt.plot(history['val_loss'],markevery=[np.argmin(history['val_loss'])], ls="", marker="o",color="orange")
plt.plot(history['loss'],markevery=[np.argmin(history['loss'])], ls="", marker="o",color="blue")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(working_directory+"lossCurve.png", bbox_inches="tight")
plt.show()


# In[ ]:


pretrain_pipeline.load_weights(val_checkpoint_pipeline_weights)
pretrain_pipeline.save_weights(trained_pipeline_weights)
pretrained_FE.save_weights(trained_FE_dir)
perplexity = 30.0
embeddings = pretrain_pipeline.predict(testData, batch_size=1024,verbose=0)
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=0, random_state=42)
tsne_projections = tsne_model.fit_transform(embeddings)
labels_argmax = np.argmax(testLabel, axis=1)
unique_labels = np.unique(labels_argmax)
utils.projectTSNE('TSNE_Embeds',pretrained_dir,ALL_ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
utils.projectTSNEWithShape('TSNE_Embeds_shape',pretrained_dir,ALL_ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
hkl.dump(tsne_projections,pretrained_dir+'tsne_projections.hkl')


# In[ ]:





# In[ ]:




