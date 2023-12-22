#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import data2vec_model
import mae_model
import simclr_model
import os

def SimCLR_ISPL_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size,input_shape):
    pretrain_callbacks = []
    encoder = simclr_model.ispl_inception_encoder(input_shape)
    transform_funcs = []
    if(hparams[HP_2] == 'Rotation'):
        transform_funcs.append(simclr_model.rotation_transform_vectorized)
    elif(hparams[HP_2] == 'Noise'):
        transform_funcs.append(simclr_model.noise_transform_vectorized)
    elif(hparams[HP_2] == 'Scaled'):
        transform_funcs.append(simclr_model.scaling_transform_vectorized)
    projection_heads = simclr_model.projection_head(encoder.output.shape[1])
    transformations = simclr_model.generate_composite_transform_function_simple(transform_funcs)
    pretrain_pipeline = simclr_model.SimCLR(encoder,
                                            projection_heads,
                                            transformations)
    SSL_loss = simclr_model.NT_Xent_loss(temperature = hparams[HP_1])

    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        pretrain_pipeline.save_weights(initWeightDir_pretrain)
    else:
        pretrain_pipeline.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")



    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = hparams[HP_0], 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss


def SimCLR_HART_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size,input_shape):
    pretrain_callbacks = []
    encoder = simclr_model.HART_encoder(input_shape)
    transform_funcs = []
    if(hparams[HP_2] == 'Rotation'):
        transform_funcs.append(simclr_model.rotation_transform_vectorized)
    elif(hparams[HP_2] == 'Noise'):
        transform_funcs.append(simclr_model.noise_transform_vectorized)
    elif(hparams[HP_2] == 'Scaled'):
        transform_funcs.append(simclr_model.scaling_transform_vectorized)
    projection_heads = simclr_model.projection_head(encoder.output.shape[1])
    transformations = simclr_model.generate_composite_transform_function_simple(transform_funcs)
    pretrain_pipeline = simclr_model.SimCLR(encoder,
                                            projection_heads,
                                            transformations)
    SSL_loss = simclr_model.NT_Xent_loss(temperature = hparams[HP_1])

    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        pretrain_pipeline.save_weights(initWeightDir_pretrain)
    else:
        pretrain_pipeline.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = hparams[HP_0], 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss


def MAE_ISPL_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size,input_shape,patch_count):
    pretrain_callbacks = []

    decoderDepthLength = [[3, 7], [3, 7, 7, 15],[3, 7, 15, 31, 31, 31]]
    enc_embedding_size = 256

    patch_layer = mae_model.PatchLayer(frame_length,frame_length)
    patch_encoder = mae_model.PatchEncoder(frame_length,enc_embedding_size,True,hparams[HP_0])    
    mae_encoder = mae_model.ispl_inception_encoder(enc_embedding_size)
    mae_decoder = mae_model.ispl_inception_decoder(enc_embedding_size,
                                                   patch_count = patch_count,
                                                   filters_number = hparams[HP_2],
                                                   network_depth = hparams[HP_1],
                                                   output_shape = input_shape)
    
    pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
                            patch_encoder,
                            mae_encoder,
                            mae_decoder)

    
    SSL_loss = tf.keras.losses.MeanSquaredError()   



    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        mae_encoder.save_weights(initWeightDir_pretrain)
    else:
        mae_encoder.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")


    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss

def MAE_HART_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size,input_shape,patch_count):
    pretrain_callbacks = []

    decoderDepthLength = [[3, 7], [3, 7, 7, 15],[3, 7, 15, 31, 31, 31]]


    enc_embedding_size = 192
    patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
    patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,hparams[HP_0])    
    mae_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
                                         num_heads = 3,
                                         filterAttentionHead = 4, 
                                         convKernels = [3, 7, 15, 31, 31, 31])

    mae_decoder = mae_model.HART_decoder(enc_embedding_size = enc_embedding_size,
                                         projection_dim = hparams[HP_2],
                                         patch_count = patch_count,
                                         num_heads = 3,
                                         filterAttentionHead = 4, 
                                         convKernels = decoderDepthLength[(hparams[HP_1] // 2)-1])

    
    tf.print((hparams[HP_1] // 2)-1)

    pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
                            patch_encoder,
                            mae_encoder,
                            mae_decoder)

    
    SSL_loss = tf.keras.losses.MeanSquaredError()   



    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        mae_encoder.save_weights(initWeightDir_pretrain)
    else:
        mae_encoder.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")


    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss


def data2vec_ISPL_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size):
    pretrain_callbacks = []


    enc_embedding_size = 256
    teacherEncoder = data2vec_model.ispl_inception_teacher_encoder(enc_embedding_size)
    studentEncoder = data2vec_model.ispl_inception_encoder(enc_embedding_size)
    
    sensorWiseFramer = data2vec_model.FrameLayer(frame_length,frame_length)
    sensorWiseMaskEncoder = data2vec_model.MaskEncoder(enc_embedding_size,hparams[HP_0],frame_length)

    pretrain_pipeline =  data2vec_model.Data2Vec(sensorWiseFramer,
                                         sensorWiseMaskEncoder,
                                         teacherEncoder,
                                         studentEncoder)   
    SSL_loss = tf.keras.losses.Huber(delta = hparams[HP_2])
    pretrain_callbacks.append(data2vec_model.EMA(decay =hparams[HP_1]))
    for teacherLayers in teacherEncoder.layers:
        teacherLayers.trainable = False



    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        pretrain_pipeline.save_weights(initWeightDir_pretrain)
    else:
        pretrain_pipeline.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")


    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss



def data2vec_HART_train(hparams,frame_length,HP_0,HP_1,HP_2,initWeightDir_pretrain,instanceDir,SSL_LR,SSL_epochs,SSL_data,SSL_val_data,SSL_batch_size):
    pretrain_callbacks = []
    enc_embedding_size = 192
    teacherEncoder = data2vec_model.HART_teacher_encoder(projection_dim = enc_embedding_size, num_heads = 3,
                                            filterAttentionHead = 4, 
                                            convKernels = [3, 7, 15, 31, 31, 31],
                                            layerAverage = 3)
    studentEncoder = data2vec_model.HART_student_encoder(projection_dim = enc_embedding_size, num_heads = 3,
                                            filterAttentionHead = 4, 
                                            convKernels = [3, 7, 15, 31, 31, 31],)
    sensorWiseFramer = data2vec_model.SensorWiseFrameLayer(frame_length,frame_length)
    sensorWiseMaskEncoder = data2vec_model.SensorWiseMaskEncoder(enc_embedding_size,hparams[HP_0],frame_length)

    pretrain_pipeline =  data2vec_model.Data2Vec(sensorWiseFramer,
                                         sensorWiseMaskEncoder,
                                         teacherEncoder,
                                         studentEncoder)   
    SSL_loss = tf.keras.losses.Huber(delta = hparams[HP_2])
    pretrain_callbacks.append(data2vec_model.EMA(decay =hparams[HP_1]))
    for teacherLayers in teacherEncoder.layers:
        teacherLayers.trainable = False



    optimizer = tf.keras.optimizers.Adam(SSL_LR)
    
    pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])
    
    pretrain_pipeline.build(input_shape = (None,128,6))

    if(not os.path.exists(initWeightDir_pretrain)):
        print("Initialized model weights not found, generating one")
        pretrain_pipeline.save_weights(initWeightDir_pretrain)
    else:
        pretrain_pipeline.load_weights(initWeightDir_pretrain)
        print("Initialized model weights loaded")


    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        instanceDir+"bestValcheckpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    pretrain_callbacks.append(stop_early)
    pretrain_callbacks.append(checkpoint_callback)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    loss = np.min(history.history['val_loss'])
    return loss