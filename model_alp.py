#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

randomSeed = 1
# tf.random.set_seed(randomSeed)

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config

class GatedLinearUnit(layers.Layer):
    def __init__(self,units,**kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.linear = layers.Dense(units * 2)
        self.sigmoid = tf.keras.activations.sigmoid
    def call(self, inputs):
        linearProjection = self.linear(inputs)
        softMaxProjection = self.sigmoid(linearProjection[:,:,self.units:])
        return tf.multiply(linearProjection[:,:,:self.units],softMaxProjection)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config

class ClassToken(layers.Layer):
    def __init__(self, hidden_size,**kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.hidden_size = hidden_size
        self.cls = tf.Variable(
            name="cls",
            initial_value=self.cls_init(shape=(1, 1, self.hidden_size), seed=randomSeed, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,})
        return config

class Prompts(layers.Layer):
    def __init__(self, projectionDims,promptCount = 1,**kwargs):
        super(Prompts, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.projectionDims = projectionDims
        self.promptCount = promptCount
        self.prompts = [tf.Variable(
            name="prompt"+str(_),
            initial_value=self.cls_init(shape=(1, 1, self.projectionDims), seed=randomSeed, dtype="float32"),
            trainable=True,
        )  for _ in range(promptCount)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        prompt_broadcasted = tf.concat([tf.cast(tf.broadcast_to(promptInits, [batch_size, 1, self.projectionDims]),dtype=inputs.dtype,)for promptInits in self.prompts],1)
        return tf.concat([inputs,prompt_broadcasted], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionDims': self.projectionDims,
            'promptCount': self.promptCount,})
        return config
    
class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = dropout_rate )
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)
    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config
def softDepthConv(inputs):
    kernel = inputs[0]
    inputData = inputs[1]
    convOutputs = tf.nn.conv1d(
    inputData,
    kernel,
    stride = 1,
    padding = 'SAME',
    data_format='NCW',)
    return convOutputs


class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead 
        self.dropPathRate = dropPathRate
        self.dropout_rate = dropout_rate
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
    def build(self, input_shape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,), 
                initializer="glorot_uniform", 
                trainable=True,  
                name="biasWeights",
                dtype="float32"
            )
        
    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
#         print(inputs.shape)
        inputShape = tf.shape(formattedInputs)
#         reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,self.projectionSize))
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = [tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ]
        convOutputs = tf.convert_to_tensor(convOutputs)
        convOutputs = self.DropPathLayer(convOutputs)

        shape = tf.shape(formattedInputs)
        localAttention = tf.reshape(convOutputs,shape)
        return localAttention
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'patchCount': self.patchCount,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'dropPathRate': self.dropPathRate,
            'dropout_rate': self.dropout_rate,
            'attentionHead': self.attentionHead,})
        return config          


class mixAccGyro(layers.Layer):
    def __init__(self,projectionQuarter,projectionHalf,projection_dim,**kwargs):
        super(mixAccGyro, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.projectionHalf = projectionHalf
        self.projection_dim = projection_dim
        self.projectionThreeFourth = self.projectionHalf+self.projectionQuarter
        self.mixedAccGyroIndex = tf.reshape(tf.transpose(tf.stack(
            [np.arange(projectionQuarter,projectionHalf), np.arange(projectionHalf,projectionHalf + projectionQuarter)])),[-1])
        self.newArrangement = tf.concat((np.arange(0,projectionQuarter),self.mixedAccGyroIndex,np.arange(self.projectionThreeFourth,projection_dim)),axis = 0)
    def call(self, inputs):
        return tf.gather(inputs,self.newArrangement,axis= 2)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'projectionHalf': self.projectionHalf,
            'projection_dim': self.projection_dim,
        })
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def mlp2(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    return x

def depthMLP(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0])(x)
    x = layers.DepthwiseConv1D(3,data_format='channels_first',activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class SensorPatchesTimeDistributed(layers.Layer):
    def __init__(self, projection_dim,filterCount,patchCount,frameSize = 128, channelsCount = 6,**kwargs):
        super(SensorPatchesTimeDistributed, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.frameSize = frameSize
        self.channelsCount = channelsCount
        self.patchCount = patchCount
        self.filterCount = filterCount
        self.reshapeInputs = layers.Reshape((patchCount, frameSize // patchCount, channelsCount))
        self.kernelSize = (projection_dim//2 + filterCount) // filterCount
        self.accProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.gyroProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.flattenTime = layers.TimeDistributed(layers.Flatten())
        assert (projection_dim//2 + filterCount) / filterCount % self.kernelSize == 0
        print("Kernel Size is "+str((projection_dim//2 + filterCount) / filterCount))
#         assert 
    def call(self, inputData):
        inputData = self.reshapeInputs(inputData)
        accProjections = self.flattenTime(self.accProjection(inputData[:,:,:,:3]))
        gyroProjections = self.flattenTime(self.gyroProjection(inputData[:,:,:,3:]))
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection_dim': self.projection_dim,
            'filterCount': self.filterCount,
            'patchCount': self.patchCount,
            'frameSize': self.frameSize,
            'channelsCount': self.channelsCount,})
        return config
    
class SensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:])
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config


class threeSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(threeSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim//3),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjections = self.gyroProjection(inputData[:,:,6:])

        Projections = tf.concat((accProjections,gyroProjections,magProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

        
class fourSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(fourSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.altProjection = layers.Conv1D(filters = int(projection_dim/4),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")

    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:6])
        magProjection = self.gyroProjection(inputData[:,:,6:9])
        altProjection = self.gyroProjection(inputData[:,:,9:])

        Projections = tf.concat((accProjections,gyroProjections,magProjection,altProjection),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

def hartModel(input_shape,activityCount, projection_dim,patchSize,timeStep,num_heads,filterAttentionHead, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = True):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input(shape=input_shape)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    if(useTokens):
        patches = ClassToken(projection_dim)(patches)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(patchCount,
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    if(useTokens):
        representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(representation)
    else:
        representation = layers.GlobalAveragePooling1D()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    logits = layers.Dense(activityCount,  activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

class AdaptiveLayer(layers.Layer):
    def __init__(self ,layer, projection_dim,generateMemories = True,memoryBankSize = 128, memorySlot = 1, std = 2.0 ,decay = 0.96,**kwargs):
        super(AdaptiveLayer, self).__init__(**kwargs)  
        self.memorySlot = memorySlot
        self.memoryBankSize = memoryBankSize
        self.initialized = False
        self.decay = decay
        self.coldStart = False
        self.layer = layer
        self.std = std
        self.maxMemoryLimit = memoryBankSize * 2
        self.generateMemories = generateMemories
        self.projection_dim = projection_dim

    def sinkhorn_matrix(self,out):
        Q = tf.exp(out / 0.05, name='exp_out')  # Q is K-by-B for consistency with notations from the paper
        Qshape =  tf.cast(tf.shape(Q), tf.float32)
        B = tf.expand_dims(Qshape[1], 0) # number of samples to assign
        K = tf.expand_dims(Qshape[0], 0)  # how many prototypes
        sum_Q = tf.reduce_sum(Q)
        # sum_Q = tf.distribute.get_replica_context().all_reduce('sum', sum_Q)  # Distributed all_reduce
        Q /= sum_Q
    
        for it in range(3):
            sum_of_rows = tf.reduce_sum(Q, axis=1, keepdims=True)
            # sum_of_rows = tf.distribute.get_replica_context().all_reduce('sum', sum_of_rows)  # Distributed all_reduce
            Q /= sum_of_rows
            Q /= K
    
            # normalize each column: total weight per sample must be 1/B
            Q /= tf.reduce_sum(Q, axis=0, keepdims=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q
    
    def build(self):  # Create the state of the layer (weights)
        self.workingMemoryCount = tf.Variable(self.memoryBankSize)
        self.memoryPlaceHolder = tf.Variable(tf.random.normal((self.maxMemoryLimit,self.projection_dim)), name = "mem_weights_"+str(self.layer))
        
    def call(self, projections ,training=None):
        if(self.coldStart):
            inputShape = tf.shape(projections)
            reshapedProjections = tf.reshape(projections,(-1,inputShape[2] ))
            projOutput = tf.math.l2_normalize(reshapedProjections, axis=1)
            workingMemory = self.memoryPlaceHolder[:self.workingMemoryCount,:]
            memoryBankNorm = tf.math.l2_normalize(workingMemory, axis=1)
            matmulOutput = tf.linalg.matmul(memoryBankNorm,projOutput,transpose_b = True)
            feature_assignments = self.sinkhorn_matrix(matmulOutput)
            tFeat = tf.transpose(feature_assignments)
            _,index = tf.nn.top_k(tFeat,1, sorted=False)
            formattedIndex = tf.reshape(index,(-1,inputShape[1]))
            memoryAssignments = tf.gather(workingMemory,formattedIndex)
            output = tf.math.reduce_mean([projections,memoryAssignments], axis = 0)
            if(training):
                feature_weights,indices =  tf.math.top_k(feature_assignments, self.memorySlot, sorted=False)
                normFeatureWeights = tf.math.divide(feature_weights, tf.reduce_sum(feature_weights, axis=1, keepdims=True))
                weightedFeatures = tf.gather(reshapedProjections,indices) * tf.expand_dims(normFeatureWeights, -1)
                aggregatedFeatures = tf.reduce_sum(weightedFeatures,1)   
        
                EMA_Weights = ((self.decay * workingMemory) + ((1 - self.decay) * aggregatedFeatures))
                featIndex = tf.math.argmax(tFeat, axis = -1)
                featureAssignments = tf.gather(tFeat,featIndex, batch_dims = 1)
                if(self.generateMemories):
                    threshold = tf.math.reduce_mean(tFeat, axis =1 ) + self.std * (tf.math.reduce_std(tFeat, axis =1 ))
                    below_threshold_mask = featureAssignments < threshold
                    indices_of_values_below_threshold = tf.where(below_threshold_mask)[:,0]
                    newMemories = tf.gather(reshapedProjections,indices_of_values_below_threshold)
                    newMemoriesShape = tf.shape(newMemories)       
                    endIndex = newMemoriesShape[0] + self.workingMemoryCount
                    if(newMemoriesShape[0] > 0 and endIndex <= self.maxMemoryLimit  ):
                        concat = tf.stack([EMA_Weights, newMemories], axis=0)
                        self.memoryPlaceHolder[:self.workingMemoryCount,:].assign(EMA_Weights)
                        self.memoryPlaceHolder[self.workingMemoryCount:endIndex,:].assign(newMemories)
                        self.workingMemoryCount.assign(endIndex)
                    else:
                        self.memoryPlaceHolder[:self.workingMemoryCount,:].assign(EMA_Weights)
                else:
                    self.memoryPlaceHolder[:self.workingMemoryCount,:].assign(EMA_Weights)
            return output
        else:
            return projections
        

def hartModel(useTokens = True):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
#     patches = SensorPatchesTimeDistributed(projection_dim,12,len(segmentTime))(inputs)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    if(useTokens):
        patches = ClassToken(projection_dim)(patches)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        x2 = MemNodeV3(layer = layerIndex, 
               generateMemories = generateMemories, 
               memoryBankSize = memoryCounts, 
               memorySlot = memorySlotSize, 
               decay = decayRate)(x1) 


        branch1 = liteFormer(patchCount,
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x2)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x2)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x2)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )

        x3 = layers.Add()([concatAttention, encoded_patches])
        x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
        x5 = mlp2(x4, hidden_units=transformer_units, dropout_rate=dropout_rate)
        encoded_patches = layers.Add()([x5, x4])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    if(useTokens):
        representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(representation)
    else:
        representation = layers.GlobalAveragePooling1D()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    # Classify outputs.
    logits = layers.Dense(activityCount,  activation='softmax')(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model





class MemNodeV4_GLU(layers.Layer):
    def __init__(self ,layer,projection_dim, globalPrototype = False, memoryBankSize = 128, memorySlot = 3, decay = 0.96,**kwargs):
        super(MemNodeV4_GLU, self).__init__(**kwargs)  
        self.memorySlot = memorySlot
        self.decay = decay
        self.memoryPlaceHolder = tf.Variable(tf.random.normal((memoryBankSize,projection_dim)), 
                                             name = "mem_weights_"+str(layer),
                                             trainable= False)

        if(globalPrototype):
            self.globalProts = tf.Variable(tf.random.normal((memoryBankSize,projection_dim)), 
                                         name = "global_weights_"+str(layer),
                                         trainable= False)
        self.coldStart = False
        self.projection_dim = projection_dim
        # self.GLU = GatedLinearUnit(projection_dim)


    def build(self, input_shape):  # Create the state of the layer (weights)
        self.GLU = GatedLinearUnit(self.projection_dim)
        
    @tf.function
    def sinkhorn_matrix(self,out):
        Q = tf.exp(out / 0.05, name='exp_out')  # Q is K-by-B for consistency with notations from the paper
        Qshape =  tf.cast(tf.shape(Q), tf.float32)
        B = tf.expand_dims(Qshape[1], 0) # number of samples to assign
        K = tf.expand_dims(Qshape[0], 0)  # how many prototypes

        sum_Q = tf.reduce_sum(Q)
        # sum_Q = tf.distribute.get_replica_context().all_reduce('sum', sum_Q)  # Distributed all_reduce
        Q /= sum_Q
    
        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = tf.reduce_sum(Q, axis=1, keepdims=True)
            # sum_of_rows = tf.distribute.get_replica_context().all_reduce('sum', sum_of_rows)  # Distributed all_reduce
            Q /= sum_of_rows
            Q /= K
    
            # normalize each column: total weight per sample must be 1/B
            Q /= tf.reduce_sum(Q, axis=0, keepdims=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q
        
    def call(self, projections,training=None):
        inputShape = tf.shape(projections)
        reshapedProjections = tf.reshape(projections,(-1,inputShape[2] ))
        projOutput = tf.math.l2_normalize(reshapedProjections, axis=1)
        memoryBankNorm = tf.math.l2_normalize(self.memoryPlaceHolder, axis=1)
        matmulOutput = tf.linalg.matmul(memoryBankNorm,projOutput,transpose_b = True)
        feature_assignments = self.sinkhorn_matrix(matmulOutput)
        tFeat = tf.transpose(feature_assignments)
        _,index = tf.nn.top_k(tFeat,1, sorted=False)
        memoryAssignments = tf.gather(self.memoryPlaceHolder,tf.squeeze(index,axis = 1))
        gatedAssignments = self.GLU(memoryAssignments)
        formattedAssignments = tf.reshape(gatedAssignments,(-1,inputShape[1],inputShape[2]))
        output = tf.math.reduce_mean([projections,formattedAssignments], axis = 0)
        if(training):
            feature_weights,indices =  tf.math.top_k(feature_assignments, self.memorySlot , sorted=False)
            normFeatureWeights = tf.math.divide(feature_weights, tf.reduce_sum(feature_weights, axis=1, keepdims=True))
            weightedFeatures = tf.gather(reshapedProjections,indices) * tf.expand_dims(normFeatureWeights, -1)
            aggregatedFeatures = tf.reduce_sum(weightedFeatures,1)    
            EMA_Weights = ((self.decay * self.memoryPlaceHolder) + ((1 - self.decay) * aggregatedFeatures))
            self.memoryPlaceHolder.assign(EMA_Weights)
        if(self.coldStart):
            return output
        else:
            return projections
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memorySlot': self.memorySlot,
            'memoryBankSize': self.memoryBankSize,})
        return config



class GatedLinearUnit(layers.Layer):
    def __init__(self,units,**kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.sigmoid = tf.keras.activations.sigmoid

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.linear = layers.Dense(self.units * 2)

    def call(self, inputs):
        linearProjection = self.linear(inputs)
        softMaxProjection = self.sigmoid(linearProjection[:,self.units:])
        return tf.multiply(linearProjection[:,:self.units],softMaxProjection)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,})
        return config


def HART_ALP_encoder(projection_dim = 192,globalPrototype = False,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1,memoryBankSize = 1024,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs

    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        x2 = MemNodeV4_GLU(layer = layerIndex, 
               globalPrototype = globalPrototype,
               projection_dim = projection_dim,
               memoryBankSize = memoryBankSize, 
               memorySlot = 3, 
               decay = 0.96)(x1) 
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x2)
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x2)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x2)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, representation, name="mae_encoder")    
