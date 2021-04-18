from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from OurTransformers import EncoderLayer
import collections
import re

class BertEmbeddingLayer(layers.Layer):
    
    def get_angles(self,pos,i,d_model):
        super(BertEmbeddingLayer, self).__init__()
        angles = 1/np.power(10000.,(2*(i//2))/np.float32(d_model))
        return pos*angles
    
    def get_segment_embedding(self,segments,d_model):
        batch = []
        for seg in segments:
            sequence = []
            for i in seg:
                word = [i for _ in range(d_model)]
                sequence.append(word)
            batch.append(sequence)
        segment_Embedded = np.array(batch)
        return segment_Embedded
        
    def call(self,inputs,segments):
        seq_length = inputs.shape[-2]
        d_model = inputs.shape[-1]
        angles = self.get_angles(np.arange(seq_length)[:,np.newaxis],
                                 np.arange(d_model)[np.newaxis,:],
                                 d_model
                                )
        angles[:,0::2] = np.sin(angles[:,0::2])
        angles[:,1::2] = np.cos(angles[:,1::2])
        
        pos_encoding = angles[np.newaxis,...]
        
        #segment_embedding = self.get_segment_embedding(segments,inputs.shape[-1])
        
        return inputs + tf.cast(pos_encoding, tf.float32) #tf.cast(segment_embedding, tf.float32)

class MyBertModel(tf.keras.Model):
    def __init__(self,
                 nb_encoder_layers,
                 FFN_units,
                 nb_attention_head,
                 nb_hidden_units,
                 dropout_rate,
                 vocab_size,
                 name="BERT"):        
        super(MyBertModel, self).__init__(name=name)
        
        self.nb_encoder_layers = nb_encoder_layers
        self.nb_hidden_units = nb_hidden_units
        self.vocab_size = vocab_size
        
        self.embedding = layers.Embedding(vocab_size,nb_hidden_units,
                                          embeddings_initializer=initializers.RandomNormal(stddev=0.01,seed=3))
        self.segmentEmbedding = layers.Embedding(3,nb_hidden_units,
                                                 embeddings_initializer=initializers.RandomNormal(stddev=0.01,seed=3))
        
        self.bert_embedding = BertEmbeddingLayer()
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.enc_layers = [EncoderLayer(FFN_units,nb_attention_head,dropout_rate,activation=self.gelu_activation_function) 
                           for _ in range(nb_encoder_layers)]

        self.Dense_layer = layers.Dense(units=nb_hidden_units,activation=self.gelu_activation_function,
                                        kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                        bias_initializer=initializers.Zeros())
        self.Normalization = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.NSP_layer = layers.Dense(units=1,activation="sigmoid",name="NSP_ouput",
                                      kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                      bias_initializer=initializers.Zeros())
        self.MLM_layer = layers.Dense(units=vocab_size, activation="softmax",name="MLM_output",
                                      kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                      bias_initializer=initializers.Zeros())
        
    def gelu_activation_function(self,x):
        cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf
    
    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]
    
    def call(self,inputs,segments_order,training):
        enc_mask = self.create_padding_mask(inputs)
        #enc_mask = None
        
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.nb_hidden_units, tf.float32))
        
        outputs = self.bert_embedding(outputs,segments_order)
        outputs += self.segmentEmbedding(segments_order)
        
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_encoder_layers):
            outputs = self.enc_layers[i](outputs,enc_mask,training)
        
        outputs = self.Dense_layer(outputs)
        outputs = self.Normalization(outputs)
              
        return outputs
    
    def train_for_MLM(self,inputs,segments_order,training):#,mask_index,MaxSeq
        enc_mask = self.create_padding_mask(inputs)
        #enc_mask = None
        
        outputs = self.embedding(inputs)
        outputs = outputs*np.sqrt(self.nb_hidden_units)
        
        outputs = self.bert_embedding(outputs,segments_order)
        outputs += self.segmentEmbedding(segments_order)
        
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_encoder_layers):
            outputs = self.enc_layers[i](outputs,enc_mask,training)
        
        outputs = self.Dense_layer(outputs)
        outputs = self.Normalization(outputs)
        
        outputs = self.MLM_layer(outputs)
        #(batch_size,64,32000)
        
        return outputs
    
    def train_for_NSP(self,inputs,segments_order,training):
        enc_mask = self.create_padding_mask(inputs)
        #enc_mask = None
        
        outputs = self.embedding(inputs)
        outputs = outputs*np.sqrt(self.nb_hidden_units)
        
        outputs = self.bert_embedding(outputs,segments_order)
        outputs += self.segmentEmbedding(segments_order)
        
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_encoder_layers):
            outputs = self.enc_layers[i](outputs,enc_mask,training)
        
        outputs = self.Dense_layer(outputs)
        outputs = self.Normalization(outputs)

        
        outputs = self.NSP_layer(outputs)
        #(batch_size,70,1)
        
        
        NSP_predict = []
        for output in outputs:
            #(64,1)
            NSP_predict.append(output[0])
        
        
        return tf.cast(NSP_predict, tf.float32)
    
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    #init_vars = tf.train.list_variables(init_checkpoint)
    init_vars = tf.compat.v1.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)