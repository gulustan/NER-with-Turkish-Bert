import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class PositionalEncoding(layers.Layer):
    
    def get_angles(self,pos,i,d_model):
        super(PositionalEncoding, self).__init__()
        angles = 1/np.power(10000.,(2*(i//2))/np.float32(d_model))
        return pos*angles
    
    def __call__(self,inputs):
        seq_length = inputs.shape[-2]
        d_model = inputs.shape[-1]
        angles = self.get_angles(np.arange(seq_length)[:,np.newaxis],
                                 np.arange(d_model)[np.newaxis,:],
                                 d_model
                                )
        angles[:,0::2] = np.sin(angles[:,0::2])
        angles[:,1::2] = np.cos(angles[:,1::2])
        pos_encoding = angles[np.newaxis,...]
        return inputs + tf.cast(pos_encoding, tf.float32)

def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)
    
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)
    
    if mask is not None:
        scaled_product += (mask * -1e9)
    
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    
    return attention

class MultiHeadAttention(layers.Layer):
    def __init__(self,nb_proj):
        super(MultiHeadAttention, self).__init__()
        self.nb_proj = nb_proj
        
    def build(self, input_shape):        
        self.d_model = input_shape[-1]
        
        assert self.d_model % self.nb_proj == 0
        
        self.d_proj = self.d_model // self.nb_proj
        
        self.query_lin = keras.layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                            bias_initializer=initializers.Zeros())
        self.key_lin = keras.layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                          bias_initializer=initializers.Zeros())
        self.value_lin = keras.layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                            bias_initializer=initializers.Zeros())
        
        self.final_lin = keras.layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                            bias_initializer=initializers.Zeros())
        
    def split2proj(self,inputs,batch_size): 
        # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,-1,self.nb_proj,self.d_proj)
        # (batch_size, seq_length, nb_proj, d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape) 
        
        # (batch_size, nb_proj, seq_length, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) 
        
    
    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]
        
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)
        
        queries = self.split2proj(queries, batch_size)
        keys = self.split2proj(keys, batch_size)
        values = self.split2proj(values, batch_size)
        
        attention = scaled_dot_product_attention(queries, keys, values, mask)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(attention,shape=(batch_size, -1, self.d_model))
        
        outputs = self.final_lin(concat_attention)
        
        return outputs
    
class EncoderLayer(layers.Layer):
    def __init__(self,FFN_units,nb_proj,dropout_rate,activation="relu"):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate

        self.activation = activation
        
    def build(self,input_shape):
        self.d_model = input_shape[-1]
        
        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = keras.layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dense_1 = keras.layers.Dense(units=self.FFN_units,activation=self.activation,
                                          kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                          bias_initializer=initializers.Zeros())
        self.dense_2 = keras.layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                          bias_initializer=initializers.Zeros())
        self.dropout_2 = keras.layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self,inputs,mask,training):
        attention = self.multi_head_attention(inputs,inputs,inputs,mask)
        attention = self.dropout_1(attention,training=training)
        attention = self.norm_1(inputs+attention)
        
        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        outputs = self.norm_2(outputs + attention)
        
        return outputs
    
class Encoder(layers.Layer):
    def __init__(self,nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder,self).__init__(name=name)
        self.nb_layers = nb_layers
        self.d_model = d_model
        
        self.embedding = layers.Embedding(vocab_size,d_model,
                                          embeddings_initializer=initializers.RandomNormal(stddev=0.01,seed=3))
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.enc_layers = [EncoderLayer(FFN_units,nb_proj,dropout_rate) 
                           for _ in range(nb_layers)]
        
    def call(self,inputs,mask,training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs,mask,training)
        
        return outputs   
    
class DecoderLayer(layers.Layer):
    def __init__(self,FFN_units,nb_proj,dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout_rate = dropout_rate
        
    def build(self,input_shape):
        self.d_model = input_shape[-1]
        
        # Self multi head attention
        self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi head attention combined with encoder output
        self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed foward
        self.dense_1 = layers.Dense(units=self.FFN_units,activation="relu",
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                    bias_initializer=initializers.Zeros())
        self.dense_2 = layers.Dense(units=self.d_model,kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                    bias_initializer=initializers.Zeros())
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self,inputs,enc_outputs,mask_1,mask_2,training):
        attention_1 = self.multi_head_attention_1(inputs,inputs,inputs,mask_1)
        attention_1 = self.dropout_1(attention_1, training)
        attention_1 = self.norm_1(attention_1 + inputs)
            
        attention_2 = self.multi_head_attention_1(attention_1,enc_outputs,enc_outputs,mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_1 + attention_2)
            
        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)
        
        return outputs    
    
class Decoder(layers.Layer):
    def __init__(self,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.nb_layers = nb_layers
        
        self.embedding = layers.Embedding(vocab_size,d_model,
                                          embeddings_initializer=initializers.RandomNormal(stddev=0.01,seed=3))
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.dec_layers = [DecoderLayer(FFN_units,nb_proj,dropout_rate) 
                           for _ in range(nb_layers)]
        
    def call(self,inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        
        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs,enc_outputs,mask_1,mask_2,training)
        
        return outputs    
    
class Transformer(tf.keras.Model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)
        
        self.encoder = Encoder(nb_layers,
                               FFN_units,
                               nb_proj,
                               dropout_rate,
                               vocab_size_enc,
                               d_model)
        
        self.decoder = Decoder(nb_layers,
                               FFN_units,
                               nb_proj,
                               dropout_rate,
                               vocab_size_dec,
                               d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec, activation="softmax",name="lin_ouput",
                                        kernel_initializer=initializers.RandomNormal(stddev=0.01,seed=3),
                                        bias_initializer=initializers.Zeros())
        
    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask
    
    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        
        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)
        
        outputs = self.last_linear(dec_outputs)
        
        return outputs    
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")

class Material():
    def loss_function(self,target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    def optimizer_function(self,learning_rate,beta_1,beta_2,epsilon):        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=beta_1,
                                             beta_2=beta_2,
                                             epsilon=epsilon)
        return optimizer    



