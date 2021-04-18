from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import OurBertModel 
import numpy as np
import re
import time


def create_optimizer(init_lr, num_train_steps, num_warmup_steps,global_step):
    
    #global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)


    learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate,
                                        global_step,
                                        num_train_steps,
                                        end_learning_rate=8e-8,
                                        power=1.0,
                                        cycle=False)
       
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        
        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
        
        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done
        
        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate.func(global_step-1) + is_warmup * warmup_learning_rate)
        
    else:
        learning_rate = learning_rate.func(global_step-1)
        

    optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate,
                                         weight_decay_rate=0.01,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=3e-7,
                                         exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    
    return optimizer
    
    

    #tvars = tf.compat.v1.trainable_variables()
    #grads = tf.compat.v1.gradients(loss, tvars)
    
    
    
    
    #return train_op

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class BertModelTrainer():
    def __init__(self,HIDDEN_UNITS):
        self.loss_object_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
        #self.loss_object_2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_object_2 = tf.keras.losses.BinaryCrossentropy()
        
        self.train_loss_MLM = tf.keras.metrics.Mean(name="train_loss")
        self.train_loss_NSP = tf.keras.metrics.Mean(name="train_loss")

        self.train_accuracy_MLM = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy_MLM")
        self.train_accuracy_NSP = tf.keras.metrics.BinaryAccuracy(name="train_accuracy_NSP")
        
        """
        leaning_rate = CustomSchedule(HIDDEN_UNITS)
        
        self.optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        """
        
        self.learning_rate = 3e-5
        self.num_train_steps = 2000
        self.num_warmup_steps = None#100
        self.global_step = 1
        
        self.optimizer = create_optimizer(self.learning_rate,
                                          self.num_train_steps,
                                          self.num_warmup_steps,
                                          self.global_step)
    
    def loss_function_for_MLM(self,target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_object_1(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def loss_function_for_NSP(self,target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_object_2(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    def CheckPoint_Model(self,BertModel,checkpoint_path,max_to_keep):
        self.ckpt = tf.train.Checkpoint(OUR_BERT=BertModel,optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=max_to_keep)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")
            return self.ckpt_manager.latest_checkpoint
        return None
        
    def __call__(self,
                 BertModel,
                 epochs,
                 inputs,
                 NSP_label,
                 mask_index,
                 mask_label,
                 segment,
                 checkpoint_path = "",
                 max2keep=0,
                 batch2Show=1):
        
        init_checkpoint = self.CheckPoint_Model(BertModel,checkpoint_path,max2keep)
        
        self.global_step = 1
        for epoch in range(epochs):
            print("Start of epoch {}".format(epoch+1))
            start = time.time()
            
            self.train_loss_MLM.reset_states()
            self.train_loss_NSP.reset_states()
            
            self.train_accuracy_MLM.reset_states()
            self.train_accuracy_NSP.reset_states()
            
            batchSegment = []
            batchRealSeq = []
            batchinput = []
            batchNSPLabel = []
            
            self.optimizer = create_optimizer(self.learning_rate,
                                              self.num_train_steps,
                                              self.num_warmup_steps,
                                              self.global_step)
            ##################################################################
            self.global_step = self.global_step+1
            #self.num_warmup_steps = self.num_warmup_steps-1
            ##################################################################
            for index,sentence in enumerate(inputs):
                mask_input = tf.convert_to_tensor(sentence)
                target_segment = tf.convert_to_tensor(segment[index])
                target_NSP = tf.constant(NSP_label[index],tf.float32)
                
                target_index = mask_index[index]
                target_label = mask_label[index]
                

                real_sentence = [s for s in sentence]
                for idx,idl in zip(target_index,target_label):
                    real_sentence[idx]=idl
                real_sentence = tf.convert_to_tensor(real_sentence)
                
                
                if index%batch2Show != 0 or index==0:
                    batchinput.append(mask_input)
                    batchSegment.append(target_segment)
                    batchRealSeq.append(real_sentence)
                    batchNSPLabel.append(target_NSP)

                if (index%batch2Show == 0 or index==len(inputs)-1) and index != 0:
                    batchinput = tf.convert_to_tensor(batchinput,tf.float32)
                    batchSegment = tf.convert_to_tensor(batchSegment,tf.float32)
                    batchRealSeq = tf.convert_to_tensor(batchRealSeq,tf.float32)
                    batchNSPLabel = tf.convert_to_tensor(batchNSPLabel,tf.float32)
                    
                    total_loss=0
                    ####################### Train Part ####################### 
                    with tf.GradientTape() as tape:
                        MLM_pred = BertModel.train_for_MLM(batchinput,batchSegment,True)
                        loss_MLM = self.loss_function_for_MLM(batchRealSeq,MLM_pred)
    
                        
                        NSP_pred = BertModel.train_for_NSP(batchRealSeq,target_segment,True)
                        loss_NSP = self.loss_object_2(batchNSPLabel, NSP_pred)
                        
                        total_loss = loss_MLM+loss_NSP
                    
                        
                    self.gradients = tape.gradient(total_loss, BertModel.trainable_variables)
                    
                        
                    ################### FOCUS ##############################
                    (self.gradients, _) = tf.clip_by_global_norm(self.gradients, clip_norm=1.0)
                    tvars = BertModel.trainable_variables
                    #tvars = tf.compat.v1.trainable_variables()
                    initialized_variable_names = {}
                    scaffold_fn = None
                    if init_checkpoint:
                        (assignment_map, initialized_variable_names) = OurBertModel.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                        
                        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                            
                    """
                    tf.compat.v1.logging.info("**** Trainable Variables ****")
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)              
                    """   
                    train_op = self.optimizer.apply_gradients(zip(self.gradients, tvars), 
                                                              global_step=self.global_step,
                                                              name=initialized_variable_names)
    
    
                    
                    train_op = tf.group(train_op, tf.convert_to_tensor([self.global_step],tf.float32))
                    
                    #########################################################
                    
                    self.train_loss_MLM(loss_MLM)
                    self.train_loss_NSP(loss_NSP)
                    
                    self.train_accuracy_MLM(batchRealSeq, MLM_pred)
                    self.train_accuracy_NSP(batchNSPLabel, NSP_pred)
                    
                    
                    if index%32 == 0 or index==len(inputs)-1:
                        grad_list = [grad for grad in self.gradients if grad is not None]
                        print("Number of not None grads is: {}".format(len(grad_list)))
                        print("ALL Trainable Variables in MLM:{}".format(len(BertModel.trainable_variables)))
                        print("Epoch:{} Batch:{} Loss_MLM:{:.6f} Accuracy_MLM:{:.6f}".format(
                            epoch+1,index,self.train_loss_MLM.result(),self.train_accuracy_MLM.result()))
                        print("__________________________________________________________")
                        print("Loss_NSP:{:.6f} Accuracy_NSP:{:.6f}".format(
                        self.train_loss_NSP.result(),self.train_accuracy_NSP.result()))
                        print("***********************************************************")
                    
                    
                    batchSegment = []
                    batchRealSeq = []
                    batchinput = []
                    batchNSPLabel = []
                    
                    if index%200000 == 0 or index==len(inputs)-1:
                        ckpt_save_path = self.ckpt_manager.save()
                        print("Saving checkpoint for epoch {} batch {} at {}".format(epoch+1,index,ckpt_save_path))
                        print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))
                        init_checkpoint = self.ckpt_manager.latest_checkpoint
                        #break
        
        return BertModel
    
    
    
class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
    
            m = tf.compat.v1.get_variable(name=param_name + "/adam_m",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())
            v = tf.compat.v1.get_variable(name=param_name + "/adam_v",
                                          shape=param.shape.as_list(),
                                          dtype=tf.float32,
                                          trainable=False,
                                          initializer=tf.zeros_initializer())
            
            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,tf.square(grad)))
              

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend([param.assign(next_param),
                               m.assign(next_m),
                               v.assign(next_v)])
            
        return tf.group(*assignments, name=name)
    
    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
    
    