from __future__ import print_function
import matplotlib
from utils.data_process_secsplit import Dataset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from utils.plot_table import plot_heatmap 
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import os
import sys


class LSTM_Model(object):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length + 2*self.opts.window_size])
        self.labels_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length])
        if self.opts.suffix:
            self.suffix_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length])
        if self.opts.num:
            self.num_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length])
        if self.opts.cap:
            self.cap_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length])
        if self.opts.jackknife:
            self.jackknife_placeholder = tf.placeholder(tf.int32, shape = [None, self.seq_length])
        self.keep_prob = tf.placeholder(tf.float32)  
        self.input_keep_prob = tf.placeholder(tf.float32)  
        self.hidden_prob = tf.placeholder(tf.float32)  
    def add_embedding(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding', self.loader.embedding_matrix.shape, initializer = tf.constant_initializer(self.loader.embedding_matrix/self.opts.embed_dropout), trainable = bool(self.opts.embedding_trainable)) 
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.seq_length+2*self.opts.window_size, inputs)]
            # when window_size >0, pad each sentence by the number 
            
        if self.opts.window_size > 0:
            inputs = [tf.concat(1, inputs[j:j+self.opts.window_size*2+1]) for j in xrange(len(inputs) - self.opts.window_size*2)]
        return inputs
#    def add_pos_embedding(self):
#        with tf.device('/cpu:0'):
#            embedding = tf.get_variable('Embedding', self.loader.pos_embedding_matrix.shape, initializer = tf.constant_initializer(self.loader.embedding_matrix), trainable = True) 
#            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
#            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.opts.pos_dim, inputs)]
#            
#        return inputs

    def add_suffix(self):
        with tf.device('/cpu:0'):
            embedding_suffix = tf.get_variable('suffix_embedding', [self.loader.nb_suffix+1, self.opts.suffix_dim], trainable = True) 
            inputs = tf.nn.embedding_lookup(embedding_suffix, self.suffix_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.seq_length, inputs)]
            return inputs
    def add_jackknife(self):
        with tf.device('/cpu:0'):
            embedding_jk = tf.get_variable('jackknife_embedding', [self.loader.jk_size+1, self.opts.jackknife_dim], trainable = True) 
            inputs = tf.nn.embedding_lookup(embedding_jk, self.jackknife_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.seq_length, inputs)]
            return inputs
    def add_num(self):
        num_inputs = [tf.cast(x, tf.float32) for x in tf.split(1, self.seq_length, self.num_placeholder)]
        return num_inputs

    def add_cap(self):
        cap_inputs = [tf.cast(x, tf.float32) for x in tf.split(1, self.seq_length, self.cap_placeholder)]
        return cap_inputs
    def add_fc(self, inputs, layer):
        fc_outputs = []
        with tf.variable_scope('fully_connected_layer{}'.format(layer)) as scope:
            if layer == 0:
                inputs_dim = self.inputs_dim
            else:
                inputs_dim = self.opts.units 
            
            for tstep, current_input in enumerate(inputs):
                if tstep >0:
                    scope.reuse_variables() # share weights aross words
                W = tf.get_variable('fc_weight', [inputs_dim, self.opts.units]) 
                b = tf.get_variable('fc_bias', [self.opts.units])
                fc_outputs.append(tf.matmul(current_input, W)+b)
            dummy_fc = tf.ones(tf.shape(fc_outputs[0]))
            dp_fc = tf.nn.dropout(dummy_fc, self.hidden_prob)
            print('synchronize over time steps, fully connected')
            fc_outputs = [o*dp_fc for o in fc_outputs]

            return fc_outputs
    
    def add_cnn(self, inputs, layer):
        if layer == 0 and self.opts.num_layers == 0:
            # apply cnn directly
            inputs_dim = self.inputs_dim
        else:
            inputs_dim = self.output_layer_dim
        conv_inputs = tf.transpose(tf.pack(inputs, 2), perm = [0, 2, 1])  # #sents by T by # units
        print('add cnn layer {}'.format(layer))
        with tf.variable_scope('Convolution{}'.format(layer)):
            W_conv = tf.get_variable('weight_conv', [2*self.opts.atwindow_size+1, inputs_dim, self.output_layer_dim])
            b_conv = tf.get_variable('bias_conv', [self.output_layer_dim])
            h_conv1 = tf.nn.relu(tf.nn.conv1d(conv_inputs, W_conv, stride = 1, padding = 'SAME')+b_conv)
            h_conv1_dropped = tf.nn.dropout(h_conv1, self.hidden_prob)
        conv_outputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.seq_length, h_conv1_dropped)]

        W_conv_list = [tf.squeeze(x, [0]) for x in tf.split(0, 2*self.opts.atwindow_size+1, W_conv)]
        k = self.opts.atwindow_size
        padded_inputs = []
        #non_paddings = tf.split(1, self.seq_length, non_padding) 
        self.contexts = []
        # make padded inputs
        for i in xrange(k):
            padded_inputs.append(tf.zeros(tf.shape(inputs[0])))
        padded_inputs.extend(inputs)
        for i in xrange(k):
            padded_inputs.append(tf.zeros(tf.shape(inputs[0])))
        for j, h in enumerate(inputs):
            h_window = padded_inputs[j:(j+2*k+1)]
            context = []
            for h_conv, W in zip(h_window, W_conv_list):
                new_h = tf.matmul(h_conv, W)
                size = (tf.reduce_sum(new_h**2, 1, keep_dims = True))**(0.5)
                context.append(size)
            self.contexts.append(tf.concat(1, context))
        #W_center = W_conv_list[self.opts.atwindow_size]
        #center_rep = []
        #for center_word  in inputs:
        #    center_vector = tf.matmul(center_word, W_center) + b_conv
        #    center_rep.append(center_vector)
        #self.contexts = []
        #for output in conv_outputs:
        #    coefs = []
        #    for center_word in center_rep:
        #        #coef = 1.0/tf.reduce_sum((output - center_word)**2, 1, keep_dims = True)
        #        coef = tf.reduce_sum(output*center_word, 1, keep_dims = True)/(tf.reduce_sum(output**2, 1, keep_dims = True)*tf.reduce_sum(center_word**2, 1, keep_dims = True)**(0.5))
        #        coefs.append(coef)
        #    coefs = tf.concat(1,coefs) 
        #    self.contexts.append(coefs)
        #    
        return conv_outputs 

    def add_lstm(self, inputs, layer, backward = False):
        print('add lstm layer{}, backward{}'.format(layer, backward))
        batch_size = tf.shape(self.input_placeholder)[0]
        with tf.variable_scope('LSTM{}backward{}'.format(layer, backward)) as scope:
            self.initial_state = [tf.zeros([batch_size, self.opts.units]), tf.zeros([batch_size, self.opts.units])]
            #self.initial_state = tf.zeros([tf.shape(self.input_placeholder)[0], 100])
            if backward:
               inputs.reverse()
            state = self.initial_state
            lstm_outputs = []
            dummy_dp = tf.ones([batch_size, self.opts.units])
            dps = [tf.nn.dropout(dummy_dp, self.hidden_prob) for _ in xrange(4)]
            if layer == 0:
                inputs_dim = self.inputs_dim
            else: 
                inputs_dim = self.opts.units
            for tstep, current_input in enumerate(inputs):
                if tstep >0:
                    scope.reuse_variables()
                
                theta_x_i = tf.get_variable('theta_x_i', [inputs_dim, self.opts.units])
                theta_x_f = tf.get_variable('theta_x_f', [inputs_dim, self.opts.units])
                theta_x_o = tf.get_variable('theta_x_o', [inputs_dim, self.opts.units] )
                theta_x_g = tf.get_variable('theta_x_g', [inputs_dim, self.opts.units])

                theta_h_i = tf.get_variable('theta_h_i', [self.opts.units, self.opts.units])
                theta_h_f = tf.get_variable('theta_h_f', [self.opts.units, self.opts.units])
                theta_h_o = tf.get_variable('theta_h_o', [self.opts.units, self.opts.units])
                theta_h_g = tf.get_variable('theta_h_g', [self.opts.units, self.opts.units])
                bias_i = tf.get_variable('bias_input', [self.opts.units])
                bias_f = tf.get_variable('bias_forget', [self.opts.units], initializer = tf.constant_initializer(1))
                bias_o = tf.get_variable('bias_output', [self.opts.units])
                bias_g = tf.get_variable('bias_extract', [self.opts.units])
                
                #RNN_H = tf.get_variable('Hmatrix', [100, 100])
                #RNN_I = tf.get_variable('Imatrix', [self.loader.embedding_matrix.shape[1], 100])
                #RNN_b = tf.get_variable('bias', [100])
                i_gate = tf.nn.sigmoid(tf.matmul(state[0]*dps[0], theta_h_i)+tf.matmul(current_input, theta_x_i)+bias_i)
                f_gate = tf.nn.sigmoid(tf.matmul(state[0]*dps[1], theta_h_f)+tf.matmul(current_input, theta_x_i)+bias_f)
                o_gate = tf.nn.sigmoid(tf.matmul(state[0]*dps[2], theta_h_o)+tf.matmul(current_input, theta_x_o)+bias_o)
                g_gate = tf.nn.tanh(tf.matmul(state[0]*dps[3], theta_h_g)+tf.matmul(current_input, theta_x_g)+bias_g)
                c_gate = i_gate*g_gate + state[1]*f_gate
                h = tf.nn.tanh(c_gate)*o_gate
                state = [h, c_gate]
                
                #
                lstm_outputs.append(h)
                #state  = tf.matmul(state, RNN_H)+tf.matmul(current_input, RNN_I)+RNN_b
                #lstm_outputs.append(state)
            self.final_state = lstm_outputs[-1]
            if not self.opts.sync:
                lstm_outputs = [tf.nn.dropout(h, self.keep_prob) for h in lstm_outputs]
            else:
                print('synchronize the output dropout')
                dummy_one = tf.ones(tf.shape(lstm_outputs[0]))
                dp = tf.nn.dropout(dummy_one, self.keep_prob)
                lstm_outputs = [dp*h for h in lstm_outputs]
            if backward:
                lstm_outputs.reverse()
            return lstm_outputs
    def concat_seq(self, concat_units):
        if len(concat_units) == 1:
            return concat_units[0] # not appending anything.  
        concatenated = []
        for concat_list in zip(*concat_units):
            concatenated.append(tf.concat(1, concat_list))

        return concatenated  

    def add_srn(self, inputs, layer, backward = False):
        batch_size = tf.shape(self.input_placeholder)[0]
        with tf.variable_scope('SRN{}backward{}'.format(layer, backward)) as scope:
            self.initial_state = tf.zeros([batch_size, self.opts.units])
            #self.initial_state = tf.zeros([tf.shape(self.input_placeholder)[0], 100])
            if backward:
               inputs.reverse()
            state = self.initial_state
            srn_outputs = []
            dummy_dp = tf.ones([batch_size, self.opts.units])
            hidden_dp = tf.nn.dropout(dummy_dp, self.hidden_prob) 
            if layer == 0:
                inputs_dim = self.inputs_dim
            else: 
                inputs_dim = self.opts.units
            for tstep, current_input in enumerate(inputs):
                if tstep >0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('Hmatrix', [self.opts.units, self.opts.units])
                RNN_I = tf.get_variable('Imatrix', [inputs_dim, self.opts.units])
                RNN_b = tf.get_variable('bias', [self.opts.units])
                state = tf.nn.sigmoid(tf.matmul(current_input, RNN_I)+tf.matmul(state*hidden_dp, RNN_H) + RNN_b)
                srn_outputs.append(state)
                
            if not self.opts.sync:
                srn_outputs = [tf.nn.dropout(h, self.keep_prob) for h in srn_outputs]
            else:
                print('synchronize the output dropout')
                dummy_one = tf.ones(tf.shape(srn_outputs[0]))
                dp = tf.nn.dropout(dummy_one, self.keep_prob)
                srn_outputs = [dp*h for h in srn_outputs]
            if backward:
                srn_outputs.reverse()
            return srn_outputs

    def add_gru(self, inputs, layer, backward = False):
        batch_size = tf.shape(self.input_placeholder)[0]
        with tf.variable_scope('GRU{}backward{}'.format(layer, backward)) as scope:
            self.initial_state = tf.zeros([batch_size, self.opts.units])
            #self.initial_state = tf.zeros([tf.shape(self.input_placeholder)[0], 100])
            if backward:
               inputs.reverse()
            state = self.initial_state
            gru_outputs = []
            dummy_dp = tf.ones([batch_size, self.opts.units])
            hidden_dps = [tf.nn.dropout(dummy_dp, self.hidden_prob) for _ in xrange(3)]
            if layer == 0:
                inputs_dim = self.inputs_dim
            else: 
                inputs_dim = self.opts.units
            for tstep, current_input in enumerate(inputs):
                if tstep >0:
                    scope.reuse_variables()
                U_z = tf.get_variable('U_z', [inputs_dim, self.opts.units])
                U_r = tf.get_variable('U_r', [inputs_dim, self.opts.units])
                U_h = tf.get_variable('U_h', [inputs_dim, self.opts.units])
                W_z = tf.get_variable('W_z', [self.opts.units, self.opts.units])
                W_r = tf.get_variable('W_r', [self.opts.units, self.opts.units])
                W_h = tf.get_variable('W_h', [self.opts.units, self.opts.units])
                b_z = tf.get_variable('b_z', [self.opts.units])
                b_r = tf.get_variable('b_r', [self.opts.units])
                b_h = tf.get_variable('b_h', [self.opts.units])
                
                z = tf.nn.sigmoid(tf.matmul(current_input, U_z)+tf.matmul(state*hidden_dps[0], W_z)+ b_z)
                r = tf.nn.sigmoid(tf.matmul(current_input, U_r)+tf.matmul(state*hidden_dps[1], W_r)+ b_r)
                h = tf.nn.tanh(tf.matmul(current_input, U_h)+tf.matmul((state*hidden_dps[2]*r), W_h)+b_h)
                state = (1.0-z)*h + z*state 
                gru_outputs.append(state)
                
            if not self.opts.sync:
                gru_outputs = [tf.nn.dropout(h, self.keep_prob) for h in srn_outputs]
            else:
                print('synchronize the output dropout')
                dummy_one = tf.ones(tf.shape(gru_outputs[0]))
                dp = tf.nn.dropout(dummy_one, self.keep_prob)
                gru_outputs = [dp*h for h in gru_outputs]
            if backward:
                gru_outputs.reverse()
            return gru_outputs

    def concat_seq(self, concat_units):
        if len(concat_units) == 1:
            return concat_units[0] # not appending anything.  
        concatenated = []
        for concat_list in zip(*concat_units):
            concatenated.append(tf.concat(1, concat_list))

        return concatenated  
    def add_attention_soft(self, inputs):
        batch_size = tf.shape(self.input_placeholder)[0]
        with tf.variable_scope('Attention'):
            U_soft = tf.get_variable('weight', [self.output_layer_dim, self.seq_length])
            b_soft = tf.get_variable('bias', [self.seq_length])
            contexts = [tf.nn.softmax(tf.matmul(h, U_soft) + b_soft) for h in inputs] # element (100, T)
            contexts_mat = tf.concat(1, contexts) # 100 by 2T 
            contexts = [tf.reshape(tf.squeeze(x, [0]), [self.seq_length, self.seq_length])  for x in tf.split(0, batch_size, contexts_mat)]
            Hmatrix = tf.concat(1, inputs) 
            Hs= [tf.reshape(tf.squeeze(x, [0]), [self.seq_length, -1]) for x in tf.split(0, batch_size, Hmatrix)]
            outputs_list = [tf.matmul(c, H) for (H, c) in zip(Hs, contexts)] # len(outputs_list) = 100 T by units
            outputs_list = tf.split(0, self.seq_length, tf.concat(1, outputs_list))
            outputs = [tf.reshape(tf.squeeze(output, [0]), [batch_size, self.output_layer_dim]) for output in outputs_list]
            
        return outputs
            
    def add_attention_soft_tanh(self, inputs):
        batch_size = tf.shape(self.input_placeholder)[0]
        with tf.variable_scope('Attention'):
            U_soft = tf.get_variable('weight', [self.output_layer_dim, self.seq_length])
            V_soft = tf.get_variable('bias', [self.seq_length, self.seq_length])
            contexts = [tf.nn.softmax(tf.matmul(tf.nn.tanh(tf.matmul(h, U_soft)), V_soft)) for h in inputs] # element (100, T)
            contexts_mat = tf.concat(1, contexts) # 100 by 2T 
            contexts = [tf.reshape(tf.squeeze(x, [0]), [self.seq_length, self.seq_length])  for x in tf.split(0, batch_size, contexts_mat)]
            Hmatrix = tf.concat(1, inputs) 
            Hs= [tf.reshape(tf.squeeze(x, [0]), [self.seq_length, -1]) for x in tf.split(0, batch_size, Hmatrix)]
            outputs_list = [tf.matmul(c, H) for (H, c) in zip(Hs, contexts)] # len(outputs_list) = 100 T by units
            outputs_list = tf.split(0, self.seq_length, tf.concat(1, outputs_list))
            outputs = [tf.reshape(tf.squeeze(output, [0]), [batch_size, self.output_layer_dim]) for output in outputs_list]
            
        return outputs

    def add_attention_window(self, inputs, bias_identity = True):

        k = self.opts.atwindow_size
        bias = np.zeros(2*k+1)
        bias[k] = 1
        # pad inputs 
        padded_inputs = []
        #non_paddings = tf.split(1, self.seq_length, non_padding) 
        for i in xrange(k):
            padded_inputs.append(tf.zeros(tf.shape(inputs[0])))
        padded_inputs.extend(inputs)
        for i in xrange(k):
            padded_inputs.append(tf.zeros(tf.shape(inputs[0])))
        
        with tf.variable_scope('attention') as scope:
            U_soft = tf.get_variable('weight', [self.output_layer_dim, 2*k+1])
            if bias_identity:
                b_soft = tf.get_variable('bias', [2*k+1], initializer = tf.constant_initializer(bias))
            else:
                b_soft = tf.get_variable('bias', [2*k+1])

            outputs = []
            contexts = []
            for j, h in enumerate(inputs):
                coefs = tf.nn.softmax(tf.matmul(h, U_soft)+b_soft)
                contexts.append(coefs)
                h_window = padded_inputs[j:(j+2*k+1)]
                h_new = tf.add_n([coef*h_ for (coef, h_) in zip(tf.split(1, 2*k+1, coefs), h_window)])
                outputs.append(h_new)
            self.contexts = contexts
        return outputs

    def add_attention_window_cos(self, inputs):

        k = self.opts.atwindow_size
        with tf.variable_scope('attention') as scope:
            U_soft = tf.get_variable('weight', [self.output_layer_dim, self.output_layer_dim])
            b_soft = tf.get_variable('bias', [self.output_layer_dim])

            outputs = []
            contexts = []
            print('not padding. window size varies')
            for j, h in enumerate(inputs):
                if j <= k:
                    h_window = inputs[:k+j+1]
                else:
                    h_window = inputs[j-k:j+k+1]
                h_module = tf.matmul(h, U_soft) + b_soft # transform the vector to a vector it wants to find 
                coefs = [tf.reduce_sum(h_module*h_, 1, keep_dims = True) for h_ in h_window] # search within the window taking inner products
                coefs = tf.nn.softmax(tf.concat(1, coefs)) # normalize add bias?
                contexts.append(coefs)
                h_new = tf.add_n([coef*h_ for (coef, h_) in zip(tf.split(1, len(h_window), coefs), h_window)])
                outputs.append(h_new)
            self.contexts = contexts
        return outputs

    def add_attention_window_cos_both(self, inputs, center = False, bias_non_identity = True, compress = False, tanh = False, add_bias = True, output_layer_dim = None, softmax = False):

        k = self.opts.atwindow_size
        if output_layer_dim is None:
            output_layer_dim = self.output_layer_dim
        with tf.variable_scope('attention') as scope:
            if not compress:
                U_soft = tf.get_variable('weight', [output_layer_dim, output_layer_dim])
                if add_bias:
                    b_soft = tf.get_variable('bias', [output_layer_dim])
            else:
                print('compress')
                U_soft = tf.get_variable('weight', [output_layer_dim, 50])
                V_soft = tf.get_variable('weight_v', [output_layer_dim, 50])
            if not compress:
                U_soft_right = tf.get_variable('weight_right', [output_layer_dim, output_layer_dim])
                if add_bias:
                    b_soft_right = tf.get_variable('bias_right', [output_layer_dim])
            else:
                U_soft_right = tf.get_variable('weight_right', [output_layer_dim, 5])
                V_soft_right = tf.get_variable('weight_right_v', [output_layer_dim, 5])
                if tanh:
                    scale = tf.get_variable('scale_at', [1])
                
            if center:
                print('use another vector for centred words')
                u_center = tf.get_variable('weight_center', [output_layer_dim, 1])
                if bias_non_identity:
                    b_center = tf.get_variable('bias_center', [1], initializer = tf.constant_initializer(0.1))
                else:
                    if add_bias:
                        b_center = tf.get_variable('bias_center', [1])
            
            #non_padding = tf.cast(tf.not_equal(self.labels_placeholder, tf.zeros(tf.shape(self.labels_placeholder), tf.int32)), tf.float32)
            #non_paddings = tf.split(1, self.seq_length, non_padding) 

            outputs = []
            contexts = [] 
            print('not padding. window size varies')
            for j, h in enumerate(inputs):
                h_window_right = inputs[j+1:k+j+1]
                #non_padding_right = non_paddings[j+1:k+j+1]
                if j <= k:
                    h_window_left = inputs[:j]
                    #non_padding_left = non_paddings[:j]
                else:
                    h_window_left = inputs[j-k:j]
                    #non_padding_left = non_paddings[j-k:j]
                h_window = h_window_left + [h] + h_window_right
                #non_padding_window = non_padding_left + [non_paddings[j]] + non_padding_right
                if not compress:
                    if add_bias:
                        h_module = tf.matmul(h, U_soft) + b_soft# transform the vector to a vector it wants to find 
                        h_module_right = tf.matmul(h, U_soft_right)+b_soft_right # transform the vector to a vector it wants to find 
                    else:
                        if softmax:
                            print('softmax right and left')
                            h_module = tf.matmul(tf.nn.softmax(h), U_soft) # transform the vector to a vector it wants to find 
                            h_module_right = tf.matmul(tf.nn.softmax(h), U_soft_right) # transform the vector to a vector it wants to find 
                        else:
                            h_module = tf.matmul(h, U_soft) # transform the vector to a vector it wants to find 
                            h_module_right = tf.matmul(h, U_soft_right)# transform the vector to a vector it wants to find 
                else:
                    #if tanh:
                    #    print('would this avoid identity?')
                    #    h_module = tf.nn.tanh(tf.matmul(h, U_soft)) # transform the vector to a vector it wants to find 
                    #    h_module_right = tf.nn.tanh(tf.matmul(h, U_soft_right)) # transform the vector to a vector it wants to find 
                    #    h_window_left = [tf.nn.tanh(tf.matmul(h_, V_soft)) for h_ in h_window_left]
                    #    h_window_right = [tf.nn.tanh(tf.matmul(h_, V_soft_right)) for h_ in h_window_right]
                    #else:
                    if softmax:
                        print('softmax right and left')
                        h_module = tf.matmul(tf.nn.softmax(h), U_soft) # transform the vector to a vector it wants to find 
                        h_module_right = tf.matmul(tf.nn.softmax(h), U_soft_right) # transform the vector to a vector it wants to find 
                    else:
                        h_module = tf.matmul(h, U_soft) # transform the vector to a vector it wants to find 
                        h_module_right = tf.matmul(h, U_soft_right) # transform the vector to a vector it wants to find 
                    h_window_left = [tf.matmul(h_, V_soft) for h_ in h_window_left]
                    h_window_right = [tf.matmul(h_, V_soft_right) for h_ in h_window_right]
                    
                coefs_left = [tf.reduce_sum(h_module*h_, 1, keep_dims = True) for h_ in h_window_left] # search within the window taking inner products
                if center:
                    if add_bias:
                        coefs_center = [tf.matmul(h, u_center) + b_center]
                    else:
                        if softmax:
                            print('softmax center')
                            coefs_center = [tf.matmul(tf.nn.softmax(h), u_center)]
                        else:
                            coefs_center = [tf.matmul(h, u_center)]
                else:
                    coefs_center = [tf.reduce_sum(h*h, 1, keep_dims = True)]
                coefs_right = [tf.reduce_sum(h_module_right*h_, 1, keep_dims = True) for h_ in h_window_right]
                if tanh:
                    print('taking tanh to avoid saturation')
                    coefs_left = [scale*tf.nn.tanh(c) for c in coefs_left]
                    coefs_center = [scale*tf.nn.tanh(c) for c in coefs_center]
                    coefs_right = [scale*tf.nn.tanh(c) for c in coefs_right]
                coefs = tf.nn.softmax(tf.concat(1, coefs_left+coefs_center+coefs_right)) # normalize add bias?
                #coefs_new = coefs*tf.concat(1, non_padding_window)
                #print('coefs ignore paddings')
                #coefs_new_normalized = coefs_new/tf.reduce_sum(coefs_new, 1, True)
                contexts.append(coefs)
                h_new = tf.add_n([coef*h_ for (coef, h_) in zip(tf.split(1, len(h_window), coefs), h_window)])
                outputs.append(h_new)
            self.contexts = contexts
        return outputs
    def add_attention_euc(self, inputs, output_layer_dim = None):
        
        batch_size = tf.shape(self.input_placeholder)[0]

        k = self.opts.atwindow_size
        if output_layer_dim is None:
            output_layer_dim = self.output_layer_dim
        with tf.variable_scope('attention') as scope:
            U_soft = tf.get_variable('weight', [output_layer_dim, 50])
            V_soft = tf.get_variable('weight_v', [output_layer_dim, 50])
            U_soft_right = tf.get_variable('weight_right', [output_layer_dim, 50])
            V_soft_right = tf.get_variable('weight_right_v', [output_layer_dim, 50])
                
            outputs = []
            contexts = [] 
            print('not padding. window size varies')
            for j, h in enumerate(inputs):
                h_window_right = inputs[j+1:k+j+1]
                #non_padding_right = non_paddings[j+1:k+j+1]
                if j <= k:
                    h_window_left = inputs[:j]
                    #non_padding_left = non_paddings[:j]
                else:
                    h_window_left = inputs[j-k:j]
                    #non_padding_left = non_paddings[j-k:j]
                h_window = h_window_left + [h] + h_window_right
                #non_padding_window = non_padding_left + [non_paddings[j]] + non_padding_right
                h_module = tf.matmul(h, U_soft) # transform the vector to a vector it wants to find 
                h_module_right = tf.matmul(h, U_soft_right) # transform the vector to a vector it wants to find 
                h_window_left = [tf.matmul(h_, V_soft) for h_ in h_window_left]
                h_window_right = [tf.matmul(h_, V_soft_right) for h_ in h_window_right]
                print('euc attention')    
                coefs_left = [1.0/(1+tf.reduce_sum((h_module-h_)**2, 1, keep_dims = True)) for h_ in h_window_left] # search within the window taking inner products
                coefs_center = [tf.ones([batch_size, 1])]
                coefs_right = [1.0/(1+tf.reduce_sum((h_module_right-h_)**2, 1, keep_dims = True)) for h_ in h_window_right]
                coefs = tf.concat(1, coefs_left+coefs_center+coefs_right) # normalize add bias?
                #coefs_new = coefs*tf.concat(1, non_padding_window)
                #print('coefs ignore paddings')
                #coefs_new_normalized = coefs_new/tf.reduce_sum(coefs_new, 1, True)
                contexts.append(coefs)
                h_new = tf.add_n([coef*h_ for (coef, h_) in zip(tf.split(1, len(h_window), coefs), h_window)])
                outputs.append(h_new)
            self.contexts = contexts
        return outputs

    def add_projection(self, inputs, tag_size = None, output_layer_dim = None):
        scope_name = 'Projection'
        if tag_size is None:
            tag_size = self.tag_size
            #scope_name += '_context' not for the model
        if output_layer_dim is None:
            output_layer_dim = self.output_layer_dim
        with tf.variable_scope(scope_name):
            proj_U = tf.get_variable('weight', [output_layer_dim, tag_size])
            proj_b = tf.get_variable('bias', [tag_size])
            outputs = [tf.matmul(i, proj_U)+proj_b for i in inputs]
        return outputs

    def add_loss_op(self, output):
        all_ones = [tf.ones([tf.shape(self.input_placeholder)[0]*(tf.shape(self.input_placeholder)[1]-self.opts.window_size*2)])]
        weight = [tf.cast(tf.not_equal(tf.reshape(self.labels_placeholder, [-1]), tf.zeros([tf.shape(self.input_placeholder)[0]*(tf.shape(self.input_placeholder)[1]-self.opts.window_size*2)], tf.int32)), tf.float32)] 
        
        #all_ones = [tf.ones([self.length*self.batch_size])]
       
        cross_entropy = sequence_loss([output], [tf.reshape(self.labels_placeholder, [-1])], weight, self.tag_size)
        tf.add_to_collection('total loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total loss'))
        return loss
    def add_accuracy(self, predictions):
        predictions = tf.to_int32(tf.argmax(predictions, 1))
        targets = tf.reshape(self.labels_placeholder, [-1])
        self.predictions = predictions 
        self.targets = targets 
        non_padding = tf.cast(tf.not_equal(targets, tf.zeros([tf.size(self.labels_placeholder)], tf.int32)), tf.int32)

        correct_predictions = non_padding*tf.cast(tf.equal(predictions, targets), tf.int32)
      
        self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))/tf.reduce_sum(tf.cast(non_padding, tf.float32))
    def add_training_op(self, loss):
        if self.opts.recurrent_attention == 2:
            optimizer = tf.train.AdamOptimizer(0.001)
        else:
            optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(self.calculate_loss)
        return train_op
        
    def __init__(self, opts, loader = None):
       
        self.opts = opts

        
        if not loader:
            self.loader = Dataset(opts)

        else:
            self.loader = loader 
           
        
        self.tag_size = self.loader.tag_size + 1 # padding include
        self.seq_length = self.loader.y_train.shape[1] # for None
        self.batch_size = 100
        self.inputs_dim = (1+2*self.opts.window_size)*self.opts.embedding_dim + self.opts.suffix_dim + self.opts.num + self.opts.cap
	if not self.loader.kfold:
	    self.inputs_dim += self.opts.jackknife_dim
	
        self.output_layer_dim = (self.opts.bi + 1)*self.opts.units # bi = 1 if yes, bi = 0 if no
        
        self.add_placeholders()
        inputs_list = [self.add_embedding()]
        
        if self.opts.suffix:
            inputs_list.append(self.add_suffix())
        if self.opts.num:
            inputs_list.append(self.add_num())
        if self.opts.cap:
            inputs_list.append(self.add_cap())
        if self.opts.jackknife and not self.loader.kfold:
            inputs_list.append(self.add_jackknife())
        self.inputs = self.concat_seq(inputs_list)
        if not self.opts.sync:
            lstm_inputs = [tf.nn.dropout(i, self.input_keep_prob) for i in self.inputs]
        else:
            print('synchronizing the input dropout')
            dummy_in = tf.ones(tf.shape(self.inputs[0]))
            dp_in = tf.nn.dropout(dummy_in, self.input_keep_prob)
            lstm_inputs = [i*dp_in for i in self.inputs]
        if opts.recurrent_attention == 0:
            if opts.attention == 10:
                num_layers = 1
                self.opts.units = self.opts.units/2
                self.output_layer_dim = self.output_layer_dim/2
            else:
                num_layers = opts.num_layers
                

            for i in xrange(num_layers):
                if opts.lstm == 1:
                    lstm_inputs = self.add_lstm(lstm_inputs, i)
                elif opts.lstm == 0:
                    print('running simple recurrent neural nets')
                    lstm_inputs = self.add_srn(lstm_inputs, i) 
                elif opts.lstm == 2:
                    print('gru')
                    lstm_inputs = self.add_gru(lstm_inputs, i) 
            self.left_lstm_outputs = lstm_inputs

            outputs_list = [self.left_lstm_outputs]
            if self.opts.bi:
                if not self.opts.sync:
                    lstm_inputs = [tf.nn.dropout(i, self.input_keep_prob) for i in self.inputs]
                else:
                    print('synchronizing the input dropout')
                    lstm_inputs = [i*dp_in for i in self.inputs]
                for i in xrange(num_layers):
                    if opts.lstm == 1:
                        lstm_inputs = self.add_lstm(lstm_inputs, i, True)
                    elif opts.lstm == 0:
                        print('running simple recurrent neural nets')
                        lstm_inputs = self.add_srn(lstm_inputs, i, True) 
                    elif opts.lstm == 2:
                        print('gru')
                        lstm_inputs = self.add_gru(lstm_inputs, i, True) 

                self.right_lstm_outputs = lstm_inputs
                outputs_list.append(self.right_lstm_outputs)

            self.proj_inputs = self.concat_seq(outputs_list)
            if opts.attention == 1:
                print('add attention')
                self.proj_inputs = self.add_attention_soft(self.proj_inputs)
            if opts.attention == 2:
                print('add soft tanh attention')
                self.proj_inputs = self.add_attention_soft_tanh(self.proj_inputs)

            if opts.attention == 3:
                print('add relative attention')
                self.proj_inputs = self.add_attention_window(self.proj_inputs, False)
            if opts.attention == 4:
                print('add relative attention with inner products')
                self.proj_inputs = self.add_attention_window_cos(self.proj_inputs)
            if opts.attention == 5:
                print('add relative attention with inner products on both sides')
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs)
            if opts.attention == 6:
                print('add relative attention with inner products on both sides centered, bias against identity')
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs, True)
            if opts.attention == 7:
                print('add relative attention with inner products on both sides centered non bias')
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs, True, False)
            if opts.attention == 11:
                print('add relative attention with inner products on both sides centered non bias')
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs, True, False, add_bias = False)
            if opts.attention == 8:
                print('add relative attention with inner products on both sides centered non bias')
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs, True, False, True)
            if opts.attention == 9:
                self.proj_inputs = self.add_attention_window_cos_both(self.proj_inputs, True, False, True, True)
            if opts.attention == 40:
                self.proj_inputs = self.add_attention_euc(self.proj_inputs)
            if opts.attention == 30:
                print('add cnn')
                for j in xrange(opts.cnn_layers):
                    self.proj_inputs = self.add_cnn(self.proj_inputs, j)
            if opts.attention == 10:
                self.inputs = self.add_attention_window_cos_both(self.proj_inputs, False, False, True)
                self.opts.units = self.opts.units*2 
                self.output_layer_dim = self.output_layer_dim*2
                if not self.opts.sync:
                    lstm_inputs = [tf.nn.dropout(i, self.input_keep_prob) for i in self.inputs]
                else:
                    print('synchronizing the input dropout')
                    dummy_in_new = tf.ones(tf.shape(self.inputs[0]))
                    dp_in_new = tf.nn.dropout(dummy_in_new, self.input_keep_prob)
                    lstm_inputs = [i*dp_in_new for i in self.inputs]
                for i in xrange(1, opts.num_layers):
                    if opts.lstm == 1:
                        lstm_inputs = self.add_lstm(lstm_inputs, i)
                    elif opts.lstm == 0:
                        print('running simple recurrent neural nets')
                        lstm_inputs = self.add_srn(lstm_inputs, i) 
                    elif opts.lstm == 2:
                        print('gru')
                        lstm_inputs = self.add_gru(lstm_inputs, i) 
                self.left_lstm_outputs = lstm_inputs

                outputs_list = [self.left_lstm_outputs]
                if self.opts.bi:
                    if not self.opts.sync:
                        lstm_inputs = [tf.nn.dropout(i, self.input_keep_prob) for i in self.inputs]
                    else:
                        print('synchronizing the input dropout')
                        lstm_inputs = [i*dp_in_new for i in self.inputs]
                    for i in xrange(1, self.opts.num_layers):
                        if opts.lstm == 1:
                            lstm_inputs = self.add_lstm(lstm_inputs, i, True)
                        elif opts.lstm == 0:
                            print('running simple recurrent neural nets')
                            lstm_inputs = self.add_srn(lstm_inputs, i, True) 
                        elif opts.lstm == 2:
                            print('gru')
                            lstm_inputs = self.add_gru(lstm_inputs, i, True) 

                    self.right_lstm_outputs = lstm_inputs
                    outputs_list.append(self.right_lstm_outputs)

                self.proj_inputs = self.concat_seq(outputs_list)
            self.outputs = self.add_projection(self.proj_inputs)
            if opts.attention == 20:
                print('projected attention')
                self.outputs = self.add_attention_window_cos_both(self.outputs, True, False, add_bias = False, output_layer_dim = self.tag_size)
        
            if opts.attention == 21:
                print('projected attention softmax')
                self.outputs = self.add_attention_window_cos_both(self.outputs, True, False, add_bias = False, output_layer_dim = self.tag_size, softmax = True)
            if opts.attention == 22:
                print('projected attention')
                self.outputs = self.add_attention_window_cos_both(self.outputs, True, False, add_bias = False, output_layer_dim = self.tag_size, softmax = True, compress = True)
        else: # if recurrent attention
            print('recurrent attention')

            for l in xrange(3):
                lstm_inputs = self.add_fc(lstm_inputs, l) # need to add dp later
            hs = lstm_inputs
            if opts.recurrent_attention == 2:
                print('add relative attention with inner products on both sides centered non bias')
                self.output_layer_dim = self.opts.units # for projection not bi
                self.proj_inputs = self.add_attention_window_cos_both(lstm_inputs, True, False)
            
                self.outputs = self.add_projection(self.proj_inputs)
            else:
            # start recurrent attention
                    
                for i in xrange(1, opts.num_layers+1): # not layer 0, dimension equal to input dim
                    if opts.lstm == 1:
                        lstm_inputs = self.add_lstm(lstm_inputs, i)
                    elif opts.lstm == 0:
                        print('running simple recurrent neural nets')
                        lstm_inputs = self.add_srn(lstm_inputs, i) 
                    elif opts.lstm == 2:
                        print('gru')
                        lstm_inputs = self.add_gru(lstm_inputs, i) 
                self.left_lstm_outputs = lstm_inputs

                outputs_list = [self.left_lstm_outputs]
                if self.opts.bi:
                    lstm_inputs = hs
                    for i in xrange(1, opts.num_layers+1):
                        if opts.lstm == 1:
                            lstm_inputs = self.add_lstm(lstm_inputs, i, True)
                        elif opts.lstm == 0:
                            print('running simple recurrent neural nets')
                            lstm_inputs = self.add_srn(lstm_inputs, i, True) 
                        elif opts.lstm == 2:
                            print('gru')
                            lstm_inputs = self.add_gru(lstm_inputs, i, True) 
                        print(hs == lstm_inputs)

                    self.right_lstm_outputs = lstm_inputs
                    outputs_list.append(self.right_lstm_outputs)

                proj_inputs = self.concat_seq(outputs_list)
                contexts = self.add_projection(proj_inputs, self.seq_length)
                contexts = [tf.nn.softmax(o) for o in contexts]
                non_padding = tf.cast(tf.not_equal(self.labels_placeholder, tf.zeros(tf.shape(self.labels_placeholder), tf.int32)), tf.float32)
                contexts = [o*non_padding for o in contexts]
                contexts = [o/tf.reduce_sum(o, 1, keep_dims = True) for o in contexts]
                self.contexts = contexts
                self.proj_inputs = [tf.add_n([coef*h_ for (coef, h_) in zip(tf.split(1, self.seq_length, context), hs)]) for context in contexts]
            
                self.outputs = self.add_projection(self.proj_inputs, None, self.opts.units)
         
        
        predictions = [tf.nn.softmax(tf.cast(o, 'float32')) for o in self.outputs]
        predictions = tf.reshape(tf.concat(1, predictions), [-1, self.tag_size])
        self.add_accuracy(predictions)
        #self.print_pred = tf.Print(self.predictions, [self.predictions])
        output = tf.reshape(tf.concat(1, self.outputs), [-1, self.tag_size])
        self.calculate_loss = self.add_loss_op(output)
        self.train_op = self.add_training_op(self.calculate_loss)
        self.train_loss = []
        self.test_loss = []
    def run_batch(self, session, testmode = False, get_contexts = None):
        if not testmode:
            feed = {}
            feed[self.input_placeholder] =  self.loader.X_train_batch
            feed[self.labels_placeholder] = self.loader.y_train_batch
            if self.opts.suffix:
                feed[self.suffix_placeholder] = self.loader.suffix_train_batch
            
            if self.opts.jackknife and not self.loader.kfold: 
                feed[self.jackknife_placeholder] = self.loader.jackknife_train_batch
            if self.opts.num:
                feed[self.num_placeholder] = self.loader.train_num_indicator_batch
            
            if self.opts.cap:
                feed[self.cap_placeholder] = self.loader.train_cap_indicator_batch
            feed[self.keep_prob] = self.opts.dropout_p
            feed[self.hidden_prob] = self.opts.hidden_p
            feed[self.input_keep_prob] = self.opts.input_dp
            
            train_op = self.train_op
        else:
            if get_contexts is not None: # make sure we can get Sentence 0 as well
                self.loader.test_one(get_contexts) # feed only one sentence
                X_test_batch = self.loader.X_test_batch[:, self.opts.window_size:self.opts.window_size+self.seq_length]
                non_padding_idx = X_test_batch!=0
                sentence = self.loader.seq_to_sent(np.squeeze(X_test_batch[non_padding_idx]))
                gold_tags = self.loader.seq_to_tags(np.squeeze(self.loader.y_test_batch[non_padding_idx]))

            feed = {}
            feed[self.input_placeholder] =  self.loader.X_test_batch
            feed[self.labels_placeholder] = self.loader.y_test_batch
            if self.opts.suffix:
                feed[self.suffix_placeholder] = self.loader.suffix_test_batch
            if self.opts.jackknife and not self.loader.kfold: 
                feed[self.jackknife_placeholder] = self.loader.jackknife_test_batch
            if self.opts.num:
                feed[self.num_placeholder] = self.loader.test_num_indicator_batch
            if self.opts.cap:
                feed[self.cap_placeholder] = self.loader.test_cap_indicator_batch
            feed[self.keep_prob] = 1.0
            feed[self.hidden_prob] = 1.0 
            feed[self.input_keep_prob] = 1.0
            train_op = tf.no_op()

        if get_contexts is not None:
            accuracy_level, loss, _, predictions, inputs, contexts =session.run([self.accuracy, self.calculate_loss, train_op, self.predictions, self.inputs, self.contexts], feed_dict = feed)
            print(self.loader.X_test_batch)
            contexts = contexts[:np.sum(non_padding_idx)]
            predictions = predictions[:np.sum(non_padding_idx)] # unpadding
            predicted_tags = self.loader.seq_to_tags(predictions)
            for i in xrange(len(predictions)):
                if predictions[i] != np.squeeze(self.loader.y_test_batch[non_padding_idx])[i]:
                    sentence[i] += '*'
            return (contexts, sentence, gold_tags, predicted_tags)
        else:
            accuracy_level, loss, _, predictions, inputs =session.run([self.accuracy, self.calculate_loss, train_op, self.predictions, self.inputs], feed_dict = feed)

        return accuracy_level, loss, predictions

    def run_epoch(self, session, testmode = False):

        if not testmode:
            epoch_start_time = time.time()
            if self.loader.kfold:
                print('kfold loading')
                next_batch = self.loader.next_batch_k
                self.epoch_size = self.loader._num_hold_in_examples
            else:
                next_batch = self.loader.next_batch
                self.epoch_size = self.loader.X_train.shape[0]
            epoch_incomplete = next_batch(self.batch_size)
            while epoch_incomplete:
                accuracy_level, loss, predictions  = self.run_batch(session)
                self.train_loss.append(loss)
                print('{}/{}, loss {:.4f}, accuracy {:.4f}'.format(self.loader._index_in_epoch,self.epoch_size,loss, accuracy_level), end = '\r')
                epoch_incomplete = next_batch(self.batch_size)
            print('\nEpoch Training Time {}'.format(time.time() - epoch_start_time))
        else: 
            total_loss = []
            total_accuracy = []
            predictions = []

            if self.loader.kfold:
                print('kfold next batch loading')
                next_test_batch = self.loader.next_test_batch_k
                self.test_size = self.loader._num_hold_out_examples
            else:
                next_test_batch = self.loader.next_test_batch
                self.test_size = self.loader.X_test.shape[0]
            test_incomplete = next_test_batch(self.batch_size)
            while test_incomplete:
                accuracy_level, loss, batch_predictions = self.run_batch(session, True)
                total_loss.append(loss)
                total_accuracy.append(accuracy_level)
                predictions.append(batch_predictions)
                print('Testmode {}/{}, loss {}, accuracy {}'.format(self.loader._index_in_test,self.test_size,loss, accuracy_level), end = '\r')
                test_incomplete = next_test_batch(self.batch_size)
            loss = np.mean(total_loss)
            accuracy_level = np.mean(total_accuracy)
            self.test_loss.append(loss)
            predictions = np.hstack(predictions)
            
        return accuracy_level, loss, predictions
    def plot_losses(self, plotname = 'loss_plot.png'):
        plotname = os.path.join(self.opts.model_dir, plotname)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.train_loss)
        self.num_batch_per_epoch = self.epoch_size//self.batch_size + 1
        #ax.plot(range(self.num_batch_per_epoch, self.num_batch_per_epoch, self.num_batch_per_epoch*2+1), self.test_loss)
        ax.plot(range(self.num_batch_per_epoch - 1, self.num_batch_per_epoch*self.loader._epoch_completed, self.num_batch_per_epoch), self.test_loss)
        ax.set_title('Loss per batch')
        fig.savefig(plotname)
        
        
def run_model(opts, loader = None, plot = True, modelname=None, epoch=0, best_accuracy=0, saving_dir = None, get_contexts = None):
    g = tf.Graph()
    with g.as_default():
        if not modelname: # do not re-seed for further training option
            tf.set_random_seed(opts.seed)
        model = LSTM_Model(opts, loader)
        saver = tf.train.Saver(max_to_keep=2)
        X_test = model.loader.X_test[:, opts.window_size:opts.window_size+model.seq_length].reshape(-1)
        unknowns = list(set(model.loader.X_test.flat)-set(model.loader.X_train.flat))
        nb_words = np.sum(model.loader.X_test != 0)
        with tf.Session() as session: 
            session.run(tf.initialize_all_variables())
            if modelname:
                print('using an existing model')
                saver = tf.train.Saver()
                saver.restore(session, modelname)
            bad_times = 0

            if get_contexts is not None:
                for sent_idx in get_contexts:
                    print('sent index{}'.format(sent_idx))
                    contexts, sentence, gold_tags, predicted_tags = model.run_batch(session, True, sent_idx)
                    model_dir = os.path.dirname(modelname)
                    plot_heatmap(contexts, sentence, os.path.join(model_dir, 'heatmaps', 'sent{}.png'.format(sent_idx)), opts)
                    plot_heatmap(contexts, sentence, os.path.join(model_dir, 'heatmaps_exp', 'sent{}.png'.format(sent_idx)), opts, exp = True)
                    #with open(os.path.join(model_dir, 'contexts{}.pkl'.format(sent_idx)), 'wb') as fhand:
                    #    print('saving contexts to {}'.format(model_dir))
                    #    pickle.dump(contexts, fhand)
                sys.exit('output contexts')
            
            if epoch==opts.max_epochs: # just testing

                test_accuracy, loss, predictions = model.run_epoch(session, True)
                test_accuracy = np.mean(predictions[X_test!=0] == model.loader.y_test.reshape(-1)[X_test!=0])
                nb_unknowns = 0
                nb_correct = 0.0
                for unknown in unknowns:
                    comp = predictions[X_test==unknown] == model.loader.y_test.reshape(-1)[X_test==unknown]
                    nb_unknowns += comp.size
                    nb_correct += np.sum(comp)
                
                print('test loss {}'.format(loss))
                print('test accuracy {}'.format(test_accuracy))
                print('we found {} unknown words out of {} words in the test set'.format(nb_unknowns, nb_words))
                print('unknown word accuracy {}'.format(nb_correct/nb_unknowns))
                 
                if saving_dir:
                    print('outputting test pred')
                    with open(os.path.join(saving_dir, 'predictions_test.pkl'), 'wb') as fhand:
                        pickle.dump(predictions, fhand)
            
            for i in xrange(epoch, opts.max_epochs):
                print('Epoch {}'.format(i+1))
                accuracy_level, loss, predictions = model.run_epoch(session)
                test_accuracy, loss, predictions = model.run_epoch(session, True)
                test_accuracy = np.mean(predictions[X_test!=0] == model.loader.y_test.reshape(-1)[X_test!=0])
                predictions[X_test==0] = 0
                nb_unknowns = 0
                nb_correct = 0.0
                for unknown in unknowns:
                    comp = predictions[X_test==unknown] == model.loader.y_test.reshape(-1)[X_test==unknown]
                    nb_unknowns += comp.size
                    nb_correct += np.sum(comp)
                
                print('test loss {}'.format(loss))
                print('test accuracy {}'.format(test_accuracy))
                print('we found {} unknown words out of {} words in the test set'.format(nb_unknowns, nb_words))
                print('unknown word accuracy {}'.format(nb_correct/nb_unknowns))
                
                if plot:
                    model.plot_losses()
                if best_accuracy < test_accuracy:
                    best_accuracy = test_accuracy 
                    if not saving_dir:
                        saving_file = os.path.join(opts.model_dir, 'epoch{0}_accuracy{1:.5f}_unknown{2:.5f}.weights'.format(i+1, test_accuracy, nb_correct/nb_unknowns))
                    else: 
                        saving_file = os.path.join(saving_dir, 'epoch{0}_accuracy{1:.5f}_unknown{2:.5f}.weights'.format(i+1, test_accuracy, nb_correct/nb_unknowns))
                    print('saving it to {}'.format(saving_file))
                    saver.save(session, saving_file)
                    bad_times = 0
                    print('test accuracy improving')
                else:
                    bad_times += 1
                    print('test accuracy deteriorating')
                if bad_times >= opts.early_stopping:
                    print('did not improve {} times in a row. stopping early'.format(bad_times))
                    #if saving_dir:
                    #    print('outputting test pred')
                    #    with open(os.path.join(saving_dir, 'predictions_test.pkl'), 'wb') as fhand:
                    #        pickle.dump(predictions, fhand)
                    break 
                
def run_model_k_fold(opts, jk_opts = None, modelname=None, epoch=0, best_accuracy=0, get_contexts = None):
    print('run k-fold on vanilla pos')


    if jk_opts:
        setattr(jk_opts, 'jackknife', 1)
        setattr(jk_opts, 'jackknife_dim', 0)
        jk_opts.task = 'Super_models'
        data_loader = Dataset(jk_opts)
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(jk_opts.seed)
            model = LSTM_Model(jk_opts, data_loader)
            saver = tf.train.Saver(max_to_keep=10)
            output_predictions = []
            K = 10 
            for k in xrange(K):
                print('{}th hold out'.format(k+1))
                model.loader.set_k()
                with tf.Session() as session: 
                    session.run(tf.initialize_all_variables())
                    best_accuracy = 0
                    bad_times = 0  
                    X_test = model.loader.X_train_k_fold[k]
                    X_test = X_test[:, jk_opts.window_size:jk_opts.window_size+model.seq_length].reshape(-1) # get rid of window padding before and after each sentence
                    unknowns = list(set(X_test)-set(model.loader.X_train_k.flat))
                    y_test = model.loader.y_train_k_fold[k]
                    nb_words = np.sum(model.loader.X_test != 0)
                    for i in xrange(jk_opts.max_epochs):
                        print('Epoch {}'.format(i+1))
                        accuracy_level, loss, predictions = model.run_epoch(session)
                        test_accuracy, loss, predictions = model.run_epoch(session, True)
                        test_accuracy = np.mean(predictions[X_test!=0] == y_test.reshape(-1)[X_test!=0])
                        predictions[X_test==0] = 0 
                    # need to change this in a way that test it on the held out set
                        
                        nb_unknowns = 0
                        nb_correct = 0.0

                        for unknown in unknowns:
                            comp = predictions[X_test==unknown] == y_test.reshape(-1)[X_test==unknown]
                            nb_unknowns += comp.size
                            nb_correct += np.sum(comp)
                        
                        print('test loss {}'.format(loss))
                        print('test accuracy {}'.format(test_accuracy))
                        print('we found {} unknown words out of {} words in the test set'.format(nb_unknowns, nb_words))
                        if not nb_unknowns==0:
                            unknown_accuracy = nb_correct/nb_unknowns
                            print('unknown word accuracy {}'.format(unknown_accuracy))
                        else: 
                            unknown_accuracy = 1.0
                           
                        if best_accuracy < test_accuracy:
                            output_prediction = predictions
                            best_accuracy = test_accuracy 
                            saving_file = os.path.join('..', 'k_fold_sec', '{0}th_hold_out_{1:.5f}_{2:.5f}.weights'.format(k+1, test_accuracy, unknown_accuracy))
                            print('saving it to {}'.format(saving_file))
                            saver.save(session, saving_file)
                            bad_times = 0
                            print('test accuracy improving')
                        else:
                            bad_times += 1
                            print('test accuracy deteriorating')
                        if bad_times >= jk_opts.early_stopping:
                            print('did not improve {} times in a row. stopping early'.format(bad_times))
                            output_predictions.append(output_prediction)
                            break 
                        # for debugging comment it out later
                        #output_predictions.append(output_prediction)
                        #break 
            with open('../k_fold_sec/predictions.pkl', 'wb') as fhand:
                pickle.dump(output_predictions, fhand)

            print('k-fold vanilla pos ended') 
    data_loader = Dataset(opts)
    
    data_loader.change_to_supertagging(opts)
    print('starting super-tagging with jackknife')
    run_model(opts, loader=data_loader, plot=False, modelname=modelname, epoch=epoch, best_accuracy=best_accuracy, get_contexts=get_contexts)
    
    
    #with g_super.as_default():
       


    #return np.hstack(output_predictions).reshape(model.seq_length, -1)
        

