"""
Alternative implementation of the DCGRU recurrent cell in Tensorflow 2
References
----------
Paper: https://arxiv.org/abs/1707.01926
Original implementation: https://github.com/liyaguang/DCRNN
Inherits this repository: https://github.com/mensif/DCGRU_Tensorflow2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations

# from lib.matrix_calc import *
# The majority of the present code originally comes from
# https://github.com/liyaguang/DCRNN/blob/master/lib/utils.py

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def build_sparse_matrix(L):
    L = L.astype('float32')
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    return tf.sparse.reorder(L)




class MFGCGRU(keras.layers.Layer):
    def __init__(self, units, num_nodes, adj_mats=[], ext_feats=[], **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(MFGCGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]

        self.built = True


    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]
        
        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate): 
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for i, support in enumerate(self.adj_mats):
            # premultiply the concatened inputs and state with support matrices
            x_support = support@x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5 # @ tf.transpose(FE)
            support = tf.nn.softmax(support) 

            x_support = support@x0
            x_support = tf.expand_dims(x_support, 0) 
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)


class MFGCGRU_S(keras.layers.Layer):
    def __init__(self, units, SE, num_nodes, adj_mats=[], ext_feats=[], **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats
        self.num_nodes = num_nodes
        self.SE = SE
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(MFGCGRU_S, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FC_sent_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu'),
                            tf.keras.layers.Dense(1, activation='relu')]) for i in range(len(self.ext_feats))]

        self.built = True


    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]
        
        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate): 
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
            
        for i, support in enumerate(self.adj_mats):
            # premultiply the concatened inputs and state with support matrices
            x_support = support@x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            FES = tf.squeeze(self.FC_sent_feat[i](tf.concat((self.SE, feat), -1)), -1)
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5 # @ tf.transpose(FE)
            # support = tf.nn.softmax(support) + FES
            support_exp = tf.exp(support)
            support_exp_sum = tf.reduce_sum(support_exp, -1)
            support = support_exp / tf.expand_dims((FES + support_exp_sum), -1)

            x_support = support@x0 #+ sent @ x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)




class MFGCGRU_SM(keras.layers.Layer):
    def __init__(self, units, SE, num_nodes, adj_mats=[], ext_feats=[], **kwargs):
        self.D = self.units = units
        self.state_size = units * num_nodes
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats
        self.num_nodes = num_nodes
        self.SE = SE
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(MFGCGRU_SM, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) #* self.num_mx

        self.r_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FC_sent_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu'),
                            tf.keras.layers.Dense(1, activation='relu')]) for i in range(len(self.ext_feats))]

        self.built = True


    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]
        
        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate): 
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
                    
        for i, support in enumerate(self.adj_mats):
            # premultiply the concatened inputs and state with support matrices
            x_support = support@x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            FES = tf.squeeze(self.FC_sent_feat[i](tf.concat((feat, self.SE), -1)), -1)
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5

            support_exp = tf.exp(support)
            support_exp_sum = tf.reduce_sum(support_exp, -1)
            support = support_exp / tf.expand_dims((FES + support_exp_sum), -1)

            x_support = support@x0 
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)


        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size, self.num_mx])

        xsave = x
        xs = []
        for i in range(self.num_mx):
            x = xsave[..., i]
            if gate == 'reset':
                x = tf.matmul(x, self.r_kernel[i])
                x = tf.nn.bias_add(x, self.r_bias[i])
            elif gate == 'update':
                x = tf.matmul(x, self.u_kernel[i])
                x = tf.nn.bias_add(x, self.u_bias[i])
            elif gate == 'candidate':
                x = tf.matmul(x, self.c_kernel[i])
                x = tf.nn.bias_add(x, self.c_bias[i])
            else:
                print('Error: Unknown gate')

            xs.append(x)
        xs = tf.stack(xs, -1)
        x = tf.reduce_mean(xs, -1)

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)


class MFGCGRU_M(keras.layers.Layer):
    def __init__(self, units, num_nodes, adj_mats=[], ext_feats=[], **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(MFGCGRU_M, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) #* self.num_mx

        self.r_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.num_mx, self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.num_mx, self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]

        self.built = True


    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]
        
        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate): 
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        
        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)        
            
        for i, support in enumerate(self.adj_mats):
            # premultiply the concatened inputs and state with support matrices
            x_support = support@x0
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5 # @ tf.transpose(FE)
            support = tf.nn.softmax(support)

            x_support = support@x0 
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size, self.num_mx])

        xsave = x
        xs = []
        for i in range(self.num_mx):
            x = xsave[..., i]
            if gate == 'reset':
                x = tf.matmul(x, self.r_kernel[i])
                x = tf.nn.bias_add(x, self.r_bias[i])
            elif gate == 'update':
                x = tf.matmul(x, self.u_kernel[i])
                x = tf.nn.bias_add(x, self.u_bias[i])
            elif gate == 'candidate':
                x = tf.matmul(x, self.c_kernel[i])
                x = tf.nn.bias_add(x, self.c_bias[i])
            else:
                print('Error: Unknown gate')

            xs.append(x)
        xs = tf.stack(xs, -1)
        x = tf.reduce_mean(xs, -1)

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)



class DCGRUCell(keras.layers.Layer):
    def __init__(self, units, adj_mx, K_diffusion, num_nodes, filter_type, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.K_diffusion = K_diffusion
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(DCGRUCell, self).__init__(**kwargs)
        self.supports = []
        supports = []
        # the formula describing the diffsuion convolution operation in the paper
        # corresponds to the filter "dual_random_walk"
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":           
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self.supports.append(build_sparse_matrix(support))
            sup0 = support
            for k in range(2, self.K_diffusion + 1):
                sup0 = support.dot(sup0)                  # (original paper version)
                # sup0 = 2 * support.dot(sup0) - sup0     # (author's repository version)
                self.supports.append(build_sparse_matrix(sup0))

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.supports)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for support in self.supports:
            # premultiply the concatened inputs and state with support matrices
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)