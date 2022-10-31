import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class STEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TEC, TEP, TEQ):        
        num_c = TEC.shape[-2]
        num_p = TEP.shape[-2]
        num_q = TEQ.shape[-2] 
        
        TE = tf.concat((TEC, TEP, TEQ), -2)
        
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        STE_C = STE[:, :num_c, ...]
        STE_P = STE[:, num_c:num_c+num_p, ...]
        STE_Q = STE[:, num_c+num_p:, ...]
        
        return STE_C, STE_P, STE_Q
    

class STEmbedding_Y(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEmbedding_Y, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TEC, TEP, TEQ, TEY):
        TEY = tf.expand_dims(TEY, axis=1)
        
        num_c = TEC.shape[-2]
        num_p = TEP.shape[-2]
        num_q = TEQ.shape[-2]
        num_y = TEY.shape[-2]
        
        TE = tf.concat((TEC, TEP, TEQ, TEY), -2)
        
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        STE_C = STE[:, : num_c, ...]
        STE_P = STE[:, num_c : num_c+num_p, ...]
        STE_Q = STE[:, num_c+num_p : num_c+num_p+num_q, ...]
        STE_Y = STE[:, num_c+num_p+num_q : , ...]
        
        return STE_C, STE_P, STE_Q, STE_Y
    
class CPTFusion(tf.keras.layers.Layer):
    def __init__(self, D, out_dim):
        super(CPTFusion, self).__init__()
        self.D = D
        self.out_dim = out_dim

    def build(self, input_shape):
        self.FC_C = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_P = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_Q = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.out_dim),])
        
    def call(self, XC, XP, XQ):
        ZC = self.FC_C(XC)
        ZP = self.FC_P(XP)
        ZQ = self.FC_Q(XQ)

        Z = tf.concat((ZC, ZP, ZQ), -1)
        Z = tf.nn.softmax(Z)
        return self.FC_H(Z[..., 0:1] * XC + Z[..., 1:2] * XP + Z[..., 2:] * XQ)

    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, use_mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.use_mask = use_mask

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        if self.use_mask:
            batch_size = tf.shape(X)[0]
            num_step = X.get_shape()[1].value
            N = X.get_shape()[2].value
            mask = tf.ones(shape = (num_step, num_step))
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
            mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
            mask = tf.cast(mask, dtype = tf.bool)
            attention = tf.compat.v2.where(
                condition = mask, x = attention, y = -2 ** 15 + 1)
            
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_S = keras.Sequential([
            layers.Dense(self.D, use_bias=False),])
        self.FC_T = keras.Sequential([
            layers.Dense(self.D),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),])
        
    def call(self, HS, HT):
        XS = self.FC_S(HS)
        XT = self.FC_T(HT)
        
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC_H(H)
        return H
    
class GSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(GSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialAttention(self.K, self.d)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SA_layer(X, STE)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H
    

class TransformAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    