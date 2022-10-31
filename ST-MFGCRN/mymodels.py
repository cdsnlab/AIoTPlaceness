import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *
from MFGCGRU_cell import *
from DeepSTN_net import *
from STResNet import *

def row_normalize(an_array):
    sum_of_rows = an_array.sum(axis=1)
    normalized_array = an_array / sum_of_rows[:, np.newaxis]
    return normalized_array


class MySTMFGCRN(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MySTMFGCRN, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        self.sentinel = args.sentinel
        self.mgcn = args.mgcn
        self.fusion = args.fusion

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC_in = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(D)])
        self.FC_XP_in = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(D)])
        self.FC_XQ_in = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(D)])


        if self.sentinel == 'x' and self.mgcn == 'cat':
            self.FC_XC_DCGRU = MFGCGRU(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XP_DCGRU = MFGCGRU(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XQ_DCGRU = MFGCGRU(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
        elif self.sentinel == 'o' and self.mgcn == 'cat':
            self.FC_XC_DCGRU = MFGCGRU_S(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XP_DCGRU = MFGCGRU_S(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XQ_DCGRU = MFGCGRU_S(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
        elif self.sentinel == 'x' and self.mgcn == 'mean':
            self.FC_XC_DCGRU = MFGCGRU_M(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XP_DCGRU = MFGCGRU_M(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XQ_DCGRU = MFGCGRU_M(units=self.D,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
        elif self.sentinel == 'o' and self.mgcn == 'mean':
            self.FC_XC_DCGRU = MFGCGRU_SM(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XP_DCGRU = MFGCGRU_SM(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
            self.FC_XQ_DCGRU = MFGCGRU_SM(units=self.D,SE=SE,num_nodes=self.num_nodes,adj_mats=self.adj_mats,ext_feats=self.ext_feats)
        else:
            print('ERROR')
        

        if self.fusion == 'weight':
            self.CPTF = CPTFusion(D, self.out_dim)
        elif self.fusion == 'add':
            self.FC_XC_out = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(self.out_dim)])
            self.FC_XP_out = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(self.out_dim)])
            self.FC_XQ_out = keras.Sequential([layers.Dense(D, activation="relu"), layers.Dense(self.out_dim)])
        else:
            print('ERROR')

        
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
                
        XC = self.FC_XC_in(XC) + STEC
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        
        XP = self.FC_XP_in(XP) + STEP
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        
        XQ = self.FC_XQ_in(XQ) + STEQ
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))

        if self.fusion == 'add':
            XC = self.FC_XC_out(XC)
            XP = self.FC_XP_out(XP)
            XQ = self.FC_XQ_out(XQ)
            Y = XC + XP + XQ
        elif self.fusion == 'weight':
            Y = self.CPTF(XC, XP, XQ)
        else:
            print('ERROR')

        return Y


class MyDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.adj_mat = row_normalize(extdata['adj_mat'])
        self.out_dim = args.out_dim
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.out_dim)])

    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        dayofweek = tf.one_hot(TEC[..., 0], depth = 7)
        timeofday = tf.one_hot(TEC[..., 1], depth = 24)
        minuteofday = tf.one_hot(TEC[..., 2], depth = 4)
        holiday = tf.one_hot(TEC[..., 3], depth = 1)
        TEC = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TEC = tf.expand_dims(TEC, axis = 2)

        TEC = tf.tile(TEC, (1, 1, self.num_nodes, 1))
        XC = tf.concat((XC, TEC), -1)

        XC = self.FC_XC_in(XC)
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(XC)

        return Y

class MyGMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.out_dim = args.out_dim

        f = open('node2vec/SE.txt', mode = 'r')
        lines = f.readlines()
        temp = lines[0].split(' ')
        N, dims = int(temp[0]), int(temp[1])
        SE = np.zeros(shape = (N, dims), dtype = np.float32)
        for line in lines[1 :]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = temp[1 :]
        self.SE = SE
        
    def build(self, input_shape):
        D = self.D
            
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        
        XC = self.FC_XC_in(XC)
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        XC = self.FC_XC_out(XC)
        Y = tf.squeeze(XC, 1)
        return Y
    
    
class MyConvLSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyConvLSTM, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T

        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.Conv2D_C = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 3),
                padding="same",
                return_sequences=False,
            )
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        
        XC = self.FC_XC0(XC) + STEC
        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, self.D))
        XC = self.Conv2D_C(XC)
        XC = tf.reshape(XC, (-1, self.H_height*self.H_width, self.D))
        XC = self.converter_mat.T @ XC
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        Y = self.FC_XC(XC)

        return Y


class MyDeepSTN():
    def __init__(self, extdata, args):
        from tensorflow.keras import backend as K
        K.set_image_data_format('channels_first')
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.out_dim = args.out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))
        
        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T
        

    def __call__(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        TE = tf.cast(TEY, tf.int32)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.tile(TE, (1, 1, self.H_height, self.H_width))

        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, 2))
        XP = tf.reshape((self.converter_mat @ XP), (-1, P_num, self.H_height, self.H_width, 2))
        XQ = tf.reshape((self.converter_mat @ XQ), (-1, Q_num, self.H_height, self.H_width, 2))

        ####################### DSTN ##################
        H=29
        W=14
        channel=2 #H-map_height W-map_width channel-map_channel
        c=6
        p=7
        t=3 #c-closeness p-period t-trend
        pre_F=64
        conv_F=64
        R_N=2 #pre_F-prepare_conv_featrue conv_F-resnet_conv_featrue R_N-resnet_number
        is_plus=True             #use ResPlus or mornal convolution
        is_plus_efficient=False  #use the efficient version of ResPlus
        plus=8
        rate=2            #rate-pooling_rate
        is_pt=False               #use PoI and Time or not
        P_N=0
        if len(self.ext_feats) == 1:
            is_pt = True
            self.poi_local = self.ext_feats[0]
            is_pt=True
            P_N=self.poi_local.shape[-1]
        T_F=28
        PT_F=6
        T=TE.shape[1] #P_N-poi_number T_F-time_feature PT_F-poi_time_feature T-T_times/day 
        drop=0
        is_summary=True #show detail
        # lr=0.0002
        kernel1=1 #kernel1 decides whether early-fusion uses conv_unit0 or conv_unit1, 1 recommended
        isPT_F=1 #isPT_F decides whether PT_model uses one more Conv after multiplying PoI and Time, 1 recommended

        
        all_channel = channel * (c+p+t)
                
        cut0 = int( 0 )
        cut1 = int( cut0 + channel*c )
        cut2 = int( cut1 + channel*p )
        cut3 = int( cut2 + channel*t )
        
        c_input = tf.reshape(tf.transpose(XC, (0, 1, 4, 2, 3)), (-1, C_num*2, self.H_height, self.H_width))
        p_input = tf.reshape(tf.transpose(XP, (0, 1, 4, 2, 3)), (-1, P_num*2, self.H_height, self.H_width))
        t_input = tf.reshape(tf.transpose(XQ, (0, 1, 4, 2, 3)), (-1, Q_num*2, self.H_height, self.H_width))
        
        from tensorflow.keras.layers import Conv2D
        K.set_image_data_format('channels_first')
        c_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(c_input)
        p_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(p_input)
        t_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(t_input)
                
        if is_pt:
            poi_in = tf.reshape(self.converter_mat @ self.poi_local, (self.H_height, self.H_width, -1))
            poi_in = tf.transpose(poi_in, (2, 0, 1))
            poi_in = tf.cast(poi_in, tf.float32)
            poi_in = tf.expand_dims(poi_in, 0)
            poi_in = tf.tile(poi_in, (tf.shape(XC)[0], 1, 1, 1))
            # T_times/day + 7days/week 
            # time_in=Input(shape=(T+7,H,W))
            time_in = TE

            PT_model=PT_trans('PT_trans',P_N,PT_F,T,T_F,H,W,isPT_F)
            
            poi_time=PT_model([poi_in,time_in])
    
            cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1,poi_time])
            if kernel1:
                cpt=conv_unit1(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
            else:
                cpt=conv_unit0(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
        
        else:
            cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1])
            if kernel1:
                cpt=conv_unit1(pre_F*3,conv_F,drop,H,W)(cpt_con1)
            else:
                cpt=conv_unit0(pre_F*3,conv_F,drop,H,W)(cpt_con1)  


        
        if is_plus:
            if is_plus_efficient:
                for i in range(R_N):
                    cpt=Res_plus_E('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)
            else:
                for i in range(R_N):
                    cpt=Res_plus('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)

        else:  
            for i in range(R_N):
                cpt=Res_normal('Res_normal_'+str(i+1),conv_F,drop,H,W)(cpt)

        cpt_conv2=Activation('relu')(cpt)
        cpt_out2=cpt_conv2 # cpt_out2=BatchNormalization()(cpt_conv2)
        cpt_conv1=Dropout(drop)(cpt_out2)
        cpt_conv1=Conv2D(filters=channel,kernel_size=(1, 1),padding="same")(cpt_conv1)
        # cpt_out1=Activation('tanh')(cpt_conv1)
        cpt_out1 = cpt_conv1

                
        print('***** pre_F : ',pre_F       )
        print('***** conv_F: ',conv_F      )
        print('***** R_N   : ',R_N         )
        
        print('***** plus  : ',plus*is_plus)
        print('***** rate  : ',rate*is_plus)
        
        print('***** P_N   : ',P_N*is_pt   )
        print('***** T_F   : ',T_F*is_pt   )
        print('***** PT_F  : ',PT_F*is_pt*isPT_F )            
        print('***** T     : ',T           ) 
        
        print('***** drop  : ',drop        )
        
        print(cpt_out1.shape)
        
        Y = cpt_out1
        Y = tf.reshape(tf.transpose(Y, (0, 2, 3, 1)), (-1, self.H_height*self.H_width, self.out_dim))
        Y = self.converter_mat.T @ Y
        return Y



# from __future__ import print_function
from tensorflow.keras.layers import (
    Input,
    Activation,
    # merge,
    add,
    Dense,
    Reshape,
    BatchNormalization
)
from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # return merge([input, residual], mode='sum')
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        # return Conv2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
        
        return Conv2D(filters=nb_filter, kernel_size=(nb_row,nb_col), strides=subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


class Conv2DFilterPool(tf.keras.layers.Layer):
    def __init__(self, num_outputs, use_bias=True, padding="same"):
        super(Conv2DFilterPool, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        if self.use_bias:
            self.offset = self.add_weight(
                shape=[1,1,1,self.num_outputs], initializer="random_normal", 
                trainable=True, dtype=tf.float32, name='offset'
            )
        self.hex_filter = tf.constant( np.array([
                [ 0,  1,  0],
                [ 1,  0,  1],
                [ 0,  1,  0],
                [ 1,  0,  1],
                [ 0,  1,  0]
            ]).reshape([5,3,1,1]) , dtype=tf.float32 )
        self.kernel = self.add_weight(
            shape=[5,3,int(input_shape[-1]), self.num_outputs], 
            initializer=tf.keras.initializers.RandomNormal(stddev=1./7.), #"random_normal", 
            trainable=True, dtype=tf.float32, name='kernel'
        ) * self.hex_filter

    def call(self, input):
        conv = tf.keras.backend.conv2d(input, self.kernel * self.hex_filter, 
                            data_format="channels_last", padding=self.padding)
        if self.use_bias:
            return tf.nn.tanh( conv + self.offset )
        else:
            return tf.nn.tanh( conv )


class MySTResNet():
    def __init__(self, extdata, args):
        from tensorflow.keras import backend as K
        K.set_image_data_format('channels_first')
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.out_dim = args.out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T
        

    def __call__(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        TE = tf.cast(TEY, tf.int32)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.tile(TE, (1, 1, self.H_height, self.H_width))

        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, 2))
        XP = tf.reshape((self.converter_mat @ XP), (-1, P_num, self.H_height, self.H_width, 2))
        XQ = tf.reshape((self.converter_mat @ XQ), (-1, Q_num, self.H_height, self.H_width, 2))
        
        XC = tf.reshape(tf.transpose(XC, (0, 1, 4, 2, 3)), (-1, C_num, 2, self.H_height, self.H_width))
        XP = tf.reshape(tf.transpose(XP, (0, 1, 4, 2, 3)), (-1, P_num, 2, self.H_height, self.H_width))
        XQ = tf.reshape(tf.transpose(XQ, (0, 1, 4, 2, 3)), (-1, Q_num, 2, self.H_height, self.H_width))

        # XC = XC[:, -3:, ...]
        # XP = XP[:, -3:, ...]
        # XQ = XQ[:, -3:, ...]

        # c_conf=XC.shape[1:]
        # p_conf=XP.shape[1:]
        # t_conf=XQ.shape[1:]
        external_dim=8
        nb_residual_unit=3
        CF=64
        nb_flow_out = 2

        # main input
        main_inputs = []
        outputs = []
        # for conf, input in zip([c_conf, p_conf, t_conf], [XC, XP, XQ]):
        for inp in [XC, XP, XQ]:
            # if conf is not None:
            len_seq, nb_flow, map_height, map_width = inp.shape[1:]
            inp = tf.reshape(inp, (-1, nb_flow * len_seq, map_height, map_width))
            # input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            # main_inputs.append(input)
            # Conv1
            # conv1 = Conv2D(
            #     filters=CF, kernel_size=(5, 3), padding="same")(inp)
            inp = tf.transpose(inp, (0, 2, 3, 1))
            conv1 = Conv2DFilterPool(CF)(inp)
            conv1 = tf.transpose(conv1, (0, 3, 1, 2))

            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=CF,
                            repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            # conv2 = Conv2D(
            #     filters=nb_flow_out, kernel_size=(5,3), padding="same")(activation)
            
            activation = tf.transpose(activation, (0, 2, 3, 1))
            conv2 = Conv2DFilterPool(nb_flow_out)(activation)
            conv2 = tf.transpose(conv2, (0, 3, 1, 2))
            outputs.append(conv2)

        # parameter-matrix-based fusion
        if len(outputs) == 1:
            main_output = outputs[0]
        else:
            from DST_network_ilayer import iLayer
            new_outputs = []
            for output in outputs:
                print('output', output.shape)
                n_output = iLayer()(output)
                # n_output = output
                new_outputs.append(n_output)
                print('n_output', n_output.shape)
            # main_output = merge(new_outputs, mode='sum')
            main_output = add(new_outputs)

        # # fusing with external component
        # if external_dim != None and external_dim > 0:
        #     # external input
        #     external_input = Input(shape=(external_dim,))
        #     main_inputs.append(external_input)
        #     embedding = Dense(10)(external_input)
        #     embedding = Activation('relu')(embedding)
        #     h1 = Dense(nb_flow_out * map_height * map_width)(embedding)
        #     activation = Activation('relu')(h1)
        #     external_output = Reshape((nb_flow_out, map_height, map_width))(activation)
        #     # main_output = merge([main_output, external_output], mode='sum')
        #     print('main,external', main_output.shape, external_output.shape)
        #     main_output = add([main_output, external_output])
        # else:
        #     print('external_dim:', external_dim)

        # main_output = Activation('tanh')(main_output)
        # model = Model(main_inputs, main_output)

        print(main_output.shape)
        Y = main_output
        Y = tf.reshape(tf.transpose(Y, (0, 2, 3, 1)), (-1, self.H_height*self.H_width, self.out_dim))
        Y = self.converter_mat.T @ Y
        print('Y', Y.shape)
        return Y



####################################################
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def ModelSet(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    return (model(kwargs) ) * extdata['max_values'] # +0.5
