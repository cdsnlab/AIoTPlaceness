import os
import datetime
import pandas as pd
import fiona
import geopandas as gpd
from scipy.sparse import coo_matrix
import math
import time, datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import mymodels

def metric(pred, label):
    mae = np.abs(np.subtract(pred, label)).astype(np.float32)
    rmse = np.square(mae)
    mae = np.mean(mae)
    rmse = np.sqrt(np.mean(rmse))
    nmae = np.sum(np.abs(np.subtract(pred, label))) / np.sum(np.abs(label))
    return mae, rmse, nmae

def metric_print(pred, label):
    mae, rmse, nmae = metric(pred, label)
    return f"{mae:.3f}\t{rmse:.3f}\t{nmae:.3f}"

parser = argparse.ArgumentParser(description='CPT-train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--model_name', type=str, default='MySTMFGCRN')
parser.add_argument('--memo', type=str, default='')
parser.add_argument('--retrain', type=str, default='True')
parser.add_argument('--testcheck', type=str, default='False')
parser.add_argument('--dataset_path', type=str, default='process-qt-CPT-sample-190301-190310-190317-190324.pkl') # process-qt-CPT-190301-190501-190601-190701.pkl
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--K', type=int, default=8) # for GMAN
parser.add_argument('--d', type=int, default=8) # for GMAN
parser.add_argument('--L', type=int, default=3) # for GMAN
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('-n','--node', action='append', help='<Required> Set flag', required=False)
parser.add_argument('-e','--edge', action='append', help='<Required> Set flag', required=False)
parser.add_argument('--sentinel', type=str, default='o') # o, x
parser.add_argument('--mgcn', type=str, default='mean') # mean, cat
parser.add_argument('--fusion', type=str, default='weight') # weight, add


args = parser.parse_args()
args.test_pred = f"demo_{args.memo}_{args.model_name}"
if args.node:
    args.test_pred += '_n_'+'-'.join(sorted(args.node))
if args.edge:
    args.test_pred += '_e_'+'-'.join(sorted(args.edge))
if args.model_name == 'MySTMFGCRN':
    args.test_pred += '_sentinel_'+args.sentinel
    args.test_pred += '_mgcn_'+args.mgcn
    args.test_pred += '_fusion_'+args.fusion

args.model_checkpoint = f"checkpoint/{args.test_pred}/{args.test_pred}"

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
if not os.path.isdir(f'checkpoint/{args.test_pred}'):
    os.mkdir(f'checkpoint/{args.test_pred}')

import sys
newtest_dir = f'test/{args.dataset_path}'
if args.testcheck == 'True':
    if os.path.isfile(f'{newtest_dir}/pred_{args.test_pred}.npy'):
        sys.exit()

print(str(args))

with open(args.dataset_path, 'rb') as f:
    mdata = pickle.load(f)

pred = mdata['test']['XQ'].mean(axis=1)
label = mdata['test']['Y']
print('Trend Mean', metric_print(pred, label), sep='\t')

pred =  mdata['test']['XP'].mean(axis=1)
label = mdata['test']['Y']
print('Period Mean', metric_print(pred, label), sep='\t')

pred = mdata['test']['XC'].mean(axis=1)
label = mdata['test']['Y']
print('Closeness Mean', metric_print(pred, label), sep='\t')

pred = mdata['test']['XC'][:, -1, :, :]
label = mdata['test']['Y']
print('Last Repeat', metric_print(pred, label), sep='\t')

for k in list(mdata.keys() - {'train', 'val', 'test'}):
    try:
        print(k, mdata[k].shape)
    except:
        print(k, mdata[k])

extdata = {k: mdata[k] for k in mdata.keys() - {'train', 'val', 'test'}}


# input normalization
max_values = mdata['train']['Y'].max(0)
for k in ['train', 'val', 'test']:
    for w in ['XC', 'XP', 'XQ']:
        mdata[k][w] = mdata[k][w] / max_values 
extdata['max_values'] = max_values


num_nodes = mdata['train']['Y'].shape[1]
extdata['num_nodes'] = num_nodes

### model train ###
XC  = layers.Input(shape=mdata['train']['XC'].shape[1:], dtype=tf.float32)
TEC = layers.Input(shape=mdata['train']['TEC'].shape[1:], dtype=tf.float32)
XP  = layers.Input(shape=mdata['train']['XP'].shape[1:], dtype=tf.float32)
TEP = layers.Input(shape=mdata['train']['TEP'].shape[1:], dtype=tf.float32)
XQ  = layers.Input(shape=mdata['train']['XQ'].shape[1:], dtype=tf.float32)
TEQ = layers.Input(shape=mdata['train']['TEQ'].shape[1:], dtype=tf.float32)
TEY = layers.Input(shape=mdata['train']['TEY'].shape[1:], dtype=tf.float32)

def model_define():
    Y = mymodels.ModelSet(model_name=args.model_name, extdata=extdata, args=args, \
                                XC=XC, TEC=TEC, XP=XP, TEP=TEP, XQ=XQ, TEQ=TEQ, TEY=TEY)
    model = keras.models.Model((XC, TEC, XP, TEP, XQ, TEQ, TEY), Y)
    return model

model = model_define()

if args.retrain.lower() == 'false':
    try:
        print('model restore successful')
        model.load_weights(args.model_checkpoint)
    except:
        print('there exists no pretrained model')
        
model.summary()
optimizer = keras.optimizers.Adam(lr=args.learning_rate)
model.compile(loss=tf.keras.metrics.mean_absolute_error, 
              optimizer=optimizer)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(args.model_checkpoint, save_weights_only=True, \
                save_best_only=True, monitor='val_loss', mode='min', verbose=0, patience=args.patience)

def tuple_datasets(train):
    return  train['XC'], train['TEC'], \
            train['XP'], train['TEP'], \
            train['XQ'], train['TEQ'], train['TEY']


import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


model.fit(
    tuple_datasets(mdata['train']), mdata['train']['Y'],
    batch_size=args.batch_size,
    epochs=args.max_epoch,
    validation_data=(tuple_datasets(mdata['val']), mdata['val']['Y']),
    callbacks=[early_stopping, reduce_lr, model_ckpt, time_callback],
)

print('Elapsed time per epoch: ', '%.4f'%np.mean(time_callback.times[1:]) + ' sec.')

model = model_define()
model.load_weights(args.model_checkpoint)

pred = model.predict(tuple_datasets(mdata['test']))
label = mdata['test']['Y']
print(f'{args.model_name}', metric_print(pred, label), sep='\t')

newtest_dir = f'test/{args.dataset_path}'
if not os.path.isdir('test'):
    os.mkdir('test')
if not os.path.isdir(f'test/{args.dataset_path}'):
    os.mkdir(f'test/{args.dataset_path}')

np.save(f'{newtest_dir}/label.npy', label)
np.save(f'{newtest_dir}/pred_{args.test_pred}.npy', pred)

# import time
# time.sleep(60)
