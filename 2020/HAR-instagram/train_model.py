
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import argparse

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--it', type=int, default=0, help='iter of crossval')
parser.add_argument('--gpu', type=int, default=0, help='gpu number')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--decay_epoch', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
#parser.add_argument('--model_file', type=str, default='modelfile/sample')

args = parser.parse_args()

args.model_file = f'modelfile/crossval_{args.it}'

log_file = os.path.join(f'train_log_{args.it}')
log = open(log_file, 'w')
log_string(log, str(args))
log_string(log, f"tensorflow version: {tf.__version__}")


def accuracy(test, pred):
    return accuracy_score(np.argmax(test, axis=1), np.argmax(pred, axis=1))


sorted(os.listdir('data'))


log_string(log, 'loading_data...')
df_img = pd.read_csv('data/100000_plus_labeled_scaled_pca_normalized_image_encoded_instagram.csv')
df_txt = pd.read_csv('data/100000_plus_labeled_scaled_text_doc2vec_instagram.csv')

df_label = pd.read_json('data/label_df.json')


unlabel_scodes = df_img[~df_img[df_img.columns[0]].isin(df_label['shortcode'])][df_img.columns[0]]


###

img_feat = dict()
scodes = df_img[df_img.columns[0]]
vals = df_img.values[:, 1:]
for i in range(len(df_img)):
    img_feat[scodes[i]] = vals[i]

txt_feat = dict()
scodes = df_txt[df_txt.columns[0]]
vals = df_txt.values[:, 1:]
for i in range(len(df_txt)):
    txt_feat[scodes[i]] = vals[i]


###

unlabel_X = []
for scode in unlabel_scodes:
    feat1 = img_feat[scode]
    feat2 = txt_feat[scode]
    unlabel_X.append([feat1, feat2])
unlabel_X = np.array(unlabel_X)
unlabel_X = unlabel_X.reshape(unlabel_X.shape[0], -1)


###

num_classes = 6

###

it = args.it
train_df = pd.read_csv(f'data/train_{it}_instagram_label.csv')
train_df = train_df[train_df['category'] < num_classes]
test_df = pd.read_csv(f'data/test_{it}_instagram_label.csv')
test_df = test_df[test_df['category'] < num_classes]

train_scodes = train_df['shortcode']
train_categories = train_df['category']
test_scodes = test_df['shortcode']
test_categories = test_df['category']

train_X = []
for scode in train_scodes:
    feat1 = img_feat[scode]
    feat2 = txt_feat[scode]
    train_X.append([feat1, feat2])

test_X = []
for scode in test_scodes:
    feat1 = img_feat[scode]
    feat2 = txt_feat[scode]
    test_X.append([feat1, feat2])

train_X = np.array(train_X)
train_X = train_X.reshape(train_X.shape[0], -1)
train_y = train_categories

test_X = np.array(test_X)
test_X = test_X.reshape(test_X.shape[0], -1)
test_y = test_categories

###

trainvalY = np.eye(num_classes)[train_y]
trainvalX = train_X[:, :300]
trainvalZ = train_X[:, 300:]

###

num_trainval = trainvalX.shape[0]
permutation = np.random.permutation(num_trainval)
trainvalX = trainvalX[permutation]
trainvalZ = trainvalZ[permutation]
trainvalY = trainvalY[permutation]

###

trainX = trainvalX[:int(num_trainval*0.85)]
valX = trainvalX[int(num_trainval*0.85):]
trainZ = trainvalZ[:int(num_trainval*0.85)]
valZ = trainvalZ[int(num_trainval*0.85):]
trainY = trainvalY[:int(num_trainval*0.85)]
valY = trainvalY[int(num_trainval*0.85):]

###

testX = test_X[:, :300]
testZ = test_X[:, 300:]
testY = np.eye(num_classes)[test_y]

###

# ros = RandomOverSampler(random_state=42)
# trainXZ = np.concatenate((trainX, trainZ), axis=-1)
# trainXZ, trainY = ros.fit_resample(trainXZ, trainY)
# trainX, trainZ = trainXZ[:, :300], trainXZ[:, 300:]

###

unlabelX = unlabel_X[:, :300]
unlabelZ = unlabel_X[:, 300:]

###

import math
import time, datetime
import numpy as np
import tensorflow as tf

#class DotDict(dict):
#    __getattr__ = dict.__getitem__
#    __setattr__ = dict.__setitem__
#    __delattr__ = dict.__delitem__   
#args = DotDict()
    
D = 300

X = tf.compat.v1.placeholder(
    shape = (None, D), dtype = tf.float32, name = 'X')
Z = tf.compat.v1.placeholder(
    shape = (None, D), dtype = tf.float32, name = 'Z')
label = tf.compat.v1.placeholder(
    shape = (None, num_classes), dtype = tf.float32, name = 'label')
is_training = tf.compat.v1.placeholder(
    shape = (), dtype = tf.bool, name = 'is_training')

num_train, _ = trainX.shape

global_step = tf.Variable(0, trainable = False)
global_step2 = tf.Variable(0, trainable = False)
bn_momentum = tf.compat.v1.train.exponential_decay(
    0.5, global_step,
    decay_steps = args.decay_epoch * num_train // args.batch_size,
    decay_rate = 0.5, staircase = True)

bn_decay = tf.minimum(0.99, 1 - bn_momentum)

dropout = tf.keras.layers.Dropout(.05)
XX = dropout(tf.compat.v1.layers.dense(X, units=1024, activation=tf.nn.leaky_relu ))
XX = dropout(tf.compat.v1.layers.dense(XX, units=512, activation=tf.nn.leaky_relu ))
#XX = dropout(tf.compat.v1.layers.dense(XX, units=256, activation=tf.nn.leaky_relu ))
XX = tf.compat.v1.layers.dense(XX, units=64, activation=None)

ZZ = dropout(tf.compat.v1.layers.dense(Z, units=1024, activation=tf.nn.leaky_relu ))
ZZ = dropout(tf.compat.v1.layers.dense(ZZ, units=512, activation=tf.nn.leaky_relu ))
#ZZ = dropout(tf.compat.v1.layers.dense(ZZ, units=256, activation=tf.nn.leaky_relu ))
ZZ = tf.compat.v1.layers.dense(ZZ, units=64, activation=None)

GX = tf.compat.v1.layers.dense(XX, units=64)
GZ = tf.compat.v1.layers.dense(ZZ, units=64)
z = tf.nn.sigmoid(tf.add(GX, GZ))
H = tf.add(tf.multiply(z, XX), tf.multiply(1 - z, ZZ))
H = tf.compat.v1.layers.dense(H, units=num_classes)

pred = tf.nn.softmax(H)

#kl = tf.keras.losses.KLDivergence()
#loss = kl(label, pred)
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(label, pred)

# predsq = pred**2
# predsq /= tf.reduce_sum(predsq, axis=1)
# usloss = 0.01*cce(predsq, pred)


#kl2 = tf.keras.losses.KLDivergence()
#loss2 = 0.01*kl2(label, pred)
cce2 = tf.keras.losses.CategoricalCrossentropy()
loss2 = 0.15* cce2(label, pred)

tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
tf.compat.v1.add_to_collection('loss2', loss2)

learning_rate = tf.compat.v1.train.exponential_decay(
    args.learning_rate, global_step,
    decay_steps = args.decay_epoch * num_train // args.batch_size,
    decay_rate = 0.7, staircase = True)
learning_rate = tf.maximum(learning_rate, 1e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step = global_step)

learning_rate2 = tf.compat.v1.train.exponential_decay(
    args.learning_rate * 0.01, global_step,
    decay_steps = args.decay_epoch * num_train // args.batch_size,
    decay_rate = 0.7, staircase = True)
learning_rate2 = tf.maximum(learning_rate2, 1e-5)
optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate2)
train_op2 = optimizer.minimize(loss2, global_step = global_step2)


parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])

log_string(log, 'trainable parameters: {:,}'.format(parameters))
log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = f"{args.gpu}"
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
log_string(log, '**** training model ****')
num_val = valX.shape[0]
num_unlabel = unlabelX.shape[0]
wait = 0
val_loss_min = np.inf
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(trainX.shape[0])
    trainX = trainX[permutation]
    trainZ = trainZ[permutation]
    trainY = trainY[permutation]
    
    permutation2 = np.random.permutation(unlabelX.shape[0])
    unlabelX_p = unlabelX[permutation2][:10000]
    unlabelZ_p = unlabelZ[permutation2][:10000]
    
    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: trainX[start_idx : end_idx],
            Z: trainZ[start_idx : end_idx],
            label: trainY[start_idx : end_idx],
            is_training: True}
        _, loss_batch = sess.run([train_op, loss], feed_dict = feed_dict)
        train_loss += loss_batch * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()
    
    # unlabel loss
    unlabelPred = []
    num_batch = math.ceil(num_unlabel / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_unlabel, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: unlabelX_p[start_idx : end_idx],
            Z: unlabelZ_p[start_idx : end_idx],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        unlabelPred.append(pred_batch)
    unlabelPred = np.concatenate(unlabelPred, axis = 0)
    #unlabelPred = np.argmax(unlabelPred, axis=-1)
    #unlabelY = np.eye(num_classes)[unlabelPred]
    unlabelY = unlabelPred**2
    an_array = unlabelY
    sum_of_rows = an_array.sum(axis=1)
    unlabelY = an_array / sum_of_rows[:, np.newaxis]
    
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_unlabel / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_unlabel, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: unlabelX_p[start_idx : end_idx],
            Z: unlabelZ_p[start_idx : end_idx],
            label: unlabelY[start_idx : end_idx],
            is_training: True}
        _, loss_batch = sess.run([train_op2, loss2], feed_dict = feed_dict)
        train_loss += loss_batch * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()
    
    
    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            X: valX[start_idx : end_idx],
            Z: valZ[start_idx : end_idx],
            label: valY[start_idx : end_idx],
            is_training: False}
        loss_batch = sess.run(loss, feed_dict = feed_dict)
        val_loss += loss_batch * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    log_string(log, '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
          args.max_epoch, end_train - start_train, end_val - start_val))
    log_string(log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    if val_loss <= val_loss_min:
        log_string(log, 'val loss decrease from %.4f to %.4f, saving model to %s' %
              (val_loss_min, val_loss, args.model_file))
        wait = 0
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
    else:
        wait += 1
        
    
    # num_test = testX.shape[0]
    # testPred = []
    # num_batch = math.ceil(num_test / args.batch_size)
    # start_test = time.time()
    # for batch_idx in range(num_batch):
    #     start_idx = batch_idx * args.batch_size
    #     end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    #     feed_dict = {
    #         X: testX[start_idx : end_idx],
    #         Z: testZ[start_idx : end_idx],
    #         is_training: False}
    #     pred_batch = sess.run(pred, feed_dict = feed_dict)
    #     testPred.append(pred_batch)
    # end_test = time.time()
    # testPred = np.concatenate(testPred, axis = 0)
    # test_acc = accuracy(testY, testPred)
    # log_string(log, f'test_acc: {test_acc}')
        

# test model
log_string(log, '**** testing model ****')
log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
log_string(log, 'model restored!')
log_string(log, 'evaluating...')
num_test = testX.shape[0]
trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: trainX[start_idx : end_idx],
        Z: trainZ[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    trainPred.append(pred_batch)
trainPred = np.concatenate(trainPred, axis = 0)
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: valX[start_idx : end_idx],
        Z: valZ[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    valPred.append(pred_batch)
valPred = np.concatenate(valPred, axis = 0)
testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: testX[start_idx : end_idx],
        Z: testZ[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    testPred.append(pred_batch)
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)


# train_acc = accuracy(trainY, trainPred)
# val_acc = accuracy(valY, valPred)
test_acc = accuracy(testY, testPred)
log_string(log, 'testing time: %.1fs' % (end_test - start_test))
# log_string(log, '                MAE\t\tRMSE\t\tMAPE')
# log_string(log, 'train acc: %.3f'%train_acc)
# log_string(log, 'val acc: %.3f'%val_acc)
log_string(log, 'test acc: %.3f'%test_acc)
log_string(log, 'performance in each prediction step')
sess.close()
log.close()