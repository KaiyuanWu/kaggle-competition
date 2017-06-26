import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from random import shuffle


class DataIter():
    def __init__(self, train_x, train_y, val_x, val_y, test_x, batch_size = 128, random_shuffle=True):
        self.train_x_ = train_x
        self.train_y_ = train_y
        self.val_x_ = val_x
        self.val_y_ = val_y
        self.test_x_ = test_x
        self.ntrain_, self.nfeature_ = train_x.shape
        self.nval_, _ = val_x.shape
        self.ntest_, _ = test_x.shape
        self.ntotal_ = self.ntrain_ + self.nval_ + self.ntest_
        self.batch_size_ = batch_size

        self.data_ = np.zeros((self.ntotal_, self.nfeature_))
        self.label_ = np.zeros((self.ntotal_,1))
        self.weight_ = np.zeros((self.ntotal_,1))
        self.data_[:self.ntrain_] = self.train_x_
        self.label_[:self.ntrain_] = self.train_y_.reshape(-1,1)
        self.weight_[:self.ntrain_] = 1
        self.data_[self.ntrain_:self.ntrain_ + self.nval_] = self.val_x_
        self.label_[self.ntrain_:self.ntrain_ + self.nval_] = self.val_y_.reshape(-1,1)
        self.weight_[self.ntrain_:self.ntrain_ + self.nval_] = 0
        self.data_[self.ntrain_ + self.nval_:] = self.test_x_
        self.weight_[self.ntrain_ + self.nval_:] = 0

        self.index_ = range(self.ntotal_)
        if random_shuffle:
            shuffle(self.index_)

        self.cur_index_ = 0

    def __iter__(self):
        return self

    def next(self):
        n = self.batch_size_
        if self.cur_index_ + n < self.ntotal_:
            cur_index = self.cur_index_
            self.cur_index_ += n
            index = self.index_[cur_index:cur_index + n]
            return self.data_[index], self.label_[index], self.weight_[index]
        else:
            if self.cur_index_ < self.ntotal_:
                cur_index = self.cur_index_
                self.cur_index_ += n
                index = [self.index_[i] for i in
                    range(cur_index, self.ntotal_) + range(self.ntrain_ - 1, 2 * self.ntrain_ - n - 1 - cur_index)]
                return self.data_[index], self.label_[index], self.weight_[index]
            else:
                self.cur_index_ = 0
                raise StopIteration()

def fully_connected(data, num_outputs, weights_initializer, biases_initializer, layer_id=0, no_act=False, ifold=0):
    shape = data.shape
    weights = tf.get_variable("weight_%d_%d" % (ifold, layer_id), [shape[1], num_outputs],
                              initializer=weights_initializer) #tf.random_normal_initializer(0, 0.05))
    bias = tf.get_variable("bias_%d_%d" % (ifold, layer_id), [num_outputs], initializer=biases_initializer) #tf.constant_initializer(0))
    fc = tf.matmul(data, weights) + bias
    if not no_act:
        relu = tf.nn.relu(fc)
        return relu
    else:
        return fc


def ae_model(input_dim, pred_loss_weight=0.9, rec_loss_weight=0.1, ifold = 0):
    data = tf.placeholder(tf.float32, [None, input_dim])
    label = tf.placeholder(tf.float32, [None, 1])
    weight = tf.placeholder(tf.float32, [None, 1])
    phase = tf.placeholder(tf.bool)
    encoder_fc1 = fully_connected(data, input_dim, weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer = tf.constant_initializer(0), layer_id=0, ifold = ifold)
    encoder_fc2 = fully_connected(encoder_fc1, int(1.3*input_dim), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer = tf.constant_initializer(0), layer_id=1, ifold = ifold)
    encoder_fc3 = fully_connected(encoder_fc2, int(input_dim/16), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=2, ifold = ifold)
    encoder_fc4 = fully_connected(encoder_fc3, int(input_dim/32), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=3, ifold = ifold)

    encoder = fully_connected(encoder_fc4, int(input_dim/64), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=4, ifold = ifold)

    ypred = fully_connected(encoder, 1, weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), no_act=True, layer_id=5, ifold = ifold)
    pred_loss = 0.5 * tf.reduce_mean(tf.multiply(tf.square(ypred - label), weight))

    decoder_fc1 = fully_connected(encoder, int(input_dim/32), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=6, ifold = ifold)

    decoder_fc2 = fully_connected(decoder_fc1, int(input_dim/16), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=7, ifold = ifold)

    decoder = fully_connected(decoder_fc2, int(input_dim), weights_initializer=tf.random_normal_initializer(0, 0.05),
                                  biases_initializer=tf.constant_initializer(0), layer_id=8, ifold = ifold)

    rec_loss = 0.5 * tf.reduce_mean(tf.square(decoder - data))
    loss = pred_loss_weight * pred_loss + rec_loss_weight * rec_loss

    return data, label, weight, phase, pred_loss, rec_loss, loss, ypred, encoder, decoder




train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_y = 0.1*(train['y']-100.)
for c in train.columns:
    if train[c].dtype == 'object':
        lb = LabelBinarizer()
        lb.fit_transform(pd.concat((train[c], test[c])))
        num_classes, = lb.classes_.shape
        print(c, num_classes)
        train_tmp = lb.transform(train[c])
        test_tmp = lb.transform(test[c])
        train = train.drop([c], axis=1)
        test = test.drop([c], axis=1)

        for i in range(num_classes):
            train['%s_%d' % (c, i)] = train_tmp[:, i]
            test['%s_%d' % (c, i)] = test_tmp[:, i]

num_train_samples, num_features = train.shape
num_test_samples, _ = test.shape

#exclude column y ID
num_features -= 2

nfolds = 5
kf = KFold(n=num_train_samples, n_folds=nfolds, shuffle=True)
data_syms = []
label_syms = []
weight_syms = []
phase_syms = []
loss_syms = []
rec_loss_syms = []
pred_loss_syms = []
train_step_syms = []
encoder_syms = []
decoder_syms = []
ypred_syms = []
learning_rate = 0.002
for i in range(nfolds):
    with tf.device('/cpu:0'):
        data, label, weight, phase, pred_loss, rec_loss, loss, ypred, encoder, decoder = ae_model(num_features, ifold = i)
        data_syms.append(data)
        label_syms.append(label)
        weight_syms.append(weight)
        phase_syms.append(phase)
        loss_syms.append(loss)
        rec_loss_syms.append(rec_loss)
        pred_loss_syms.append(pred_loss)
        encoder_syms.append(encoder)
        decoder_syms.append(decoder)
        ypred_syms.append(ypred)
        train_step_syms.append(tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss))

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

num_epochs = 2000
for (train_index, val_index), ifold in zip(kf, range(nfolds)):
    dtrain = train.drop(['y','ID'], axis=1).iloc[train_index]
    dtrain_y = train_y.iloc[train_index]
    dval = train.drop(['y','ID'], axis=1).iloc[val_index]
    dval_y = train_y.iloc[val_index]
    dtest = test.drop(['ID'], axis=1)

    data_iter = DataIter(train_x=dtrain.values, train_y=dtrain_y.values, val_x=dval.values,
                         val_y=dval_y.values, test_x=dtest.values)

    for iepoch in range(num_epochs):
        ibatch = 0
        for data, label, weight in data_iter:
            ibatch += 1
            if ibatch%20 != 0:
                sess.run(train_step_syms[ifold],
                     feed_dict={data_syms[ifold]: data, label_syms[ifold]: label, weight_syms[ifold]: weight,
                                phase_syms[ifold]: True})
            else:
                _,loss,pred_loss, rec_loss = sess.run((train_step_syms[ifold], loss_syms[ifold], pred_loss_syms[ifold], rec_loss_syms[ifold]),
                         feed_dict={data_syms[ifold]: data, label_syms[ifold]: label, weight_syms[ifold]: weight,
                                    phase_syms[ifold]: True})
                print('batch #', ibatch, 'loss', loss, 'pred loss', pred_loss, 'rec loss', rec_loss)
        ypred = sess.run(ypred_syms[ifold], feed_dict={data_syms[ifold]: dval.values, phase_syms[ifold]: False})
        #print(ypred)
        print('Fold #', ifold, 'Epoch #', iepoch, 'val r2=', r2_score(dval_y.values, ypred))
