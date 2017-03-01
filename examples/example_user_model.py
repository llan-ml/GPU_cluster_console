# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:20:12 2016

@author: lanlin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorlayer as tl

def get_feed_dict(x, y_, network, FLAGS):
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1),
                                    path="./data/")
    
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    start_idx = 0
    while True:
        excerpt = np.copy(indices[start_idx: start_idx + FLAGS.batch_size])
        start_idx += FLAGS.batch_size
        if start_idx > len(X_train) - FLAGS.batch_size:
            np.random.shuffle(indices)
            start_idx = 0
        feed_dict = {x: X_train[excerpt], y_: y_train[excerpt]}
#        feed.dict.update(network.all_drop)        
        yield feed_dict


def get_validate_data(x, y_, FLAGS):
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1), path="./data/")
    
    X_train = np.asarray(X_train, dtype=np.float32)    
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    for this_X_val, this_y_val in tl.iterate.minibatches(
                                        X_val, y_val,
                                        batch_size=FLAGS.batch_size,
                                        shuffle=True):
        feed_dict = {x: this_X_val, y_: this_y_val}
        yield feed_dict

def inference(FLAGS):
    x = tf.placeholder(tf.float32,
                       shape=[FLAGS.batch_size, 28, 28, 1],
                       name="x")
    y_ = tf.placeholder(tf.int64,
                        shape=[FLAGS.batch_size,],
                        name="y_")
    
    network = tl.layers.InputLayer(x, name="input_layer")
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 1, 32],
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    name="cnn_layer_0")
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME",
                                  pool=tf.nn.max_pool,
                                  name="pool_layer_0")
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 32, 64],
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    name="cnn_layer_1")
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME",
                                  pool=tf.nn.max_pool,
                                  name="pool_layer_1")
    network = tl.layers.FlattenLayer(network, name="flatten_layer")
#    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop_0")
    network = tl.layers.DenseLayer(network, n_units=256,
                                   act=tf.nn.relu, name="dense_layer_1")
#    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop_1")
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity, name="output_layer")
    return [x, y_, network]

def calc_loss(true, pred):
    return tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=pred, labels=true))