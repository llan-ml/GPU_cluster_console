# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:53:31 2016

@author: lanlin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("user_model_file", "example_user_model.py",
                    "The filename of user model.")
flags.DEFINE_float("learning_rate", 0.01,
                   "Initial learnig rate. Default is 0.01.")
flags.DEFINE_integer("max_step", 1e5,
                     """The max steps to run min-batch training. """
                     """Default is 1e5.""")
flags.DEFINE_integer("batch_size", 128,
                     "Batch size in a singe GPU. Default is 128.")
flags.DEFINE_string("optimizer", "sgd",
                    "Optimizer. Default is sgd.")
flags.DEFINE_integer("sync", 0,
                     """"Synchronous training" or "Asynchronous training". """
                     """Default is "Asynchronous training".""")
flags.DEFINE_string("checkpoint_dir", "checkpoints/",
                    "Directory where to write checkpoints.")
flags.DEFINE_string("tensorboard_dir", "tensorboard",
                    "Directory where to write event logs.")
flags.DEFINE_integer("summary_period", 30,
                     "Seconds to save a summary.")

FLAGS.max_step = int(FLAGS.max_step)

user_model_imported = os.path.splitext(FLAGS.user_model_file)[0]
user_model = __import__(user_model_imported)

os.system("mkdir -p {}".format(FLAGS.checkpoint_dir))
os.system("mkdir -p {}".format(FLAGS.tensorboard_dir))

def trainer(cost, global_step):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss=cost, global_step=global_step)
    return train_op
    

def main(argv=None):        
    cluster_spec = eval(os.environ.get("CLUSTER_SPEC"))
    job_name = os.environ.get("JOB_NAME")
    task_index = int(os.environ.get("TASK_INDEX"))
    
    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)
                             
    if job_name == "ps":
        print("Current process id: {}".format(os.getpid()))
        server.join()
    elif job_name == "worker":
        print("Current process id: {}".format(os.getpid()))
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:{}/task:{}".format(job_name, task_index),
            cluster=cluster)):
            
            # Build model...
            x, y_, network = user_model.inference(FLAGS)
                            
            # Calculate loss...
            loss = user_model.calc_loss(y_, network.outputs)
            correct_prediction = tf.equal(tf.arg_max(network.outputs, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            tf.summary.scalar("train_loss", loss)
            val_loss = tf.placeholder(tf.float32, shape=(), name="val_loss")
            tf.summary.scalar("val_loss", val_loss)
            val_accuracy = tf.placeholder(tf.float32, shape=(), name="val_accuracy")
            tf.summary.scalar("val_accuracy", val_accuracy)
            
            # Get train operation...
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = trainer(loss, global_step)
            
            init_op = tf.global_variables_initializer()
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver(sharded=True)
            
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                     init_op=init_op,
                                     logdir=FLAGS.checkpoint_dir,                                     
                                     summary_op=None,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=60,
                                     summary_writer=None)
                                     
            with sv.managed_session(server.target) as sess:
                if task_index == 0:
                    print("Save tensorboard files into: {}.".format(
                        FLAGS.tensorboard_dir))
                    writer = tf.summary.FileWriter(FLAGS.tensorboard_dir,
                                                sess.graph)

                step = 0
                feed_dict_generator = user_model.get_feed_dict(
                    x, y_, network, FLAGS)
                if task_index == 0:
                    next_summary_time = time.time() + FLAGS.summary_period
                while not sv.should_stop() and step < FLAGS.max_step:
                    this_feed_dict = feed_dict_generator.next()
                    _, step = sess.run([train_op, global_step],
                             feed_dict=this_feed_dict)
                        
                    if task_index == 0 \
                        and next_summary_time < time.time():
                        this_val_accuracy = []
                        this_val_loss = []
                        for val_feed_dict in user_model.get_validate_data(
                                                            x, y_, FLAGS):
                            this_acc, this_loss = sess.run(
                               [accuracy, loss], feed_dict=val_feed_dict)
                            this_val_accuracy.append(this_acc)
                            this_val_loss.append(this_loss)
                        this_val_accuracy = np.mean(this_val_accuracy)
                        this_val_loss = np.mean(this_val_loss)
                        print("{} {}".format(this_val_accuracy, this_val_loss))
                        summary_feed_dict = {
                            val_loss: this_val_loss.astype(np.float32),
                            val_accuracy: this_val_accuracy.astype(np.float32)}
                        summary_feed_dict.update(this_feed_dict)
                        summary_value, step = sess.run(
                            [summary_op, global_step],
                            feed_dict=summary_feed_dict)
                        writer.add_summary(summary_value, step)
                        writer.flush()
                        next_summary_time += FLAGS.summary_period
            if task_index == 0:
                writer.close()
                        
if __name__ == "__main__":
    tf.app.run()
