#! /usr/bin/env python

import unittest
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator, LSTMPolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker

## ice game
import gym
import gym_icegame

from constants import *

tf.flags.DEFINE_string("model_dir", MODEL_DIR, "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", ENV, "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_integer("t_max", T_MAX, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", MAX_GLOBAL_STEPS, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", EVAL_EVERY, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", RESET, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", PARALLELISM, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

def make_env(wrap=True):
    env = gym.make(FLAGS.env)
    return env

env_ = make_env()
VALID_ACTIONS = list(range(env_.action_space.n))

# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
    NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
#CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists('.'.join([MODEL_DIR])):
    os.makedirs('.'.join([MODEL_DIR]))
#if not os.path.exists(MODEL_DIR):
#    os.makedirs(MODEL_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

print('[TR] checkpoint_dir: {}'.format(CHECKPOINT_DIR))
print('[TR] model_dir: {}'.format(MODEL_DIR))
print ('[TR] Valid action space size = {}'.format(VALID_ACTIONS))
print ('[TR] {} Workers in Use'.format(NUM_WORKERS))

with tf.device(DEVICE):

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:

        if LSTM_POLICY:
            policy_net = LSTMPolicyEstimator(num_outputs=len(VALID_ACTIONS))
        else:
            policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS))
        value_net = ValueEstimator(reuse=True)

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for worker_id in range(NUM_WORKERS):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer

        worker = Worker(
            name="worker_{}".format(worker_id),
            env=make_env(wrap=False),
            policy_net=policy_net,
            value_net=value_net,
            global_counter=global_counter,
            discount_factor = 0.99,
            summary_writer=worker_summary_writer,
            max_global_steps=FLAGS.max_global_steps)
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=5)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
        env=make_env(wrap=False),
        policy_net=policy_net,
        summary_writer=summary_writer,
        saver=saver)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord, FLAGS.t_max)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval task
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)
