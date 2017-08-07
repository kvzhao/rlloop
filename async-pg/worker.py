import gym
import gym_icegame

import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

import pickle

from constants import SUMMARY_EACH_STEPS, LSTM_POLICY

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

# from lib import plotting
from estimators import ValueEstimator, PolicyEstimator, LSTMPolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops

def make_train_op(local_estimator, global_estimator):
    """
    Creates an op that applies local estimator gradients
    to the global estimator.
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
                    global_step=tf.contrib.framework.get_global_step())


class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
        name: A unique name for this worker
        env: The Gym environment used by this worker
        policy_net: Instance of the globally shared policy net
        value_net: Instance of the globally shared value net
        global_counter: Iterator that holds the global step
        discount_factor: Reward discount factor
        summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
        max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """
    def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            if LSTM_POLICY:
                self.policy_net = LSTMPolicyEstimator(policy_net.num_outputs)
            else:
                self.policy_net = PolicyEstimator(policy_net.num_outputs)
            self.value_net = ValueEstimator(reuse=True)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            # self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
            self.state = self.env.reset()
            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, local_t, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess)

            except tf.errors.CancelledError:
                return

    def run_n_steps(self, n, sess):
        transitions = []
        if LSTM_POLICY:
            self.policy_net.reset_lstm_features()
        for _ in range(n):
            # Take a step
            if LSTM_POLICY:
                action_probs = self.policy_net.action_inference(self.state)
            else:
                action_probs = self.policy_net.action_prediction(self.state)

            # eps-greedy action
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self.env.step(action)
            # next_state = atari_helpers.atari_make_next_state(self.state, self.sp.process(next_state))

            # Store transition
            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 1000 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                # self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
                self.state = self.env.reset()
                ### reset features
                if LSTM_POLICY:
                    self.policy_net.reset_lstm_features()
                break
            else:
                self.state = next_state
        return transitions, local_t, global_t

    def update(self, transitions, sess):
        """
        Updates global policy and value networks based on collected experience

        Args:
            transitions: A list of experience transitions
            sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        if not transitions[-1].done:
            reward = self.value_net.predict_value(transitions[-1].next_state)
            #reward = self._value_net_predict(transitions[-1].next_state, sess)
        
        if LSTM_POLICY:
            init_lstm_state = self.policy_net.get_init_features()

        # Accumulate minibatch exmaples
        states = []
        policy_targets = []
        value_targets = []
        actions = []
        features = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            policy_target = (reward - self.value_net.predict_value(transition.state))
            #policy_target = (reward - self._value_net_predict(transition.state, sess))
            # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        if LSTM_POLICY:
            feed_dict = {
                self.policy_net.states: np.array(states),
                self.policy_net.targets: policy_targets,
                self.policy_net.actions: actions,
                self.policy_net.state_in[0]: np.array(init_lstm_state[0]),
                self.policy_net.state_in[1]: np.array(init_lstm_state[1]),
                self.value_net.states: np.array(states),
                self.value_net.targets: value_targets,
            }
        else:
            feed_dict = {
                self.policy_net.states: np.array(states),
                self.policy_net.targets: policy_targets,
                self.policy_net.actions: actions,
                self.value_net.states: np.array(states),
                self.value_net.targets: value_targets,
            }

        # Train the global estimators using local gradients
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op,
            self.policy_net.summaries,
            self.value_net.summaries
        ], feed_dict)

        # Write summaries
        if self.summary_writer is not None and global_step % SUMMARY_EACH_STEPS == 0:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
