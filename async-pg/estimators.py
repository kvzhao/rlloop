import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tflib import *

def build_shared_head(X, add_summaries=False):
    print ('Use my shared head-network')

    conv1 = tf.nn.relu(conv2d(X, 8, "l1", [3, 3], [2, 2], pad="VALID"))
    conv2 = tf.nn.relu(conv2d(conv1, 12, "l2", [3, 3], [2, 2], pad="VALID"))
    conv2 = flatten(conv2)

    fc = tf.nn.relu(linear(conv2, 128, "fc", normalized_columns_initializer(0.01)))

    ## Add summary
    if add_summaries:
        activation_summary(conv1, name='conv1')
        activation_summary(conv2, name='conv2')
        activation_summary(fc, name='fc')

    return fc

class LSTMPolicyEstimator():
    def __init__(self, num_outputs, reuse=False, trainable=True):
        """
            LSTM Policy approximator
        """
        self.num_outputs = num_outputs
        # Icegame input are 4 channel frame with shape 32x32 each
        self.states = tf.placeholder(shape=[None, 32, 32, 4], dtype=tf.float32, name="X")

        # TD Targets
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        cell_size = 256

        X = self.states

        with tf.variable_scope("shared", reuse=reuse):
            fc = build_shared_head(X, add_summaries=(not reuse))

        with tf.variable_scope("policy_net"):
            # lstm 
            batch_size = tf.shape(self.states)[0]

            # introduce the fake batch size here
            x = tf.expand_dims(fc, [0])
            lstm = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.states)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size, time_major=False)
            
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, cell_size])

            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            self.logits = linear(x, num_outputs, "action", normalized_columns_initializer(0.01))
            self.probs = tf.nn.softmax(self.logits) + 1e-8

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
            }

            # We add entropy to the loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices) 

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram(self.entropy.op.name, self.entropy)

            if trainable:
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                    global_step=tf.contrib.framework.get_global_step())

        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)

    def get_init_features(self):
        return self.state_init

    def reset_lstm_features(self):
        #self.lstm_out_features = self.get_init_features()
        self.lstm_out_features = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))

    def action_inference (self, state):
        sess = tf.get_default_session()
        # should catch state_out
        feed_dict = {self.states: [state], self.state_in[0]: self.lstm_out_features[0], 
                                            self.state_in[1]: self.lstm_out_features[1]}
        fetched = sess.run([self.logits, self.probs] + self.state_out, feed_dict)
        logits, action_probs = fetched[0], fetched[1]
        action_probs = action_probs[0]
        self.lstm_out_features = fetched[2:]
        return action_probs

class PolicyEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
        num_outputs: Size of the action space.
        reuse: If true, an existing shared network will be re-used.
        trainable: If true we add train ops to the network.
            Actor threads that don't update their local models and don't need
            train ops would set this to false.
    """

    def __init__(self, num_outputs, reuse=False, trainable=True):
        self.num_outputs = num_outputs

        # Placeholders for our input
        self.states = tf.placeholder(shape=[None, 32, 32, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # Normalize
        # X = tf.to_float(self.states) / 255.0
        X = tf.to_float(self.states) 
        batch_size = tf.shape(self.states)[0]

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_head(X, add_summaries=(not reuse))


        with tf.variable_scope("policy_net"):
            self.logits = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8
            # probs := policy pi(a|s)

            self.predictions = {
                "logits": self.logits,
                "probs": self.probs
            }

            # We add entropy to the loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram(self.entropy.op.name, self.entropy)

            if trainable:
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                    global_step=tf.contrib.framework.get_global_step())

        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)


    def action_prediction (self, state):
        sess = tf.get_default_session()
        feed_dict = { self.states: [state] }
        preds = sess.run(self.predictions, feed_dict)
        return preds["probs"][0]


class ValueEstimator():
    """
    Value Function approximator. Returns a value estimator for a batch of observations.

    Args:
        reuse: If true, an existing shared network will be re-used.
        trainable: If true we add train ops to the network.
            Actor threads that don't update their local models and don't need
            train ops would set this to false.
    """

    def __init__(self, reuse=False, trainable=True):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        # Icegame input are 4 channel frame with shape 32x32 each
        self.states = tf.placeholder(shape=[None, 32, 32, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        # X = tf.to_float(self.states) / 255.0
        X = tf.to_float(self.states) 

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            fc1 = build_shared_head(X, add_summaries=(not reuse))

        with tf.variable_scope("value_net"):
            self.logits = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=1,
                activation_fn=None)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
            }

            # Summaries
            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss.name, self.loss)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                # self.optimizer = tf.train.AdamOptimizer(1e-4)
                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                    global_step=tf.contrib.framework.get_global_step())

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
    
    def predict_value (self, state):
        sess = tf.get_default_session()
        feed_dict = { self.states: [state] }
        preds = sess.run(self.predictions, feed_dict)
        return preds["logits"][0]
