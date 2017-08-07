import os

# NAMES
TASK='vanilla'
ENV = 'IceGameEnv-v0'

# PATH
MODEL_DIR = 'logs'
MODEL_DIR = '/'.join([MODEL_DIR, TASK])
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# TRAINING PROCESS
T_MAX = 4
MAX_GLOBAL_STEPS = None
EVAL_EVERY = 600
SUMMARY_EACH_STEPS = 10000

LSTM_POLICY = True
RESET = False

PARALLELISM = 8
DEVICE = "/cpu:0"

# HYPER-PARAMETERS
BETA_ENTROPY = 0.01
LEARNING_RATE = 0.001
MOMENTUM = 0.0
