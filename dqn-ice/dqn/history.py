import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, screen_height, screen_width, screen_channels= \
        config.batch_size, config.history_length, config.screen_height, config.screen_width, config.screen_channels

    self.history = np.zeros(
        [history_length, screen_channels, screen_height, screen_width], dtype=np.float32)
    # can we use np.float16 here?, history is formated with float16

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
