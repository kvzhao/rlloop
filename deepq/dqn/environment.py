import gym
import gym_icegame 
import random
import numpy as np
from .utils import rgb2gray, imresize

class IceEnvironment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name)

    screen_width, screen_height, screen_channels, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.screen_channels, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_channels, screen_height, screen_width)
    self.state_dim = self.env.observation_space.shape[0]
    self.num_action = self.env.action_space.n

    self._screen = None
    self.reward = 0
    self.terminal = False

  def new_game(self, from_random_game=False):
    init_site = 100
    self.env.start(init_site)
    self._screen = self.env.get_obs()
    self.render()
    return self.screen, 0, 0, self.terminal

  def new_random_game(self):
    self.new_game(True)
    #for _ in xrange(random.randint(0, self.random_start - 1)):
    #  self._step(self.env.sample_action())
    return self.screen, 0, 0, self.terminal

  def _step(self, action, from_random_game=False):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def sample_action(self):
        return self.env.sample_icemove_action_index()

  def _random_step(self):
    action = self.env.sample_action()
    self._step(action, from_random_game=True)

  @ property
  def screen(self):
    return self._screen

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class GlobalIceEnvironment(IceEnvironment):
  def __init__(self, config):
    super(GlobalIceEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action, from_random_game=is_training)
    return self.state

