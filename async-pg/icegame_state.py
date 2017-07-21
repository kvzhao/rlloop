# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym
import gym_icegame

from constants import ACTION_SIZE

class GameState(object):
  def __init__(self, rand_seed, display=False):
    self.env = gym.make('IceGameEnv-v0')

    ## start here?
    init_site = 100
    self.env.start(init_site)
    self.reset()

  def _process_frame(self, action):
    timeout = self.env.timeout()
    #x_t, reward, done, info = self.env.step(action)
    x_t, reward, done, info = self.env.step(action)
    terminal = timeout or done
    # x_t *= (1.0/255.0)
    return reward, terminal, x_t

  def reset(self):
    self.env.reset()

    _, _, self.s_t = self._process_frame(7)

    self.reward = 0
    self.terminal = False
    #self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

  def process(self, action):
    # convert original 18 action index to minimal action set index

    r, t, self.s_t1 = self._process_frame(action)

    self.reward = r
    self.terminal = t
    #self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)

  def update(self):
    self.s_t = self.s_t1
