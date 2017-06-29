from __future__ import division
import gym
from gym import error, spaces, utils, core
#from six import StingIO
import sys
import numpy as np
import random
from ice_set import IceSet

rnum = np.random.randint

'''
    This gym loads icestates from dataset and working as trainer, containing two stages
    1. Teaching Phase
        provides action and states from dataset
    2. Learning Phase
        interact with a new state (from dataset, but new situation)
'''

class IceTrainerEnv(core.Env):
    def __init__ (self, dataset_path):
        self.L = 32
        # Where should dataset path be accessed is appropriate?
        self.dataset_path = dataset_path
        self.iceset = IceSet(self.dataset_path)
        
        self.next_round_need_reset = False

        self.name_mapping = dict({
                                  0 :   'right',       
                                  1 :   'down',
                                  2 :   'left',
                                  3 :   'up', 
                                  4 :   'upper_right',
                                  5 :   'upper_left',
                                  6 :   'lower_left',
                                  7 :   'lower_right', 
                                  8 :   'metropolis',
                                  9 :   'noop'
                                  })

        self.index_mapping = dict({
                                  'right': 0,       
                                  'down' : 1,
                                  'left' : 2,
                                  'up' : 3, 
                                  'upper_right' : 4,
                                  'upper_left' : 5,
                                  'lower_left' : 6,
                                  'lower_right' : 7, 
                                  'metropolis' : 8,
                                  'noop' : 9
                                  })
        
        ### action space and state space
        self.action_space = spaces.Discrete(len(self.name_mapping))
        self.observation_space = spaces.Box(np.zeros(self.L**2 * 4), \
                                            np.ones(self.L**2  * 4))
        self.reward_range = (-1, 1)
        
        ### action space and state space
        self.action_space = spaces.Discrete(len(self.name_mapping))
        self.observation_space = spaces.Box(np.zeros(self.L**2 * 4), \
                                            np.ones(self.L**2  * 4))
        self.reward_range = (-1, 1)

    def step(self, action):
        '''
            s, a --> s', r
            1. get state from iceset
            2. act action from agent
            3. get observation 
            4. capture reward and wheter done or not
        '''
        terminate = False
        reward = 0.0
        obs = None
        done =False
        #print ('Receive command {}'.format(action))

        if (self.next_round_need_reset):
            #print ('next round reset')
            self.next_round_need_reset = False
            terminate = True
            obs =  self.get_obs()
            return obs, reward, terminate

        if type(action) is int:
            if (0 <= action < 9):
                action = self.name_mapping[action]
            else:
                print ('Illegal action value!')
                action = 0
        
        reward, done = self.iceset.act(action)
        #print ('execute act')
        obs =  self.get_obs()

        if (done):
            terminate = True
            self.next_round_need_reset = True

        return obs, reward, terminate

    # Start function used for agent learing
    def start(self, is_guiding=True):
        self.reset()
        self.iceset.init()
        self.iceset.fetch()
        if is_guiding:
            start = self.iceset.pop_loop()
            self.iceset.set_agent_site(start)
            self.iceset.set_start_site(start)
            #print ('Guide start from site {}'.format(self.agent_site))
            #self.iceset.flip_on_site()
        else:
            start = (rnum(self.L), rnum(self.L))
            self.iceset.set_agent_site(start)
            self.iceset.set_start_site(start)
            #print ('Start from site {}'.format(self.agent_site))
    
    def reset(self):
        self.next_round_need_reset = False
        self.iceset.reset_maps()
    
    def sample_action(self):
        return self.iceset.pop_loop_actions()

    @property
    def agent_site(self):
        return self.iceset.get_agent_site()

    @property
    def action_name_mapping(self):
        return self.name_mapping

    @property
    def name_action_mapping(self):
        return self.index_mapping
    
    def render(self, mapname ='s1', mode='ansi', close=False):
        #of = StringIO() if mode == 'ansi' else sys.stdout
        print ('Energy: {}, Defect: {}'.format(self.iceset.cal_energy_diff(), self.iceset.cal_defect_density()))
        s = None
        if mapname == 's1':
            s = self.iceset.get_config()
        elif mapname == 'diff':
            s = self.iceset.get_config_difference()
        elif mapname == 'traj':
            s = self.iceset.get_traj_map()
        elif mapname == 'amap':
            s = self.iceset.get_agent_map()
        elif mapname == 'sup':
            s = self.iceset.get_spinup_map()
        elif mapname == 'sdown':
            s = self.iceset.get_spindown_map()

        screen = '\r'
        screen += '\n\t'
        screen += '+' + self.L * '---' + '+\n'
        for i in range(self.L):
            screen += '\t|'
            for j in range(self.L):
                p = (i, j)
                spin = s[p]
                if spin == -1:
                    screen += ' o '
                elif spin == +1:
                    screen += ' * '
                elif spin == 0:
                    screen += '   '
                elif spin == +2:
                    screen += ' @ '
                elif spin == -2:
                    screen += ' O '
            screen += '|\n'
        screen += '\t+' + self.L * '---' + '+\n'
        sys.stdout.write(screen)
    
    def get_obs(self):
        return self.iceset.get_stacked_map()

    @property
    def unwrapped(self):
        """Completely unwrap this env.
            Returns:
                gym.Env: The base non-wrapped gym.Env instance
        """
        return self