import numpy as np
import random
#import matplotlib.pyplot as plt
import h5py as hf
import pickle

rnum = np.random.randint

'''
    Provides functions used in ice trainer
'''

class IceSet():
    def __init__(self, dataset_path):
        self.dataset_path= dataset_path

        self.icestates_dsname = 'LOOPSET.h5'
        self.trajects_dsname = 'trajectorys.pickle'
        self.actions_dsname = 'actions.pickle'

        self.hfile = hf.File('/'.join( [self.dataset_path, 
                            self.icestates_dsname]), 'r')

        with open('/'.join([self.dataset_path, self.actions_dsname]) , 'rb') as handle:
            self.LOOP_ACTIONS = pickle.load(handle)

        with open('/'.join([self.dataset_path, self.trajects_dsname]) , 'rb') as handle:
            self.LOOPS = pickle.load(handle)

        self.S1 = self.hfile['S1']
        self.S2 = self.hfile['S2']
        self.TMAP = self.hfile['TMAP']

        # important
        self.dnum = len(self.S1)

        # data specs
        self.L = 32
        self.N = self.L ** 2
        self.J = 1.0
        # 1/kT
        self.beta = 1e4

        # max steps each episode is 64
        self.MAX_STEPS = 2 * self.L

        # Note: configuration stored in 2D

        # s1 used for current state which is major env

        # From dataset
        self.s0 = None
        self.s1 = None
        self.s2 = None

        self.sup_map = None
        self.sdown_map = None

        self.eng_map = None

        # called traj in dataset, closed forming a loop
        self.loop = None
        self.loop_index = 0
        self.loop_len = 0

        # called actions in dataset, actions in loop
        self.loop_actions = None
        self.loop_action_index = 0
        self.loop_action_len = 0
        self.loop_map = None

        self.traj = []
        self.traj_map = None

        # agent position
        self.agent = (0,0)
        self.agent_map = None
        self.start_pos = None

        self.actions = None # in string

        self.eng_diff_counter = 0.0
        self.running_counter = 0
    
    def init(self):
        self.agent_map = np.zeros((self.L, self.L))
        self.traj_map = np.zeros((self.L, self.L))
        self.sup_map = np.zeros((self.L, self.L))
        self.sdown_map = np.zeros((self.L, self.L))
        self.eng_map = np.zeros((self.L, self.L))
    
    def reset_maps(self):
        self.traj = []
        self.loop_action_index = 0
        self.loop_action_len = 0
        self.loop_len = 0
        self.loop_index = 0
        self.agent_map = np.zeros((self.L, self.L))
        self.traj_map = np.zeros((self.L, self.L))
        self.sup_map = np.zeros((self.L, self.L))
        self.sdown_map = np.zeros((self.L, self.L))
        self.eng_map = np.zeros((self.L, self.L))
        self.running_counter = 0
        self.eng_diff_counter = 0.0
    
    def reset_trajectory(self):
        self.traj = []
        self.traj_map = np.zeros((self.L, self.L))

    def fetch(self):
        '''
            Fetch teaching material from dataset
        '''
        # fetch s1, s2 and create loop from dataset
        self.reset_trajectory()
        self.reset_maps()
        index = rnum(self.dnum)
        self.s1 = self.S1[index]
        self.s0 = self.S1[index]
        self.sup_map = self.s1.clip(min=0)
        self.sdown_map = np.abs(self.s1.clip(max=0))
        self.loop = self.LOOPS[index]
        self.loop_actions = self.LOOP_ACTIONS[index]
        self.loop_action_len = len(self.loop_actions)
        self.loop_len = len(self.loop)

    def _pdb(self, s, d):
        return ((s+d)%self.L + self.L) % self.L

    def flip_on_site(self):
        # Calculate current site 
        spin = self.s1[self.agent]
        self.s1[self.agent] *= -1
        se = self.cal_site_energy(self.s1, self.agent)
        self.eng_diff_counter += se
        self.traj.append(self.agent)
        #print ('Flip on site {} with spin {} and se = {}'.format(self.agent, self.s1[self.agent], se))
        if (spin > 0):
            # + --> -
            self.sup_map[self.agent] = 0
            self.sdown_map[self.agent] = 1
        else:
            # - --> +
            self.sup_map[self.agent] = 1
            self.sdown_map[self.agent] = 0

        # recalculate eng along traj
        for t in self.traj:
            se = self.cal_site_energy(self.s1, t)
            # normalization
            self.eng_map[t] = (se / 6.0 / self.J)

    def cal_site_energy(self, s, site):
        i, j = site[0], site[1]
        se = 0.0
        ip = self._pdb(i,+1)
        im = self._pdb(i,-1)
        jp = self._pdb(j,+1)
        jm = self._pdb(j,-1)
        if ((i+j) % 2 == 0):
            se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
            + s[ip][jm] + s[im][jp])
        else:
            se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
            + s[im][jm] + s[ip][jp])
        se *= self.J * s[i,j]
        return se

    def cal_energy(self, s):
        eng = 0.0
        J = 1.0
        for j in range(self.L):
            for i in range(self.L):
                se = 0.0
                ip = self._pdb(i,+1)
                im = self._pdb(i,-1)
                jp = self._pdb(j,+1)
                jm = self._pdb(j,-1)
                if ((i+j) % 2 == 0):
                    se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
                            + s[ip][jm] + s[im][jp])
                else:
                    se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
                            + s[im][jm] + s[ip][jp])
                se *= self.J*s[i,j]
                eng += se
        #eng = eng / float(self.L**2)
        return eng 

    def cal_defect_density(self):
        dd = 0.0
        for j in range(self.L):
            for i in range(self.L):
                ip = self._pdb(i,+1)
                im = self._pdb(i,-1)
                jp = self._pdb(j,+1)
                jm = self._pdb(j,-1)
                if ((i+j) % 2 == 0):
                    dd += (float(self.s1[i][j]+self.s1[ip][j]+self.s1[i][jp]+self.s1[ip][jp]))
        dd /= float(self.L**2)
        return dd
    
    def cal_energy_diff(self):
        return (self.cal_energy(self.s1) - self.cal_energy(self.s0))
    
    def metropolis(self):
        is_accept = False
        dE = self.cal_energy_diff()
        # or dE = self.diff_eng_counter
        dd = self.cal_defect_density()
        boltzman_weight = np.exp(-dE * self.beta)
        dice = np.random.uniform()
        if dE == 0 and dd == 0 and (self.agent == self.start_pos) :
            is_accept = True
        #elif boltzman_weight > dice:
        #    is_accept = True

        res = 'Accept' if is_accept else 'Reject'
        print ('[{}] dE = {}, dd = {} and BW = {}'.format(res, dE, dd, boltzman_weight))
        print ('dE counter = {}'.format(self.eng_diff_counter))
        return is_accept

    def pop_loop_actions(self):
        if self.loop_actions:
            if (self.loop_action_index < self.loop_action_len):
                a = self.loop_actions[self.loop_action_index]
                self.loop_action_index += 1
                return a
            else:
                return 'noop'
        else:
            return None

    def pop_loop(self):
        if self.loop:
            if (self.loop_index < self.loop_len):
                p = self.loop[self.loop_index]
                self.loop_index += 1
                return p
            else:
                return (-1, -1)
        else:
            return None

    def act(self, action):
        reward = 0.0
        done = False

        x, y = self.agent
        xp = self._pdb(x,+1)
        xm = self._pdb(x,-1)
        yp = self._pdb(y,+1)
        ym = self._pdb(y,-1)

        if action == 'metropolis':
            is_accept = self.metropolis()
            if (is_accept and len(self.traj) >= 4):
                reward = 1.0
            else:
                reward = -1.0
            done = True
        elif action == 'flip_up':
            newpos = (x, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_up':
            newpos = (x, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_down':
            newpos = (x, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_down':
            newpos = (x, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_left':
            newpos = (xm, y)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_left':
            newpos = (xm, y)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_right':
            newpos = (xp, y)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_right':
            newpos = (xp, y)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_upperright':
            newpos = (xp, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_upperright':
            newpos = (xp, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_upperleft':
            newpos = (xm, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_upperleft':
            newpos = (xm, ym)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_lowerleft':
            newpos = (xm, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_lowerleft':
            newpos = (xm, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'flip_lowerright':
            newpos = (xp, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
            self.flip_on_site()
        elif action == 'go_lowerright':
            newpos = (xp, yp)
            self.set_agent_site(newpos)
            self.traj_map[newpos] = 1
        elif action == 'noop':
            pass
        #print ('current agent position {}'.format(self.agent))
        self.running_counter += 1
        return reward, done

    @property
    def get_start_site(self):
        return self.start_pos
    
    def set_start_site(self, pos):
        # Need boundary check
        self.start_pos = pos

    def get_agent_site(self):
        return self.agent

    def set_agent_site(self, p):
        if (p[0] < self.L and p[1] < self.L):
            self.agent_map[self.agent] = 0
            self.agent = p
            self.agent_map[self.agent] = 1
        else:
            print ('Illegal agent position assignment!')
    
    def get_config(self):
        return self.s1
    
    def get_config_difference(self):
        return self.s0 - self.s1
    
    def get_state_map(self):
        state = np.copy(self.s1)
        state=state[state<0]=0
        return state
    
    def get_agent_map(self):
        return self.agent_map
    
    def get_traj(self):
        return self.traj

    def get_traj_map(self):
        return self.traj_map
    
    def get_eng_map(self):
        return self.eng_map
    
    def get_spinup_map(self):
        return self.sup_map
    
    def get_spindown_map(self):
        return self.sdown_map
    
    def get_stacked_map(self):
        return np.stack([self.sup_map,
                        self.agent_map,
                        self.traj_map,
                        self.eng_map
        ])
    
    def plot_obs_maps(self):
        #get so many maps
        # import matplot
        pass
    
    def action_mask(self):
        mask = np.ones(9)
    
    @property
    def is_closed(self):
        is_done = False
        if (self.agent == self.start_pos) and self.running_counter > 1:
            is_done = True
        return is_done

    @property
    def timeout(self):
        return True if self.running_counter >= self.MAX_STEPS else False