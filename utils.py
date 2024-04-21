#DataLoader and one-hot
import torch
import numpy as np
import pandas as pd
import os
import operator
import pyprind
import gc

def one_hot(x, num_x, data_type='numpy', device=None):
    if data_type == 'numpy':
        res = np.zeros(num_x)
    elif data_type == 'torch':
        res = torch.zeros(num_x).to(device)
    res[x] = 1.0
    return res

class DataLoader(object):
    def __init__(self, encoded_data, rng, minibatch_size, drop_smaller_than_minibatch, device, str_id=""):
        '''
        If encoded_data is str, it is inferred as path-to-file name and will be loaded using pd
        If encoded_data is dict it is inferred as data and will be used directly
        Use: first call make_transition_train_data() once, then call reset() before each epoch of getting all minibatches
        '''
        self.rng = rng
        self.device = device
        self.str_id = str_id  # optional: used in logging
        self.minibatch_size = minibatch_size
        self.drop_smaller_than_minibatch = drop_smaller_than_minibatch
        self.ps = None
        self.ns = None
        if isinstance(encoded_data, str):
            self.encoded_data_file = os.path.abspath(encoded_data)
            self.encoded_data = pd.read_csv(self.encoded_data_file)
        elif isinstance(encoded_data, dict):
            self.encoded_data = encoded_data
        else:
            raise ValueError("Unknown encoded data.")
        self.transition_data = {}
        self.transition_indices_pos_last = []
        self.transition_indices_neg_last = []
        self.transition_indices = None
        self.transition_data_size = None
        self.transitions_head = None
        self.transitions_head_pos = None
        self.transitions_head_neg = None
        self.epoch_finished = True  # to enforce reset() before use
        self.epoch_pos_finished = True
        self.epoch_neg_finished = True
        self.num_minibatches_epoch = None
    
    def reset(self, shuffle, pos_samples_in_minibatch, neg_samples_in_minibatch):
        self.ps = pos_samples_in_minibatch
        self.ns = neg_samples_in_minibatch
        if shuffle:
            self.rng.shuffle(self.transition_indices)
            self.rng.shuffle(self.transition_indices_pos_last)
            self.rng.shuffle(self.transition_indices_neg_last)
        self.transitions_head = 0
        self.transitions_head_pos = 0
        self.transitions_head_neg = 0
        self.epoch_finished = False
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False
        self.num_minibatches_epoch = int(np.floor(self.transition_data_size / self.minibatch_size)) + int(1 - self.drop_smaller_than_minibatch)

    def make_transition_data(self, release=False):
        print("DataLoader: making transitions (s,a,r,s') " + self.str_id)
        self.transition_data['s'] = {}
        self.transition_data['actions'] = {}
        self.transition_data['rewards'] = {}
        self.transition_data['next_s'] = {}
        self.transition_data['terminals'] = {}
        indices_pos = []
        indices_neg = []
        counter = 0
        bar = pyprind.ProgBar(len(list(self.encoded_data['traj'].keys())))
        for traj in self.encoded_data['traj'].keys():
            bar.update()
            for t in range(self.encoded_data['traj'][traj]['actions'].shape[0] - 1):
                self.transition_data['s'][counter] = self.encoded_data['traj'][traj]['s'][t, :]
                self.transition_data['next_s'][counter] = self.encoded_data['traj'][traj]['s'][t+1, :]
                self.transition_data['actions'][counter] = self.encoded_data['traj'][traj]['actions'][t]
                self.transition_data['rewards'][counter] = self.encoded_data['traj'][traj]['rewards'][t]
                self.transition_data['terminals'][counter] = 0
                if traj in self.encoded_data['pos_traj']:
                    indices_pos.append(counter)
                else:
                    indices_neg.append(counter)
                counter += 1
            # For the last transition in the trajectory
            tlast = self.encoded_data['traj'][traj]['actions'].shape[0] - 1
            self.transition_data['s'][counter] = self.encoded_data['traj'][traj]['s'][tlast, :]
            self.transition_data['next_s'][counter] = np.zeros_like(self.encoded_data['traj'][traj]['s'][tlast, :])
            self.transition_data['actions'][counter] = self.encoded_data['traj'][traj]['actions'][tlast]
            self.transition_data['rewards'][counter] = self.encoded_data['traj'][traj]['rewards'][tlast]
            self.transition_data['terminals'][counter] = 1
            if traj in self.encoded_data['pos_traj']:
                self.transition_indices_pos_last.append(counter)
            else:
                self.transition_indices_neg_last.append(counter)
            counter += 1
        self.transition_data_size = counter
        self.transition_indices = np.arange(self.transition_data_size)
        if release:
            del self.encoded_data
            self.encoded_data = None
            gc.collect()
    
    def get_next_minibatch(self):
        if self.epoch_finished == True:
            print('Epoch finished, please call reset() method before next call to get_next_minibatch()')
            return None
        # Getting data from dictionaries
        offset = self.ns + self.ps
        minibatch_main_index_list = list(self.transition_indices[self.transitions_head:self.transitions_head + self.minibatch_size - offset])
        minibatch_pos_last_index_list = self.transition_indices_pos_last[self.transitions_head_pos:self.transitions_head_pos + self.ps]
        minibatch_neg_last_index_list = self.transition_indices_neg_last[self.transitions_head_neg:self.transitions_head_neg + self.ns]
        self.transitions_head_pos += self.ps
        self.transitions_head_neg += self.ns
        minibatch_index_list = minibatch_main_index_list + minibatch_pos_last_index_list + minibatch_neg_last_index_list
        get_from_dict = operator.itemgetter(*minibatch_index_list)
        s_minibatch = get_from_dict(self.transition_data['s'])
        actions_minibatch = get_from_dict(self.transition_data['actions'])
        rewards_minibatch = get_from_dict(self.transition_data['rewards'])
        next_s_minibatch = get_from_dict(self.transition_data['next_s'])
        terminals_minibatch = get_from_dict(self.transition_data['terminals'])
        # Updating current data head
        self.transitions_head += self.minibatch_size
        self.epoch_finished = self.transitions_head + self.drop_smaller_than_minibatch*self.minibatch_size >= self.transition_data_size
        self.transitions_head_pos = self.transitions_head_pos % len(self.transition_indices_pos_last)
        self.transitions_head_neg = self.transitions_head_neg % len(self.transition_indices_neg_last)
        return s_minibatch, actions_minibatch, rewards_minibatch, next_s_minibatch, terminals_minibatch, self.epoch_finished


