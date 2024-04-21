# SC-data-encode
#DataLoader and one-hot
import torch
import numpy as np
import pandas as pd
import os
import pyprind
from model import *
from parameter import *
from tqdm import tqdm
from utils import one_hot, DataLoader
from exp import DQNExperiment
import yaml

class StateConstructor(object):
    def __init__(self, train_data_file, validation_data_file, minibatch_size, rng, device, save_for_testing,
                sc_method, state_dim, sc_learning_rate, ais_gen_model, ais_pred_model, sc_neg_traj_ratio, 
                folder_location, folder_name, num_actions, obs_dim):
        '''
        We assume discrete actions and scalar rewards!
        '''
        self.rng = rng
        self.device = device
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.minibatch_size = minibatch_size
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.sc_method = sc_method
        self.sc_lr = sc_learning_rate
        self.sc_neg_traj_ratio = sc_neg_traj_ratio
        store_path = os.path.join(folder_location, folder_name)
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        else:
            print("Folder " + store_path + " is found.")
        if not os.path.exists(os.path.join(store_path, 'ais')):
            os.mkdir(os.path.join(store_path, 'ais'))
        if not os.path.exists(os.path.join(store_path, 'ais_checkpoints')):
            os.mkdir(os.path.join(store_path, 'ais_checkpoints'))
        if not os.path.exists(os.path.join(store_path, 'ais_data')):
            os.mkdir(os.path.join(store_path, 'ais_data'))
        self.store_path = store_path
        self.checkpoint_file = os.path.join(store_path, 'ais_checkpoints/checkpoint.pt')
        self.save_checkpoints_for_testing = save_for_testing
        self.ais_gen_file = os.path.join(store_path, 'ais_data/ais_gen.pt')
        self.ais_pred_file = os.path.join(store_path, 'ais_data/ais_pred.pt')
        self.ais_data_folder = os.path.join(store_path, 'ais_data')
        if ais_gen_model == 1:
            self.ais_gen_model = AISGenerate_1
        elif ais_gen_model == 2:
            self.ais_gen_model = AISGenerate_2
        if ais_pred_model == 1:
            self.ais_pred_model = AISPredict_1
        elif ais_pred_model == 2:
            self.ais_pred_model = AISPredict_2
    
    def reset(self):
        self.epoch_pos_finished = False
        self.epoch_neg_finished = False
        self.epoch_finished = False
        self.train_data_transition_head = 0
        self.train_data_transition_head_pos = 0
        self.train_data_transition_head_neg = 0
        self.train_data_transition_head_pos_last = 0
        self.train_data_transition_head_neg_last = 0
        self.rng.shuffle(self.train_data_transition_indices)
        self.rng.shuffle(self.train_data_transition_indices_pos)
        self.rng.shuffle(self.train_data_transition_indices_pos_last)
        self.rng.shuffle(self.train_data_transition_indices_neg)
        self.rng.shuffle(self.train_data_transition_indices_neg_last)
        return self.epoch_finished
    
    def reset_sc_networks(self):
        print('Reset SC-Network')
        self.ais_gen = self.ais_gen_model(self.state_dim, self.obs_dim, self.num_actions).to(self.device)
        self.ais_pred = self.ais_pred_model(self.state_dim, self.obs_dim, self.num_actions).to(self.device)
    
    def load_model_from_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        self.ais_gen.load_state_dict(checkpoint['gen_state_dict'])
        self.ais_pred.load_state_dict(checkpoint['pred_state_dict'])
        print("SC-Network: generator and predictor models loaded.")

    def load_mk_train_validation_data(self):
        print("SC-Network: loading raw data and making trajectory-level data")
        train_data = pd.read_csv(self.train_data_file)
        self.train_data_trajectory = self.make_trajectory_data(train_data)
        validation_data = pd.read_csv(self.validation_data_file)
        self.validation_data_trajectory = self.make_trajectory_data(validation_data)

    @staticmethod
    def make_trajectory_data(data, mode = 'nr'):
    # def make_trajectory_data(self, data):
        print('SC-Network: making trajectory data')
        # obs_cols = [i for i in data.columns if i[:2] == 'o:']
        # ac_cols  = [i for i in data.columns if i[:2] == 'a:']
        # rew_cols = [i for i in data.columns if i[:2] == 'r:']
        obs_cols = STATE_COLS
        ac_cols = ACTION_COLS
        rew_cols = REWARD_COLS
        #Assuming discrete actions and scalar rewards:
        assert len(obs_cols) > 0, 'No observations present, or observation columns not prefixed with "o:"'
        assert len(ac_cols) > 0, 'No actions present, or actions column not prefixed with "a:"'
        assert len(rew_cols) > 0, 'No rewards present, or rewards column not prefixed with "r:"'
        assert len(ac_cols) == 1, 'Multiple action columns are present when a single action column is expected'
        assert len(rew_cols) == 1, 'Multiple reward columns are present when a single reward column is expected'
        ac_col = ac_cols[0]
        rew_col = rew_cols[0]
        data[ac_col] = data[ac_col]
        all_actions = data[ac_col].unique()
        all_actions.sort()
        try:
            all_actions = all_actions.astype(np.int32)
        except:
            raise ValueError('Actions are expected to be integers, but are not.')
        # if not all(all_actions == np.arange(self.num_actions, dtype=np.int32)):
        #     print(Font.red + 'Some actions are missing from data or all action space not properly defined.' + Font.end)
        print("Number of actions in the file: ", len(all_actions))
        trajectories = data['traj'].unique()
        data_trajectory = {}
        data_trajectory['obs_cols'] = obs_cols
        data_trajectory['ac_col']  = ac_col
        data_trajectory['rew_col'] = rew_col
        # data_trajectory['num_actions'] = self.num_actions
        data_trajectory['num_actions'] = data[ACTION_COLS[0]].nunique()
        data_trajectory['obs_dim'] = len(obs_cols)
        data_trajectory['traj'] = {}
        data_trajectory['pos_traj'] = []
        data_trajectory['neg_traj'] = []
        bar = pyprind.ProgBar(len(trajectories))
        for i in trajectories:
            bar.update()
            traj_i = data[data['traj'] == i].sort_values(by=TIME_COLS[0])
            data_trajectory['traj'][i] = {}
            # data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to(self.device)
            # data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i[ac_col].values.astype(np.int32)).to(self.device).long()
            # data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i[rew_col].values).to(self.device)

            data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to("cuda" if torch.cuda.is_available() else "cpu")
            data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i[ac_col].values.astype(np.int32)).to("cuda" if torch.cuda.is_available() else "cpu").long()
            data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i[rew_col].values).to("cuda" if torch.cuda.is_available() else "cpu")
            if traj_i[rew_col].values[-1] < 0: 
                data_trajectory['pos_traj'].append(i)
            else:
                data_trajectory['neg_traj'].append(i)
        return data_trajectory

    def train_state_constructor(self, sc_num_epochs, saving_period, resume, train_data, valid_data):
        self.train_data_trajectory = train_data
        self.validation_data_trajectory = valid_data
        if self.sc_method == 'AIS':
            print('Training State Construction Network')
            device = self.device
            num_actions = self.train_data_trajectory['num_actions']
            obs_dim = self.train_data_trajectory['obs_dim']
            self.ais_gen = self.ais_gen_model(self.state_dim, obs_dim, num_actions).to(device)
            self.ais_pred = self.ais_pred_model(self.state_dim, obs_dim, num_actions).to(device)
            self.optimizer = torch.optim.Adam(list(self.ais_gen.parameters()) + list(self.ais_pred.parameters()), lr=self.sc_lr, amsgrad=True)
            self.sc_losses = []
            self.sc_losses_validation = []
            self.best_validation_loss = 1e20  # Initialize the best validation loss to be very high...
            positive_trajectories = self.train_data_trajectory['pos_traj']
            negative_trajectories = self.train_data_trajectory['neg_traj']
            epoch_trajectories = list(self.train_data_trajectory['traj'].keys())
            if self.sc_neg_traj_ratio != 'NA':
                if len(negative_trajectories)/len(epoch_trajectories) > self.sc_neg_traj_ratio:
                    target_len_positive_trajectories = int(np.round((1-self.sc_neg_traj_ratio)*len(negative_trajectories)/self.sc_neg_traj_ratio))
                    epoch_trajectories = negative_trajectories + target_len_positive_trajectories//len(positive_trajectories)*positive_trajectories + positive_trajectories[:target_len_positive_trajectories%len(positive_trajectories)]
                else:
                    target_len_negative_trajectories = int(np.round(self.sc_neg_traj_ratio*len(negative_trajectories)/(1-self.sc_neg_traj_ratio)))
                    epoch_trajectories = positive_trajectories + target_len_negative_trajectories//len(negative_trajectories)*negative_trajectories + negative_trajectories[:target_len_negative_trajectories%len(negative_trajectories)]
            if resume:
                checkpoint = torch.load(self.checkpoint_file)
                self.ais_gen.load_state_dict(checkpoint['gen_state_dict'])
                self.ais_pred.load_state_dict(checkpoint['pred_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_0 = checkpoint['epoch'] + 1
                self.sc_losses = checkpoint['loss']
                self.sc_losses_validation = checkpoint['validation_loss']
                self.best_validation_loss = checkpoint['best_validation_loss']
                print('Starting from epoch: {0} and continuing upto epoch {1}'.format(epoch_0, sc_num_epochs))
            else:
                epoch_0 = 0
            for epoch in range(epoch_0, sc_num_epochs):
                epoch_loss = []
                print("SC-Network {0}: training Epoch = ".format(self.sc_method), epoch+1, 'out of', sc_num_epochs, 'epochs')
                bar = pyprind.ProgBar(len(epoch_trajectories))
                for traj in tqdm(epoch_trajectories):
                    bar.update()
                    loss_pred = 0
                    h = torch.zeros(self.state_dim).to(device).view(1,-1)
                    obs = self.train_data_trajectory['traj'][traj]['obs']
                    actions = self.train_data_trajectory['traj'][traj]['actions'].view(-1,1)
                    rewards = self.train_data_trajectory['traj'][traj]['rewards'].view(-1,1)
                    ais = torch.zeros(obs.shape[0], self.state_dim).to(device)
                    action = torch.zeros(num_actions).to(device) #Initial action; all zeros
                    rew = torch.zeros(1).to(device) #Initial rewrad; zero
                    for step in range(obs.shape[0]-1):
                        h = self.ais_gen(torch.cat((obs[step,:], action)).view(1,-1), h)
                        ais[step,:] = h
                        action = one_hot(actions[step], num_actions, data_type='torch', device=device)
                        rew = rewards[step]
                        obs_pred_next_probs = self.ais_pred((torch.cat((ais[step,:], action))).view(1,-1))
                        # Loss in predicting distribution of next observation
                        loss_pred += -torch.distributions.MultivariateNormal(obs_pred_next_probs[0,:], torch.eye(obs_pred_next_probs[0,:].shape[0]).to(device)).log_prob(obs[step+1,:])
                    self.optimizer.zero_grad()
                    if obs.shape[0] > 1:
                        loss_pred.backward()
                        self.optimizer.step()
                        epoch_loss.append(loss_pred.detach().cpu().numpy())
                self.sc_losses.append(epoch_loss)

                if (epoch+1) % saving_period == 0:
                    #Computing validation loss
                    epoch_validation_loss = []
                    for traj in self.validation_data_trajectory['traj'].keys():
                        loss_val = 0
                        h_val = torch.zeros(self.state_dim).to(device).view(1, -1)
                        obs_val = self.validation_data_trajectory['traj'][traj]['obs']
                        actions_val = self.validation_data_trajectory['traj'][traj]['actions'].view(-1, 1)
                        rewards_val = self.validation_data_trajectory['traj'][traj]['rewards'].view(-1, 1)
                        ais_val = torch.zeros(obs_val.shape[0], self.state_dim).to(device)
                        action_val = torch.zeros(num_actions).to(device) #Initial action; all zeros
                        rew_val = torch.zeros(1).to(device) #Initial reward; zero
                        for step in range(obs_val.shape[0]-1):
                            with torch.no_grad():
                                h_val = self.ais_gen(torch.cat((obs_val[step,:], action_val)).view(1,-1), h_val)
                                ais_val[step,:] = h_val
                                action_val = one_hot(actions_val[step], num_actions, data_type='torch', device=device)
                                rew_val = rewards_val[step]
                                obs_pred_next_probs_val = self.ais_pred((torch.cat((ais_val[step,:], action_val))).view(1,-1))
                                # Loss in predicting distribution of next observation
                                loss_val += -torch.distributions.MultivariateNormal(obs_pred_next_probs_val[0,:], torch.eye(obs_pred_next_probs_val[0,:].shape[0]).to(device)).log_prob(obs_val[step+1,:])    
                        if obs_val.shape[0] > 1:
                            epoch_validation_loss.append(loss_val.detach().cpu().numpy())
                    self.sc_losses_validation.append(epoch_validation_loss)

                    # Save off checkpoint every epoch for testing if specified
                    if self.save_checkpoints_for_testing:
                        try:
                            torch.save({
                                'epoch': epoch,
                                'gen_state_dict': self.ais_gen.state_dict(),
                                'pred_state_dict': self.ais_pred.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.perception_losses,
                                'validation_loss': self.perception_losses_validation,
                                }, self.checkpoint_file[:-3] + str(epoch) +'.pt')
                            np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.perception_losses))
                        except:
                            pass
                    
                    # Save off checkpoint if improved overall validation loss.
                    if np.mean(epoch_validation_loss) <= self.best_validation_loss:
                        self.best_validation_loss = np.mean(epoch_validation_loss)  # Reset the current best validation loss
                        try:
                            torch.save({
                                'epoch': epoch,
                                'gen_state_dict': self.ais_gen.state_dict(),
                                'pred_state_dict': self.ais_pred.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.sc_losses,
                                'validation_loss': self.sc_losses_validation,
                                'best_validation_loss': self.best_validation_loss,
                                }, self.checkpoint_file[:-3] + '_best' +'.pt')
                            np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.sc_losses))
                        except:
                            pass

                    # Save off validation losses
                    try:
                        np.save(self.ais_data_folder + '/ais_validation_losses.npy', np.array(self.sc_losses_validation))
                    except:
                        pass

                    # We want to maintain the most recent model for checkpointing purposes
                    try:
                        torch.save(self.ais_gen.state_dict(), self.ais_gen_file)
                        torch.save(self.ais_pred.state_dict(), self.ais_pred_file)
                        torch.save({
                            'epoch': epoch,
                            'gen_state_dict': self.ais_gen.state_dict(),
                            'pred_state_dict': self.ais_pred.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.sc_losses,
                            'validation_loss': self.sc_losses_validation,
                            'best_validation_loss': self.best_validation_loss,
                            }, self.checkpoint_file)
                        np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.sc_losses))
                    except:
                        pass

            print('SC-Network training finished successfully')

    # def train_state_constructor(self, data, sc_num_epochs=100):
    #     self.train_data_trajectory = data
    #     if self.sc_method == 'AIS':
    #         print('Training State Construction Network')
    #         device = self.device
    #         num_actions = self.train_data_trajectory['num_actions']
    #         obs_dim = self.train_data_trajectory['obs_dim']
    #         self.ais_gen = self.ais_gen_model(self.state_dim, obs_dim, num_actions).to(device)
    #         self.ais_pred = self.ais_pred_model(self.state_dim, obs_dim, num_actions).to(device)
    #         self.optimizer = torch.optim.Adam(list(self.ais_gen.parameters()) + list(self.ais_pred.parameters()), lr=self.sc_lr, amsgrad=True)
    #         self.sc_losses = []
    #         self.sc_losses_validation = []
    #         self.best_validation_loss = 1e20  # Initialize the best validation loss to be very high...
    #         positive_trajectories = self.train_data_trajectory['pos_traj']
    #         negative_trajectories = self.train_data_trajectory['neg_traj']
    #         epoch_trajectories = list(self.train_data_trajectory['traj'].keys())
            
    #         # 省略了与负样本比例相关的处理部分，如果这部分逻辑对新模型训练非常重要，可以保留
            
    #         for epoch in range(sc_num_epochs):
    #             epoch_loss = []
    #             print("SC-Network {0}: training Epoch = ".format(self.sc_method), epoch+1, 'out of', sc_num_epochs, 'epochs')
                
    #             # 省略了进度条相关的代码，如pyprind.ProgBar，因为它不是训练逻辑的核心部分
                
    #             for traj in tqdm(epoch_trajectories):
    #                 # 省略了进度条更新的代码，如bar.update()
                    
    #                 loss_pred = 0
    #                 h = torch.zeros(self.state_dim).to(device).view(1,-1)
    #                 obs = self.train_data_trajectory['traj'][traj]['obs']
    #                 actions = self.train_data_trajectory['traj'][traj]['actions'].view(-1,1)
    #                 rewards = self.train_data_trajectory['traj'][traj]['rewards'].view(-1,1)
    #                 ais = torch.zeros(obs.shape[0], self.state_dim).to(device)
    #                 action = torch.zeros(num_actions).to(device) #Initial action; all zeros
    #                 rew = torch.zeros(1).to(device) #Initial reward; zero
    #                 for step in range(obs.shape[0]-1):
    #                     h = self.ais_gen(torch.cat((obs[step,:], action)).view(1,-1), h)
    #                     ais[step,:] = h
    #                     action = one_hot(actions[step], num_actions, data_type='torch', device=device)
    #                     rew = rewards[step]
    #                     obs_pred_next_probs = self.ais_pred((torch.cat((ais[step,:], action))).view(1,-1))
    #                     # Loss in predicting distribution of next observation
    #                     loss_pred += -torch.distributions.MultivariateNormal(obs_pred_next_probs[0,:], torch.eye(obs_pred_next_probs[0,:].shape[0]).to(device)).log_prob(obs[step+1,:])
    #                 # 省略的代码包括了训练逻辑
                    
    #                 self.optimizer.zero_grad()
    #                 if obs.shape[0] > 1:
    #                     loss_pred.backward()
    #                     self.optimizer.step()
    #                     epoch_loss.append(loss_pred.detach().cpu().numpy())
    #             self.sc_losses.append(epoch_loss)
    #         self.epoch = sc_num_epochs
    
    @staticmethod
    def train_state_constructor_stage2(self):
        # 省略了关于验证集损失的计算和保存检查点的部分代码
        torch.save({
            'epoch': self.epoch,
            'gen_state_dict': self.ais_gen.state_dict(),
            'pred_state_dict': self.ais_pred.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': self.perception_losses,
            # 'validation_loss': self.perception_losses_validation,
            }, os.path.join(os.path.join(self.store_path, 'ais_checkpoints'),str(self.epoch) +'.pt'))
        # np.save(self.ais_data_folder + '/ais_losses.npy', np.array(self.perception_losses))    
        print('SC-Network training finished successfully')

    def encode_data(self, data_trajectory):
        d = data_trajectory.copy()
        print("SC-Network: encoding data")
        bar = pyprind.ProgBar(len(data_trajectory['traj'].keys()))
        for traj in data_trajectory['traj'].keys():
            bar.update()
            obs = data_trajectory['traj'][traj]['obs']
            actions = data_trajectory['traj'][traj]['actions'].view(-1,1)
            rewards = data_trajectory['traj'][traj]['rewards'].view(-1,1)
            ais = torch.zeros(obs.shape[0], self.state_dim).to(self.device)
            h = torch.zeros(self.state_dim).to(self.device).view(1,-1)
            a = torch.zeros(self.num_actions).to(self.device)
            # r = torch.zeros(1).to(self.device)
            with torch.no_grad():
                for step in range(obs.shape[0]):
                    h = self.ais_gen(torch.cat((obs[step,:], a)).view(1,-1), h)
                    ais[step,:] = h
                    a = one_hot(actions[step], self.num_actions, data_type='torch', device=self.device)
                    # r = rewards[step]
            d['traj'][traj]['obs'] = ais.cpu().numpy()
            d['traj'][traj]['s'] = d['traj'][traj].pop('obs')  # switch to "s" (sinces it's state)
            d['traj'][traj]['actions'] = d['traj'][traj]['actions'].cpu().numpy()
            d['traj'][traj]['rewards'] = d['traj'][traj]['rewards'].cpu().numpy()
        s_cols = ['s:' + str(i) for i in range(self.state_dim)]
        d['s_cols'] = s_cols
        d['s_dim'] = len(s_cols)
        return d
    
    @staticmethod
    def encoded_trajectory_data_to_file(trajectory_data, filename):
        print('SC-Network: Writing encoded trajectory data to file')
        col_names = ['traj', 'step']
        col_names.extend(['s:'+ i[2:] for i in trajectory_data['s_cols']])
        col_names.append('a:action')
        col_names.append('r:reward')
        all_data = []
        bar = pyprind.ProgBar(len(list(trajectory_data['traj'].keys())))
        for i in trajectory_data['traj'].keys():
            bar.update()
            for ctr in range(trajectory_data['traj'][i]['actions'].shape[0]):
                all_data.append([])
                all_data[-1].append(i)
                all_data[-1].append(ctr)
                for s_index in range(trajectory_data['traj'][i]['s'].shape[1]):
                    all_data[-1].append(trajectory_data['traj'][i]['s'][ctr, s_index])
                all_data[-1].append(int(trajectory_data['traj'][i]['actions'][ctr]))
                all_data[-1].append(trajectory_data['traj'][i]['rewards'][ctr])
        df = pd.DataFrame(all_data, columns=col_names)
        df.to_csv(filename, index=False)



#RL Train Class

class RL(object):
    def __init__(self, state_dim, nb_actions, gamma,
                 learning_rate, update_freq,use_ddqn,
                 rng, device, network_size):
        self.rng = rng
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.update_counter = 0
        self.use_ddqn = use_ddqn
        self.device = device
        self.network_size = network_size
        if self.network_size == 'small':
            QNetwork = QNetwork_64
        elif self.network_size == 'large':
            QNetwork = QNetwork_128
        elif self.network_size == '2layered':
            QNetwork = QNetwork_6464
        self.network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.target_network = QNetwork(state_dim=self.state_dim, nb_actions=self.nb_actions)
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, amsgrad=True)
   

    def train_on_batch(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        q = self.network(s)
        q2 = self.target_network(s2).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)  #与Q(s,a)不同，Q(s)同时预测每个a的值，然后再选取a的分量
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        if 1:
            # bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
            bellman_target = torch.clamp(r, max=CL_UPPER, min=CL_LOWER) + self.gamma * torch.clamp(q2_max.detach(), max=CL_UPPER, min=CL_LOWER) * (1 - t)
       

        loss = F.smooth_l1_loss(q_pred, bellman_target)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_loss(self, s, a, r, s2, t):
        s  = torch.FloatTensor(np.float32(s)).to(self.device)
        s2 = torch.FloatTensor(np.float32(s2)).to(self.device)
        a  = torch.LongTensor(np.int64(a)).to(self.device)
        r  = torch.FloatTensor(np.float32(r)).to(self.device)
        t  = torch.FloatTensor(np.float32(t)).to(self.device)

        with torch.no_grad():
            q = self.network(s).detach()
            q2 = self.target_network(s2).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1) 
        if self.use_ddqn:
            q2_net = self.network(s2).detach()
            q2_max = q2.gather(1, torch.max(q2_net, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            q2_max = torch.max(q2, 1)[0]
        if 1:
            # bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
            bellman_target = torch.clamp(r, max=CL_UPPER, min=CL_LOWER) + self.gamma * torch.clamp(q2_max.detach(), max=CL_UPPER, min=CL_LOWER) * (1 - t)

        # if self.sided_Q == 'negative':
        #     bellman_target = torch.clamp(r, max=0.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=0.0, min=-1.0) * (1 - t)
        # elif self.sided_Q == 'positive':
        #     bellman_target = torch.clamp(r, max=1.0, min=0.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0) * (1 - t)
        # elif self.sided_Q == 'both':
        #     bellman_target = torch.clamp(r, max=1.0, min=-1.0) + self.gamma * torch.clamp(q2_max.detach(), max=1.0, min=-1.0) * (1 - t)
        
        loss = F.smooth_l1_loss(q_pred, bellman_target)
        return loss.detach().cpu().numpy()

    def get_q(self, s):
        s = torch.FloatTensor(s).to(self.device)
        return self.network(s).detach().cpu().numpy()

    def get_max_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        q = self.network(s).detach()
        return q.max(1)[1].cpu().numpy()

    def get_action(self, states, epsilon=0.1):
        if np.random.rand() < epsilon:
            # 以ε的概率随机选择一个动作
            return np.random.choice(2, 1) 
        else:
            return self.get_max_action(states)
  # 使用.item()将单个值张量转换为Python数值
    # def get_action(self, states):
    #     return self.get_max_action(states)

    def learn(self, s, a, r, s2, term):
        """ Learning from one minibatch """
        loss = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return loss

    def dump_network(self, weights_file_path):
        try:
            torch.save(self.network.state_dict(), weights_file_path)
        except:
            pass

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def resume(self, network_state_dict, target_network_state_dict, optimizer_state_dict):
        self.network.load_state_dict(network_state_dict)
        self.target_network.load_state_dict(target_network_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        return _dict
    

def load_best_sc_network(params, rng, data_trajectory):
    # NOTE: returned SC-Network will need to either re-load data or load some new test data

    store_path = os.path.join(params["folder_location"], params["folder_name"])  # this is `sc_network.store_path` if a SC-Network is loaded with params 
    # Initialize the SC-Network
    sc_network = StateConstructor(train_data_file=params["train_data_file"], validation_data_file=params["validation_data_file"], 
                            minibatch_size=params["minibatch_size"], rng=rng, device=params["device"], save_for_testing=params["save_all_checkpoints"],
                            sc_method=params["sc_method"], state_dim=params["embed_state_dim"], sc_learning_rate=params["sc_learning_rate"], 
                            ais_gen_model=params["ais_gen_model"], ais_pred_model=params["ais_pred_model"], sc_neg_traj_ratio=params["sc_neg_traj_ratio"], 
                            folder_location=params["folder_location"], folder_name=params["folder_name"], 
                            num_actions=params["num_actions"], obs_dim=params["obs_dim"])
    sc_network.train_data_trajectory = data_trajectory
    sc_network.reset_sc_networks()
    # Provide SC-Network with the pre-trained parameter set
    sc_network.load_model_from_checkpoint(checkpoint_file_path=os.path.join(store_path, "ais_checkpoints", "checkpoint_best.pt"))
    return sc_network


def make_data_loaders(train_data_encoded, validation_data_encoded, rng, device):
    # Note that the loaders will be reset in Experiment
    loader_train = DataLoader(train_data_encoded, rng, 64, False, device, ": train data")
    loader_validation = DataLoader(validation_data_encoded, rng, 256, False, device, ": validation data")
    loader_train.make_transition_data(release=True)
    loader_validation.make_transition_data(release=True)
    return loader_train, loader_validation


def train(params, rng, loader_train, loader_validation):
    qnet = RL(state_dim=params["embed_state_dim"], nb_actions=params["num_actions"], gamma=params["gamma"], learning_rate=params["rl_learning_rate"], 
                update_freq=params["update_freq"], use_ddqn=params["use_ddqn"], rng=rng, device=params["device"], 
                network_size=params["rl_network_size"])
    print('Initializing Experiment')
    expt = DQNExperiment(data_loader_train=loader_train, data_loader_validation=loader_validation, q_network=qnet, ps=0, ns=2,
                        folder_location=params["folder_location"], folder_name=params["folder_name"], 
                        saving_period=params["exp_saving_period"], rng=rng, resume=params["rl_resume"])
    with open(os.path.join(expt.storage_rl, 'config_exp.yaml'), 'w') as y:
            yaml.safe_dump(params, y)  # saving new params for future reference
    print('Running experiment (training Q-Networks)')
    expt.do_epochs(number=params["exp_num_epochs"])
    print("Training Q-Networks finished successfully")