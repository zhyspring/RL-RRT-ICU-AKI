#加载控制函数
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pyprind
import pickle as pk
import re
from trainmodel import RL


np.random.seed(42)
torch.manual_seed(42)
rng = np.random.RandomState(42)


def load_best_rl(root_dir_run, params):
    checkpoint_dir = os.path.join(root_dir_run, 'rl' + '_checkpoints')
    f = os.listdir(checkpoint_dir)
    f = [k for k in f if k[-3:] == ".pt"]
    last_checkpoint_idx = max([int(k[10:][:-3]) for k in f])
    last_rl_checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint' + str(last_checkpoint_idx) + '.pt'))
    best_rl = np.argmin(last_rl_checkpoint['validation_loss'])
    print("Best Q-Network: ", best_rl, ' :: ', root_dir_run)
    best_rl_check_point = torch.load(os.path.join(checkpoint_dir, f[best_rl]))
    # print(best_rl_check_point['rl_network_state_dict'])
    rl = RL(state_dim=params['embed_state_dim'], nb_actions=params["num_actions"], gamma=1.0, learning_rate=0, update_freq=0, 
            use_ddqn=False, rng=rng, device='cpu', network_size=params['rl_network_size'])
    rl.network.load_state_dict(best_rl_check_point['rl_network_state_dict'])
    print("Q-Network loaded")
    return rl

def load_rl_pool(root_dir_run, params):
    rl_pool = []
    checkpoint_dir = os.path.join(root_dir_run, 'rl' + '_checkpoints')
    f = os.listdir(checkpoint_dir) #checkpointxx.pt
    f = [k for k in f if k[-3:] == ".pt"]
    for k in f:
        rl = RL(state_dim=params['embed_state_dim'], nb_actions=params["num_actions"], gamma=1.0, learning_rate=0, update_freq=0, 
            use_ddqn=False, rng=rng, device='cpu', network_size=params['rl_network_size'])
        rl.network.load_state_dict(torch.load(os.path.join(checkpoint_dir, k))['rl_network_state_dict'])
        id = re.findall(r'-?\d+', k)[0]
        rl_pool.append({'iter':id,'net':rl})
    print("Q-Network loaded")
    return rl_pool


class TestClass():
    @staticmethod
    #合并在avg_loss_cal中
    def avg_reward(data_loader_test, q_network, storage_rl):
        epoch_val_steps = 0
        epoch_val_q = 0
        epoch_done = False
        # bar = pyprind.ProgBar(data_loader_test.num_minibatches_epoch)
        while not epoch_done:
            # bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = epoch_val_q.get_next_minibatch()
            epoch_val_steps += len(s)
            maxq = np.max(q_network.get_q(s))
            epoch_val_q += maxq
        
        # 计算平均q
        average_q = epoch_val_q / epoch_val_steps if epoch_val_steps != 0 else 0

        try:
            np.save(os.path.join(storage_rl, 'q_test_q.npy'), np.array(average_q))
        except Exception as e:
            print(f"Failed to save test q due to: {e}")
        # 保存结果
        return average_q, epoch_val_steps

    @staticmethod
    def avg_loss_cal(data_loader_test, q_network, storage_rl):
        epoch_val_steps = 0
        epoch_val_loss = 0
        epoch_done = False
        data_loader_test.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
        bar = pyprind.ProgBar(data_loader_test.num_minibatches_epoch)
        while not epoch_done:
            bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = data_loader_test.get_next_minibatch()
            epoch_val_steps += len(s)
            loss = q_network.get_loss(s, actions, rewards, next_s, terminals)
            epoch_val_loss += loss
        
        # 计算平均损失
        average_loss = epoch_val_loss / epoch_val_steps if epoch_val_steps != 0 else 0

        try:
            np.save(os.path.join(storage_rl, 'q_test_losses.npy'), np.array(average_loss))
        except Exception as e:
            print(f"Failed to save test losses due to: {e}")
        # 保存结果
        return average_loss

    @staticmethod
    def action_reward_consistency(data_loader_test, q_network, storage_rl):
        epoch_done = False
        action_reward_consistency = []  # 用于存储动作一致性和奖励
        data_loader_test.reset(shuffle=False, pos_samples_in_minibatch=0, neg_samples_in_minibatch=0)
        bar = pyprind.ProgBar(data_loader_test.num_minibatches_epoch)
        
        while not epoch_done:
            bar.update()
            s, actions, rewards, next_s, terminals, epoch_done = data_loader_test.get_next_minibatch()
            
            # 计算最优动作并比较
            best_actions = q_network.get_max_action(s)  # 假设这个方法返回s中每个状态的最优动作
            consistencies = actions == best_actions  # 比较已知动作与最优动作
            action_reward_consistency.extend(zip(consistencies, rewards))

        try:
            # 保存动作一致性和奖励的关系
            np.save(os.path.join(storage_rl, 'action_reward_consistency.npy'), np.array(action_reward_consistency))
        except Exception as e:
            print(f"Failed to save data due to: {e}")
        
        # 返回动作一致性和奖励的数据
        return action_reward_consistency


    @staticmethod
    def save_validation_loss(storage_rl, validation_loss):
        try:
            np.save(os.path.join(storage_rl, 'q_validation_losses.npy'), np.array(validation_loss))
        except Exception as e:
            print(f"Failed to save validation losses due to: {e}")

    

# def get_dn_rn_info(qnet_dn, qnet_rn, encoded_data, sepsis_data):
#     traj_indices = encoded_data['traj'].unique()
#     data = {'traj': [], 'step': [], 's': [], 'a': [], 'q_dn': [], 'q_rn': [], 'category': []}
#     state_cols = [k for k in encoded_data.columns if k[:2] == 's:']
#     reward_col = [k for k in encoded_data.columns if k[:2] == 'r:'][0]  # only one `r` col
#     action_col = [k for k in encoded_data.columns if k[:2] == 'a:'][0]  # only one `a` col
#     bar = pyprind.ProgBar(len(traj_indices))
#     print("Making Q-values...")
#     for traj in traj_indices:
#         bar.update()
#         traj_states = encoded_data[encoded_data['traj'] == traj][state_cols].to_numpy().astype(np.float32)
#         traj_q_dn = qnet_dn.get_q(traj_states)
#         traj_q_rn = qnet_rn.get_q(traj_states)
#         traj_q_dn = np.clip(traj_q_dn, -1, 0)
#         traj_q_rn = np.clip(traj_q_rn, 0, 1)
#         traj_r = sepsis_data[sepsis_data['traj'] == traj][reward_col].to_numpy().astype(np.float32)
#         traj_a = sepsis_data[sepsis_data['traj'] == traj][action_col].to_numpy().astype(np.int32)
#         steps = sepsis_data[sepsis_data['traj'] == traj]["step"].to_numpy().astype(np.int32)
        
#         for i, action in enumerate(traj_a):
#             data['traj'].append(traj)
#             data['step'].append(steps[i])
#             data['s'].append(traj_states[i, :])
#             data['a'].append(action)
#             data['q_dn'].append(traj_q_dn[i, :])
#             data['q_rn'].append(traj_q_rn[i, :])
#             if traj_r[-1] == -1.0:
#                 data['category'].append(-1)
#             elif traj_r[-1] == 1.0:
#                 data['category'].append(1)
#             else:
#                 raise ValueError('last reward of a trajectory is neither of -+1.')
#     data = pd.DataFrame(data)
#     print("Q values made.")
#     return data


