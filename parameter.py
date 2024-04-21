#SC Train parameter
import torch
import numpy as np
import pickle as pk
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)


io_params = {
    'input_path_root':r'./data/input',
    'output_path_root':r'./data/output',
    'experiment_name': 'exp1',
    'oral_dataset':'fakedata.csv',
    'oral_dataset_col':'fakedata_col.data'
}

with open(os.path.join(io_params['input_path_root'], io_params['oral_dataset_col']),'rb') as f:
    data_col = pk.load(f)

LABEL_COLS = data_col['ID_col']
TIME_COLS = data_col['time_col']
ACTION_COLS = data_col['action_col']
STATE_COLS = data_col['state_col']
REWARD_COLS = data_col['reward_col']

# 构建适合pd.read_csv dtypes参数的字典
DTYPE_DICT = {}
DTYPE_DICT.update({col: 'str' for col in LABEL_COLS})
DTYPE_DICT.update({col: 'int' for col in TIME_COLS})
DTYPE_DICT.update({col: 'int' for col in ACTION_COLS})
DTYPE_DICT.update({col: 'float' for col in STATE_COLS})
DTYPE_DICT.update({col: 'float' for col in REWARD_COLS})

# INPUT_PATH = 'E:\\ICU_RRT\\code\\data\\temp_acts_v3'
# OUTPUT_PATH = 'E:\\ICU_RRT\\code\\data\\temp_acts_v4_csv'
CL_UPPER = 1
CL_LOWER = -1


split_params = {
    'train_size' : 70,
    'val_size' : 15,
    'test_size' : 15,
    }

sc_params = {
    "train_data_file": None,  # 训练数据文件路径
    "validation_data_file": None,  # 验证数据文件路径
    "minibatch_size": 32,  # 批处理大小
    "rng" : np.random.RandomState(42),
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 使用GPU还是CPU
    "save_for_testing": True,  # 是否保存所有检查点
    "sc_method": "AIS",  # 状态构建方法
    "state_dim": 128,  # 嵌入状态的维度
    "sc_learning_rate": 0.001,  # 学习率
    "ais_gen_model": 1,  # AIS生成模型类
    "ais_pred_model": 1,  # AIS预测模型类
    "sc_neg_traj_ratio": 0.5,  # 负样本轨迹比例
    "folder_location": io_params['output_path_root'],  # 结果保存文件夹位置
    "folder_name": io_params['experiment_name'],  # 结果保存文件夹名称
    "num_actions": 3,  # 行动数量
    "obs_dim": 30,  # 观测维度

}

sc_ectra_p = { 'random_seed': 42,  # 随机种子，用于复现结果
    'sc_num_epochs': 1,
    'sc_saving_period':1 ,
    'sc_resume':False,}

#RL parameter
params = {
    'train_data_file': 'path/to/your/train/data.csv',  # 训练数据集的文件路径
    'validation_data_file': 'path/to/your/validation/data.csv',  # 验证数据集的文件路径
    'test_data_file': 'path/to/your/test/data.csv',  # 测试数据集的文件路径
    'minibatch_size': 32,  # 批量大小
    'device': 'cuda',  # 训练使用的设备，cuda 或者 cpu
    'save_all_checkpoints': True,  # 是否保存所有训练过程中的模型检查点
    'sc_method': 'AIS',  # 状态构建方法
    'embed_state_dim': 128,  # 状态嵌入的维度
    'sc_learning_rate': 0.001,  # 状态构建的学习率
    'ais_gen_model': 1,  # AIS生成模型的类名
    'ais_pred_model': 1,  # AIS预测模型的类名
    'sc_neg_traj_ratio': 0.5,  # 负轨迹比例
    'folder_location': io_params['output_path_root'],  # 存储文件夹的位置
    'folder_name': io_params['experiment_name'], # 实验文件夹名称
    'num_actions': 3 * len(ACTION_COLS),  # 可能的行动数量
    'obs_dim': len(STATE_COLS),  # 观测空间的维度
    'gamma': 0.99,  # 强化学习中的折扣因子
    'rl_learning_rate': 0.0001,  # 强化学习的学习率
    'update_freq': 4,  # Q网络更新频率
    'use_ddqn': True,  # 是否使用双重DQN
    'rl_network_size': 'small',  # 强化学习网络的大小
    'exp_saving_period': 5,  # 实验保存周期
    'exp_num_epochs': 10,  # 实验时代数
    'rl_resume': False,  # 是否从之前的检查点继续强化学习过程
    'random_seed': 42,  # 随机种子，用于复现结果
    'sc_num_epochs': 42,
    'sc_saving_period':3 ,
    'sc_resume':False,
}

