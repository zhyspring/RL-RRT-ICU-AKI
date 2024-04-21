import pandas as pd
from trainmodel import StateConstructor
from parameter import *
from os.path import join as ojoin
import pickle as pk
from copy import deepcopy
import random 
random.seed(params['random_seed'])
def dataprocess():
    print('start')
    # sc_constructor = StateConstructor()
    # stay_id = pd.read_csv(ojoin(io_params['input_path_root'],  r'demo.csv'), dtype={'stay_id':str})['stay_id'].tolist()
    file = pd.read_csv(ojoin(io_params['input_path_root'], io_params['oral_dataset']), dtype=DTYPE_DICT)
    # stay_id = random.sample(stay_id, split_params['train_size'] + split_params['test_size'] + split_params['val_size'])
    # file = file[file['stay_id'].isin(stay_id)]
    file = file.rename(columns={LABEL_COLS[0]:'traj'})
    data_trajectory = StateConstructor.make_trajectory_data(file, 'try')
    # data_trajectory_s = StateConstructor.make_trajectory_data(file, 'ns')    # # 
    with open(ojoin(io_params['input_path_root'],'traj_data_demo.traj'), 'wb') as f:
        pk.dump(data_trajectory, f)

        

def data_set_split(data_trajectory):

    # 假设 data_trajectory['traj'] 是一个字典
    traj_dict = data_trajectory['traj']

    # # 计算训练集、验证集和测试集的大小
    # train_size = int(len(traj_dict) * 0.6)
    # val_size = int(len(traj_dict) * 0.3)
    # # 测试集的大小是剩下的部分
    # test_size = int(len(traj_dict) * 0.1)
    train_size = split_params['train_size']
    val_size = split_params['val_size']
    test_size = split_params['test_size']


    # 随机抽取键的列表
    all_keys = list(traj_dict.keys())
    random.shuffle(all_keys)

    # 分配键到训练集、验证集和测试集
    train_keys = all_keys[:train_size]
    val_keys = all_keys[train_size:train_size + val_size]
    test_keys = all_keys[train_size + val_size:train_size + val_size + test_size]

    # 创建对应的训练集、验证集和测试集字典
    train_traj_dict = {key: traj_dict[key] for key in train_keys}
    val_traj_dict = {key: traj_dict[key] for key in val_keys}
    test_traj_dict = {key: traj_dict[key] for key in test_keys}

    # 复制data_trajectory并替换traj键对应的值
    train_data_trajectory = deepcopy(data_trajectory)
    val_data_trajectory = deepcopy(data_trajectory)
    test_data_trajectory = deepcopy(data_trajectory)

    train_data_trajectory['traj'] = train_traj_dict
    val_data_trajectory['traj'] = val_traj_dict
    test_data_trajectory['traj'] = test_traj_dict

    # 现在你有 train_data_trajectory, val_data_trajectory, 和 test_data_trajectory
    # 分别对应训练集、验证集和测试集，其他键值对保持不变。
    #  函数用于剔除不在'traj'中的键
    # Function to filter lists of dictionaries based on a set of keys to keep
    def filter_list_of_dicts(list_of_trajid, keys_to_keep):
        return [d for d in list_of_trajid if d in keys_to_keep]

    # Update 'pos_traj' and 'neg_traj' to only include entries that correspond to 'traj' keys
    train_data_trajectory['pos_traj'] = filter_list_of_dicts(train_data_trajectory['pos_traj'], train_keys)
    train_data_trajectory['neg_traj'] = filter_list_of_dicts(train_data_trajectory['neg_traj'], train_keys)

    val_data_trajectory['pos_traj'] = filter_list_of_dicts(val_data_trajectory['pos_traj'], val_keys)
    val_data_trajectory['neg_traj'] = filter_list_of_dicts(val_data_trajectory['neg_traj'], val_keys)

    test_data_trajectory['pos_traj'] = filter_list_of_dicts(test_data_trajectory['pos_traj'], test_keys)
    test_data_trajectory['neg_traj'] = filter_list_of_dicts(test_data_trajectory['neg_traj'], test_keys)
    # 现在 train_data_trajectory, val_data_trajectory, 和 test_data_trajectory
    # 包含更新后的训练集、验证集和测试集
    return train_data_trajectory, val_data_trajectory, test_data_trajectory