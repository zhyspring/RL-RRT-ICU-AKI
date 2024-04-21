import pickle as pk
from parameter import *
from model import *
from trainmodel import *
from utils import  DataLoader
from test import load_best_rl, load_rl_pool, TestClass
from exp import DQNExperiment
from data_process import data_set_split
import random
from os.path import join as ojoin
random.seed(params['random_seed'])
from data_process import dataprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

#loading and processing data---------------------------
dataprocess()
with open(ojoin(io_params['input_path_root'],'traj_data_demo.traj'), 'rb') as f:
    data_trajectory = pk.load(f)
train_data_trajectory, val_data_trajectory, test_data_trajectory = data_set_split(data_trajectory)

###state transition network training-------------------
print('train the sc network')
# train of SC-network
print('Initializing and training SC-Network')
sc_network = StateConstructor(**sc_params)
sc_network.train_state_constructor(sc_num_epochs=sc_ectra_p["sc_num_epochs"], saving_period=sc_ectra_p["sc_saving_period"], resume=sc_ectra_p["sc_resume"], 
                                   train_data= train_data_trajectory,
                                   valid_data= val_data_trajectory)
print("SC-Network training finished successfully")


#Rl Q-network learning-------------------------------
print('initialization...')
folder = os.path.abspath(params["folder_location"])
np.random.seed(params['random_seed'])
torch.manual_seed(params['random_seed'])
rng = np.random.RandomState(params['random_seed'])
# Initialize and load the pre-trained parameters for the SC-Network
sc_network = load_best_sc_network(params, rng, data_trajectory)  # note that the loaded SC-Network has no data inside
train_data_encoded = sc_network.encode_data(train_data_trajectory)
print("loading rl Validation data ...")
validation_data_encoded = sc_network.encode_data(val_data_trajectory)
loader_train, loader_validation = make_data_loaders(train_data_encoded, validation_data_encoded, rng, params['device'])
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


###test------------------------------------
# https://poe.com/s/H9SO5szP9sDYSy7IQbzv
print('Start to test')
# target:每隔一段时间对test_data去进行测试，目前只实现一次的demo
#加载测试数据、RL网络
print('Encoding test data')
test_trajectory_encoded = sc_network.encode_data(test_data_trajectory)
loader_test = DataLoader(test_trajectory_encoded, rng, 25600, False, 'cpu', ": test data")
loader_test.make_transition_data(release=True)



### test2: the loss of pre and one-step evaluation

#single net
print("Loading best Q-Networks and making Q-values ...")
root_dir_run = ojoin(io_params['output_path_root'], io_params['experiment_name'])
qnet = load_best_rl(root_dir_run, params)

# if for iter test, then use iter-steps net for testing
# a demo for avg_loss calculate
TestClass.avg_loss_cal(loader_test, qnet, root_dir_run)

# and a demo for iter avg_loss trends
print("Loading Q-Networks pool ...")
net_pool = load_rl_pool(root_dir_run, params)
loss_dic = [{net['iter']:TestClass.avg_loss_cal(loader_test, net['net'], root_dir_run)} for net in net_pool]
# result display
# 展示training_validation_test过程中loss的变化
# import matplotlib.pyplot as plt
# # 提取迭代次数和平均损失值
# iterations = [list(d.keys())[0] for d in loss_dic]
# losses = [list(d.values())[0] for d in loss_dic]

# # 将迭代次数从字符串转换为整数，以便正确排序
# iterations = [int(iteration) for iteration in iterations]

# # 将数据根据迭代次数排序（因为字典可能未按顺序）
# sorted_indices = sorted(range(len(iterations)), key=lambda k: iterations[k])
# iterations = [iterations[i] for i in sorted_indices]
# losses = [losses[i] for i in sorted_indices]

# # 绘制图表
# plt.figure(figsize=(10, 5))
# plt.plot(iterations, losses, marker='o', linestyle='-', color='b')
# plt.title('Average Loss vs. Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('Average Loss')
# plt.grid(True)
# plt.show()



### test1：the whole reward(avg return) (暂时采用Q进行代替)


### test3: the value for the key point (sample pre weight)


### test4: value distribution for different scenes






### decision support
# 对每个真实决策步骤的预测动作与真实动作进行比较，分析效果
action_reward_consistency = TestClass.action_reward_consistency(loader_test, qnet, root_dir_run)
# print(action_reward_consistency)
#先完成单个的，之后对整个轨迹进行分析

