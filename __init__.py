# start.py: 完整的运行文件

# test.py: 在调试时使用的文件

# parameter.py 储存参数

# Model 储存DL 与 RL 的model

# exp 储存环境

# training 储存网络的训练类

# test 储存结果的训练类

# pipeline
'''
1. 读取ns,nr所需数据(注意ns数据的平衡性)
2. 将数据划分为训练集, 验证集, 测试集, 分别转化为轨迹数据
3. 对ns,nr构建SC网络
4. 对ns,nr进行训练
5. 选择效果较好的网络,进行轨迹数据的q值计算
6. 对Q值进行真值判断,并进行后续的结果统计分析
'''