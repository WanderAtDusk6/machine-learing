# tensorflow 过拟合(深度学习)
# 解决方案一，提供足够的大量的数据
# 方案二，l1,l2..regularization(正规化，，，，噗)
# # y = Wx
# l1 : cost = (Wx - real y )^2 + abs(W)     W变化的快，cost变的也快，使其变成一种惩罚机制
# l2 : cost = (Wx - real y )^2 + W^2
# l3,l4,,,,,,
# 方案三，（一种适用于神经网络的方法），dropout regularization,每次随机忽略神经元（使其不完整），根本上减少对W的依赖

# 本文件无代码233333