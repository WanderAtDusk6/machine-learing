# 线性回归预测PM2.5
import csv
import numpy as np
from numpy.linalg import inv  # ???
import random
import math
import sys

# read data
data = []
for i in range(18):
    data.append([])
    # data = [[],[],[],,,[],[]]

n_row = 0
#text = open('data/train.csv', 'r', encoding='big5')
text = open('F:\\data（机器学习）\\source\\1-线性回归_预测PM2.5\\data\\train.csv', 'r', encoding='big5')
# 喵喵喵，big5是什么诡异的编码方式
row = csv.reader(text, delimiter=",")
# !!!!csv.reader() 只有dialect
for r  in row:
    # 第0列没有资料
    if n_row != 0:
        # 每一列只有第3-27格有数据（即一天内24h的数值）
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
                # 什么鬼
                # 顺便撸一下NR是什么意思，有：not reported not relevant not required
            else:
                data[(n_row-1)%18].append(float(0))    # NR无关算作0
    n_row += 1
text.close()


# parse data to (x,y)
x = []
y = []
# 每12个月
for i in range(12):
    # 一个月取连续10小时的data可以有471
    for j in range(471):
        # 471471471是什么鬼
        x.append([])
        # 18种污染物
        for t in range(18):
        # 连续9小时
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])  # 480是因为20天*24时
        y.append(data[9][480*i+j+9])
x = np.array(x)    # x_shape: (5652, 162)
y = np.array(y)    # 觉得shape不对，print出来的shape 5652*162
print("x_shape:",x.shape)
print("y_shape:",x.shape)
# add square term
# x = np.concatenate((x,x**2),axis=0)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)

# init weigh&hyperparams
w = np.zeros(len(x[0]))    # 163,
print('w',w.shape)
l_rate = 10   # 学习率
repeat = 10000

# check your ans with close form solution
# however,this cannot be used in hw1.sh
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x))))

# start training
x_t = x.transpose()   # 163*5652
print("x_t:",x_t.shape)
s_gra = np.zeros(len(x[0]))   # shape 163
print("s_gra:",s_gra.shape)
print("x_shape:",x.shape)
print("w_shape:",w.shape)



for i in range(repeat):
    hypo = np.dot(x,w)   # 5652*163 163
    loss = hypo - y      # 
    cost = np.sum(loss**2)/len(x)   # len(s)=471,方差取平均
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t,loss)   # (163,)
    # print("gra",gra.shape)
    s_gra += gra**2         # 跟adagrad算法有关
    ada = np.sqrt(s_gra)    # 跟adagrad算法有关
    w = w - l_rate * gra/ada   #
    print("iteration:%d | Cost: %f "%(i,cost_a))


# save maodel
np.save('麻瓜模型.npy',w)
# read model
w = np.load('麻瓜模型.npy')

# read testing data
test_x = []
n_row = 0
text = open('F:\\data（机器学习）\\source\\1-线性回归_预测PM2.5\\data\\test.csv','r')
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row%18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]))
    else:
        for i in range(2,11):
            if r[i] != "NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(float(0))
    n_row = n_row + 1
text.close()
test_x = np.array(test_x)
print('test_x:',test_x.shape, test_x.shape[0], test_x.shape[1])
# add squre term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0]),1)), axis=1)

# get ans.csv with my model
ans = []
for i in range(len(text_x):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append()

filename = "result/predict.csv"
text = open(filename,"w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

