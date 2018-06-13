# 例3 添加层
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):         # ## 第五次更新，形参增加n_layer
    # add one more layer and return the output of this layer
    layer_name = 'layer(%s)' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)                     # ##第五次更新时增加，可视化
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 一行多列，建议初始值不为零，所以加个0.1
            tf.summary.histogram(layer_name + '/biases', biases)    # ##同上为了观察biases的变化
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b  # 这是针对线性关系的情况，不需要激励函数
        else:
            outputs = activation_function(Wx_plus_b,)
        tf.summary.histogram(layer_name+'outputs', outputs)        # ## 同上
        return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 加个维度
noise = np.random.normal(0, 0.05, x_data.shape)  # 噪点
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('x_and_y_placeholder'):        # tf.name_scope()名字的象征
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 后来加的，先占住这个位置,x_data属性只有1所以填1 ## 第四次更新，加上了name(为了可视化)
    ys = tf.placeholder(tf.float32, [None, 1],name='y_input')  # ## 同上

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  # 原x_data改为xs，y同样 ### 第五更，补上n_layer=1
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)    # 见上一行

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(np.square(ys - prediction),
                                        reduction_indices=[1], name='sum'), name='average_them')  # 简单说下，平均 求和（按[1]维度） 方差
    tf.summary.scalar('loss', loss)       # ##loss用另一种方法看,它在 EVENTS 里面显示
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 一个梯度下降的优化器,学习率要小于1

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()                # ## 合并
writer = tf.summary.FileWriter('f:/ML/', sess.graph)         # 4 可视化时加的，且版本有改动 放到 logs/ 文件夹里面
sess.run(init)

fig = plt.figure()  # 这是第三次更新时加的,这是plt中输出可视化的东西
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # 以 点 的形式输出来
plt.ion()                  # 好像是去除plot的暂停功能的
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # run train_step
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:                               # 先抹除线，再plot
            ax.lines.remove(lines[0])  # 去除lines[0]，,见上一行，只有一个故等于全清
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})  # 这也是第三次更新加的，哇受不了了
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)            # plot线的形式，x,y,线宽为5
        plt.pause(0.5)  # 暂停尼玛的0.1s
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})   # ##庆贺第五次更新，加入 merged 另，注意feed_data变化了
        writer.add_summary(result, i)                                            # ##庆贺第五次更新qnmd
# 在anaconda 输入 tensorboard --logdir F:/ML/  打开网址即可

