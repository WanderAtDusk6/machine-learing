# CNN卷积神经网络

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})     # 预测值 全连接
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)      # stddev(标准差)  tf.truncated_normal() 产生正态分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):          # 卷积神经网络层  x 输入的值， W 即Weight
    # stride [1, x_movement, y_movement, 1]       padding的两种方法，见我爪机截图，woc困死了
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')              # 二维的,mmp    strides(步长),前和后必为1，然后中间的是x,y方向，这边定的跨度是1


def max_pool_2x2(x):      # pooling的一个图层      kszie是卷积核的大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')          # 传入的W就包含了卷积核的个数信


# pooling的步长是2，与conv2的步长不同，整个的大小相当于被压缩2
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input_image')    # 28*28
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input_r_num')
    keep_prob = tf.placeholder(tf.float32, name='dropout_neure')      # ##这里有个没被使用的keep_prob
x_image = tf.reshape(xs, [-1, 28, 28, 1])         # 变换xs的shape的形式  -1，指的是不用指定这一维度的大小，系统会自己推算,因为图像为黑白，所以最后一位是1
# print(x_image.shape)      # [n_samples, 28, 28, 28, 1]        继续补充上一行的-1，可理解为导入数据有多少的图片

## conv1 layer ##    第一个卷积层
with tf.name_scope('layer_conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])       # [patch 5x5的像素, in size， out size],,,那个东西的底是1，高度是32，个人看法，5x5x32=800>784
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])         # 传入这样的一个shape,只有32的长度  ##补上下一行的注释，这里是same padding,所以是28，而vaild的话就是24
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)         # ###暂时还有问题，哪天回来填坑####,,,tf.nn.relu()非线性处理,# output size 28x28x32
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)     # output size 14x14x32
## conv2 layer ##
with tf.name_scope('layer_conv2'):
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable(([5, 5, 32, 64]))    # 传入传出都改变了，变厚
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])    # 同理，改变了
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   # 卷积,来自上面的pool output size 14x14x64
    with tf.name_scope('b_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)              # pooling output size7x7x64
## func1 layer ##
with tf.name_scope('layer_func1'):
    with tf.name_scope('W_func1'):
        W_fc1 = weight_variable([7*7*64, 1024])   # mmp什么鬼
    with tf.name_scope('b_func1'):
        b_fc1 = bias_variable([1024])
            # 数据扁平化 # [n_samples, 7,7,64] ->>       [n_samples, ]
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # 有点不对，矩阵乘出来应该是？x`1024然后再加上b_fc1
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)       # !!!!!dropout在这里!!!!

## func2 layer ##
with tf.name_scope('layer_func2'):
    with tf.name_scope('W_func2'):
        W_fc2 = weight_variable([1024, 10])   # 改了改了，传入1024输出10
    with tf.name_scope('b_func2'):
        b_fc2 = bias_variable([10])    # 同理 10
            # 不需要做扁平化了
    with tf.name_scope('prediction'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)     # prediction也可以看做h_fc2，！！！这里用softmax处理

# the error between prediction and real data
with tf.name_scope('loss_cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                                  reduction_indices=[1], name='sum'), name='average_them')
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)     # Adam算法，，，只是不知道这个学习率怎么会这么小，，

sess = tf.Session()
writer = tf.summary.FileWriter("f:/ML2/", sess.graph)              # 可视化呀可视化
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
                        mnist.test.images[:1000], mnist.test.labels[:1000]))

