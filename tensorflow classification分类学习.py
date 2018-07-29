# tensorflow classification分类学习
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)    # 不上中文手写数据集了，另，mnist(手写数字识别),one_hot(独热码)


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


def computer_accuracy(v_xs, v_ys):                     #
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})     # ##########mmp,苟，跪了跪了 这他妈都是什么###########
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # argmax(),,返还这个最大数的下标 \
                                                                            # ## 因为独热码，所以基于位置上最大值的选取并比较是可行的\
                                                                            #  妈蛋，自己都快看不懂自己写的东西了
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        # 放眼望去全是些不认识的函数,转化为tf.float32，mean(取平均)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result              # 输出的result是个百分比，越高，即越准确


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])    # 28*28
ys = tf.placeholder(tf.float32, [None, 10])     # 独，，，独热码？

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)    # softmax（适用于贪心学习）

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))    # loss  mmp这都是什么东西
#                    顺便撸一下英语 cross_entropy 交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)   # batch 批 每次处理100组数据
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(computer_accuracy(
                    mnist.test.images, mnist.test.labels))     # 计算下test_data的准确度，test与训练分开，确保不会互相影响

# 我去尼玛的分类学习
# 输出的值，有几个分类就能输出几个值，预测出来的值是概率的一个值