# -*- coding: utf-8 -*-
'''
基于tensorflow的CNN模型实现文本分类
p.s. 写在前面由于自己也只是初学，积累的不够深厚可能写的不够好不够优化(甚至还有可能根本跑不起来)233333
'''
import tensorflow as tf

# 数据的预处理，构建词向量
# 这边是构建词向量的代码
# import word2vec
# word2vec,word2vec('某分词文件.txt', '某构建好的词向量文件.bin', size=200, verbose=True)  # 定义每个输出的词向量是200的维度
# model = word2vec.load('某构建好的词向量文件.bin')
# print(model, vectors)          # 查看词向量
# model = word2vec.load('corpusWord2Vec.bin')
# index = 1000    # 举个栗子
# print (model.vocab[index])     # 查看词表中的词

# 写个正则和列表推导式把每篇文章分开，存成列表text_list = [text1, text2,,,,,,,text9999]
# 注，总有种不好的预感，，，，

# 这边是利用jieba分词,生成一个分好词的文件
# import jieba
# for text in text_list:
#     split_list = jieba.cut(text, cut_all=True)  # 在文章内分词
#     f = open('放分词文件的路径.txt', 'w')
#     for i in split_list:
#         f.write('[')
#         f.write(' '.join(split_list))
#         f.write(']')
#     f.close()
# 然后一波骚炒作手动把这个.txt后缀改成.py  另：这边代码有错误，，但我不想改了

# 处理好的数据的格式应该是这个样子: [[文本1]，[文本2]，，，，，[文本N]]
# [文本1]=[[词1], [词2],,,,,[词40]]
# [词1]=[tf.float32(1),,,,,,,tf.float32(40)]
'''写在这部分的后面
其实在这一步，我是很想根据每一小句（可以理解为一个小维度）进行分词，少的地方补0，再构建词向量
（因为我觉得这符合人脑的工作机制，但是，这将会造成那个小维度层次上的数量不同一，
而且我又不清楚详细的构造步骤，所以也写不来，但后来转念一想，在一个足够庞大的数据下，是几乎可以抵消这一差距的，
所以我这里直接按 jieba分词 与 词向量 来构造了）
'''


def pro_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})        # 预测值 全连接
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 横向比较出最大值的位置
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # 转类型，取平均
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    # 权重
    initial = tf.truncated_normal(shape, stddev=0.1)     # 产生正态分布
    return tf.Variable(initial)


def bias_variable(shape):
    # 偏差
    initial = tf.constant(0.1, shape=shape)              # 同上
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积
    # conv2d(输入，过滤器， 步长， 填充方式)
    # p.s.我其实不太懂长方形的张量是怎么扫描过去的（碰到我的那个40词的边上是否停止,我只做过正方形的），
    # 我电脑里只有28x28的手写数字集，直接改步长后也能正常的运行，后来参考网上的也有这样的，所以，这里我还是使用默认值[1 1 1 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')      # padding的方法


def max_pool_2x2(x):
    # 池化
    # max_pool(输入, 池化窗口，步长，填充方式)
    # 池化窗口我选择2x2，本来步长打算调成2 1，但是我有点怂，，，不敢
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])   # 这边参数还没调好

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 8000], name='文章')       # 每一篇文章 算它选取前40个词，每个词有200个维度
ys = tf.placeholder(tf.float32, [None, 18], name='类别')        # 设有18种类别
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
x_image = tf.reshape(xs, [-1, 40, 200, 1])                # 展开成此形状
# print(x_image, shape)  -> [n_samples, 40, 200, 1]

## conv1 layer ##    卷积一
W_conv1 = weight_variable([5, 5, 1, 320])       # [patch 5x5的像素(可以这么理解), in size， out size],,,卷积结果：高度：1->320
b_conv1 = bias_variable([320])                  # 对应biases
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # 激励函数relu() output size 40x200x320
h_pool1 = max_pool_2x2(h_conv1)     # output size 20x100x320

## conv2 layer ##
W_conv2 = weight_variable(([5, 5, 320, 640]))    # 变厚 320->640
b_conv2 = bias_variable([640])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   #outputsize 20x100x320
h_pool2 = max_pool_2x2(h_conv2)              # pooling output size 10x50x640

## conv3 layer ##          # 没试过这个数据量 点乘造成的运算速度与卷积时丢失的信息 该如何取舍，但我更愿意它快一些，所以加一层
W_conv3 = weight_variable(([5, 5, 320, 640]))    # 变厚 320->640
b_conv3 = bias_variable([640])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)   #outputsize 10x50x640
h_pool3 = max_pool_2x2(h_conv3)              # pooling output size 5x25x1280


## func1 layer ##
W_fc1 = weight_variable([5*25*1280, 1024])     # 给它一个足够大的第二维度1024
b_fc1 = bias_variable([1024])
    # 数据扁平化 # [n_samples, 5,25,1280] ->>       [n_samples, ，，，]
h_pool3_flat = tf.reshape(h_pool3, [-1, 5*25*1280])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # 矩阵乘法，n个横着的1024x竖着的1024
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)       # dropout 每次训练随机忽略部分神经元

## func2 layer ##
W_fc2 = weight_variable([1024, 18])   # 1024 ->18
b_fc2 = bias_variable([18])    # 同理 18

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# prediction即h_fc2，用softmax处理：不会完全舍弃那些不太符合的W与b
# softmax() 给每种结果附上一个0-1的数，可以简单理解成 概率

# the error between prediction and real data
# 交叉熵,即这里的loss（我也觉得这名字拗口难懂，但打出来就是这么丑装逼2333333）
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))
# 解释上行代码 就是 平均《-求和《-负的（正确值*log(预测值)）
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Adam算法，一个较小的学习率，选出交叉熵中最小的那种方案
# 该算法具体公式见我笔记


# tensorflow的基本套路，，，，，会话窗口
sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):                                 # 先假装我有十万组新闻文本（然而并没有）
    batch_xs, batch_ys =  # ！！！注，留一个x,y数据输入的接口(其实是我写不动了，，)                            # batch 批 每次处理100组数据
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(pro_accuracy(
                                 某分词产生的词向量字典[某分词好的文件的列表[:1000]], 某文本的对应标签列表[:1000] ))

# 以下为保存训练的W与b的代码
'''
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, '路径/文件.ckpt')   # 别问我ckpt是种什么后缀
    print('Save  path', save_path)

# restore variables_
# define the same shape and same type for your variable
W = tf.Variable(W.reshape((2, 3)), dtype=tf.float32, name='weights')
b = tf.Variable(b.reshape((1, 3)), dtype=tf.float32, name='biases')

# 不需要init step

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'f:/ML/save_net.ckpt')   # 举个栗子，读出这些W和b
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))
'''

# 来波总结：
# CNN卷积神经网络核心算法已完成，正如之前所言，它跑不起来，
# 缺少数据是硬伤，但就实现过程而言，其实投机取巧，借用了word2vec这一经过处理的词向量
# 当然，构建词向量本身就是深度学习做文本的副产物(我弱弱的说一句，这部分其实我也能写，就是，，，学不动了)，
# 但我就是来秀一波自己近一个月在学的东西，这种学以致用的感觉还是很爽的2333
# 方法么没什么创新，但胜在不用费脑子写正则啊弄出停用词啊再慢慢的分离它们处理出一个较为纯净的数据
# 例如文本分类的元老级算法 tf_idf算法，个人觉得，深度学习还是略胜它一筹的

# 另：我的github账号： WanderAtDusk6 极度的不定期更新,求star，蟹蟹
