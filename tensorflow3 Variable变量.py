# tensorflow3 Variable变量
import tensorflow as tf

state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)     # 一个变量加上一个常量等于变量
update = tf.assign(state, new_value)   # 把后面的值更新到前面 assign 赋值

init = tf.global_variables_initializer()    # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))      # 直接print（state）是没有用的，一定要放到run()上一下，才会出现state的结果