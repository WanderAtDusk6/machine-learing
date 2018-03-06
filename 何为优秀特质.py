# 何为优秀特质
import numpy as np
import matplotlib.pyplot as plt     # 不存在dir中的pyplot

greyhounds = 500       # 灰猎犬
labs = 500
# 狗子的平均身高（单位：英寸）
grey_height = 28 + 4 * np.random.randn(greyhounds)  # 加减四英寸，别问我randn是干嘛的
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()


# 若你要训练识别（这不能叫识别吧，，，）高度这一特征，请从训练数据删除高度相关特征
#之所以这是很好的做法，是因为决策器（分类器）没有足够的智能明白e.g.厘米与身高是同一件事物
#这可能会重复计算该特征的重要性


# 对于人工智障而言，最好给它一个处理过的数据e.g.欲知信件寄送时间，给它以路程与时间，而不是经纬度




# 另：我的matplotlib怎么这么丑
# 此处代码要改，日后填坑
