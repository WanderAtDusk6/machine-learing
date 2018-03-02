# 机器学习 hello world


from sklearn import tree                            # 原先import sklearn
features = [[140, 1], [130, 1],[150, 0], [170, 0]]  # 0 粗糙 1 光滑
lables = [0, 0, 1, 1]                               # 0 苹果 1橘子
clf = tree.DecisionTreeClassifier()                 # 创建了一个分类器
clf = clf.fit(features, lables)                     # fit 翻译作拟合最合适了
print(clf.predict([[150, 1],[160, 1], [120, 1]]))
