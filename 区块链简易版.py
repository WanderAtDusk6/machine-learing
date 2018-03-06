# -*- coding: utf-8 -*-
# 区块链简易版
import hashlib
from datetime import datetime

# 定义区块类
class Block(object):
    def __init__(self, index, time, data, pre_hash):
        self.index = index
        self.time = time
        self.data = data
        self.pre_hash = pre_hash
        self.hash = self.hash_block()    # 这里有一个hash_block()的调用，但不知道在哪
 
    def hash_block(self):
        sha = hashlib.sha256()
        sha.update('{}{}{}{}'.format(self.index, self.time, self.data, self.pre_hash))    # 暂时不知道这是干嘛的
        return sha.hexdiget()   # hash值加密，用了sha256方法，返回

# 创建初始区块
def head_block_create():
    return Block(0, datetime.now(), 'Head Block', '0')   # 这部分朕能勉强看懂，分别是刷个指针，更个日期，取个名字，初始个哈希

# 创建新的区块
def new_block(last_block):                 # ???本应该接参数列表的地方放了last_block
    _index = last_block.index + 1
    _time = datetime.now()
    _data = 'block {}'.format(index)    # 讲真，为什么是花括号我也很迷
    _hash = last_block.hash
    return Block(_index, _time, _data, _hash)    # 还有前面的短下划线

blockchain = [ head_block_create()]  # 此处报错！！！！
pre_block = blockchain[0]                 # 这边还没看，估计是创建一个什么东西
print('Block id:0 Hash value:()'.format(blockchain[0].hash))

# 待测新增区块的数量
num_block = 10

for i in range(num_block):
    one_block = new_block(pre_block)         #
    blockchain.append(one_block)
    pre_block = one_block
    print('Block id:{} Hash value:{}'.format(one_block.index, one_block.hash))
    


# 各种报错mmp

# 又是一个失败品，鉴定完毕
