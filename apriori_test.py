from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) #store all the item unrepeatly

    C1.sort()
    # return map(frozenset, C1)#frozen set, user can't change it.
    return list(map(frozenset, C1))   # frozenset->不可以改变元素的集合


def scanD(D, Ck, minSupport):
    # 参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
    ssCnt = {}
    for tid in D:  # 遍历数据集
        for can in Ck:  # 遍历候选项
            if can.issubset(tid):  # 判断候选项中是否含数据集的各项
                if can not in ssCnt:
                    ssCnt[can] = 1  # 不含设为1
                else:
                    ssCnt[can] += 1  # 有则计数加1
    numItems = float(len(D))  # 数据集大小
    retList = []  # L1初始化
    supportData = {}  # 记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems#计算支持度
        if support >= minSupport:
            retList.insert(0,key)#满足条件加入L1中
        supportData[key] = support
    return retList, supportData


