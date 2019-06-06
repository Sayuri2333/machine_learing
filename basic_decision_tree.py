from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator

"""
函数说明:创建测试数据集

Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性

"""


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


"""
函数说明:计算给定数据集的经验熵(香农熵)

Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)

"""


def calcShannonEnt(dataset):
    # 返回数据集的行数
    numberEntires = len(dataset)
    # 保存每个标签出现次数的字典
    labelCounts = {}
    # 对每一组特征向量进行统计
    for featVec in dataset:
        # 提取label信息
        currentLabel = featVec[-1]
        # 如果标签没有放入统计次数的词典 添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # label计数
        labelCounts[currentLabel] += 1
    # 初始化香农熵
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 计算概率
        prob = float(labelCounts[key]) / numberEntires
        # 公式计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无

"""


def splitDataSet(dataset, axis, value):
    # 创建返回值的数据集的列表
    retDataset = []
    # 遍历原始数据集
    for featVec in dataset:
        # 判断axis轴上的值是否为value
        if featVec[axis] == value:
            # 将符合条件的数据单元去掉axis特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            # 将其添加至返回的数据集
            retDataset.append(reducedFeatVec)
    return retDataset


"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值

"""


def chooseBestFeatureToSplit(dataset):
    # 求特征数量
    numFeatures = len(dataset[0]) - 1
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataset)
    # 初始化最优信息增益
    bestInfoGain = 0.0
    # 初始化最优特征值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 对每个特征值取其所有的取值作为数组
        featList = [example[i] for example in dataset]
        # 创建集合使得包含唯一存在的每个特征值的不同取值
        uniqueVals = set(featList)
        # 初始化条件熵
        newEntropy = 0.0
        # 对特征值的每个取值
        for value in uniqueVals:
            # subdataset中为第i个特征值取值为value时的数据集分片
            subDataset = splitDataSet(dataset, i, value)
            # prob为第i个特征值取值为value的概率
            prob = len(subDataset) / float(len(dataset))
            # subdataset的香农熵为H(Y|X)
            newEntropy += prob * calcShannonEnt(subDataset)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 输出并更新信息增益的最优值
        print(f"第{i}个特征的增益为: {infoGain}")
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def calIntrinsicValue(dataset):
    numFeature = len(dataset[0]) - 1
    intrinsicValue = []
    for i in range(numFeature):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        thisIntrinsicValue = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            thisIntrinsicValue -= prob * log(prob, 2)
        intrinsicValue.append(thisIntrinsicValue)
    return intrinsicValue


def chooseBestFeatureToSplit_0(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    infoGainList = []
    totalInfoGain = 0.0
    bestFeature = -1
    intrinsicValueList = calIntrinsicValue(dataset)
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataSet(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        # 输出并更新信息增益的最优值
        print(f"第{i}个特征的增益为: {infoGain}")
        infoGainList.append(infoGain)
        totalInfoGain += infoGain
    averageInfoGain = totalInfoGain / float(numFeatures)
    bestGainRatio = 0.0
    for i in range(len(infoGainList)):
        print(f'最优增益率为: {bestGainRatio}')
        gainRatio = 0.0
        if infoGainList[i] > averageInfoGain:
            gainRatio = float(infoGainList[i]) / intrinsicValueList[i]
            print(f"第{i}个特征的增益率为: {gainRatio}")
        if gainRatio > bestGainRatio:
            bestFeature = i
    print(f'最优特征取值为: {bestFeature}')
    return bestFeature
"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)

"""


def majorityCount(classList):
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回classList中出现次数最多的元素
    return sortedClassCount[0][0]


"""
函数说明:创建决策树

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树

"""


def createTree(dataset, labels, featLabels):
    # 选取分类标签(是否放贷)
    classList = [example[-1] for example in dataset]
    # 如果当前数据集类别相同则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果特征值已经用完了(dataset长度为1)的时候 返回此时出现次数最多的类的标签
    if len(dataset[0]) == 1:
        return majorityCount(classList)
    # 选择当前数据集中的最优特征的index值
    bestFeat = chooseBestFeatureToSplit_0(dataset)
    # 存储最优特征的标签(也就是最优特征的名字)
    bestFeatLabel = labels[bestFeat]
    # 记录已经使用过的特征标签(到时使用决策树时要按顺序输入)
    featLabels.append(bestFeatLabel)
    # 根据最优特征生成决策树分支
    mytree = {bestFeatLabel:{}}
    # 从特征标签列表中删除已经使用过的特征(当前选择的最优特征)
    del(labels[bestFeat])
    # 得到当前数据集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataset]
    # 使用集合去掉重复的属性值
    uniqueVals = set(featValues)
    # 对于当前特征的每一个取值创建新的决策树分支
    for value in uniqueVals:
        # 使用符合当前最优特征的value值的数据集(split方法筛选)进行决策树分支创建
        mytree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), labels, featLabels)
    return mytree



"""
函数说明:使用决策树分类

Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
Author:
    Jack Cui

"""


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


dataset, labels = createDataSet()
featLabels = []
myTree = createTree(dataset, labels, featLabels)
testVec = [0, 1]
result = classify(myTree, featLabels, testVec)
if result == 'yes':
    print('放贷')
if result == 'no':
    print('不放贷')