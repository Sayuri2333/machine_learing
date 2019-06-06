import numpy as np
from functools import reduce
"""
函数说明:创建实验样本

Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量

"""


def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型

"""


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                                    #创建一个其中所含元素都为0的向量
    for word in inputSet:                                                #遍历每个词条
        if word in vocabList:                                            #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表

"""


def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率

"""

# ---以下是没有经过拉普拉斯平滑和下溢出处理的训练函数
# def trainNB0(trainMatrix, trainCategory):
#     numTrainDocs = len(trainMatrix)                            # 计算训练的文档数目
#     numWords = len(trainMatrix[0])                            # 计算每篇文档的词条数
#     pAbusive = sum(trainCategory)/float(numTrainDocs)        # 文档属于侮辱类的概率 侮辱类总数/文档总数
#     p0Num = np.zeros(numWords)
#     p1Num = np.zeros(numWords)                              # 创建numpy.zeros数组,词条出现数初始化为0
#     p0Denom = 0.0
#     p1Denom = 0.0                            # 分母初始化为0
#     for i in range(numTrainDocs):
#         if trainCategory[i] == 1:                            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
#             # p1num是numpy数组 用来记录在分类1的条件下每个词汇出现的次数
#             p1Num += trainMatrix[i]
#             # p1Denom是用来记录分类1的句子的词汇总数的
#             p1Denom += sum(trainMatrix[i])
#         else:                                                # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
#             p0Num += trainMatrix[i]
#             p0Denom += sum(trainMatrix[i])
#     # 分类1的每个词汇出现数/分类1总词汇数 p1Vect还是numpy数组
#     p1Vect = p1Num/p1Denom
#     # 分类0每个词汇出现数/分类0总词汇数 p0Vect也是numpy数组
#     p0Vect = p0Num/p0Denom
#     return p0Vect, p1Vect, pAbusive                            # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        #文档属于侮辱类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)      #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0                            #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            # p1num是numpy数组 用来记录在分类1的条件下每个词汇出现的次数
            p1Num += trainMatrix[i]
            # p1Denom是用来记录分类1的句子的词汇总数的
            p1Denom += sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            # 取对数，防止下溢出
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive                            # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类

"""


# ---处理之前的classify函数---
# def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
#     # reduce(function, sequence) 对sequence中的元素连续使用function 第一次调用传递sequence的前两个元素 此后把前一次调用的结果和sequence的下一个元素传递给function
#     p1 = reduce(lambda x, y: x*y, vec2Classify * p1Vec) * pClass1             # 对应元素相乘  P(w0|1)*P(w1|1)*P(w2|1)......*P(1)
#     p0 = reduce(lambda x, y: x*y, vec2Classify * p0Vec) * (1.0 - pClass1)     # 对应元素相乘  P(w0|0)*P(w1|0)*P(w2|0)......*P(0)
#     print('p0:', p0)
#     print('p1:', p1)
#     if p1 > p0:
#         return 1
#     else:
#         return 0


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无

"""


def testingNB():
    listOPosts,listClasses = loadDataset()                                  # 创建实验样本
    myVocabList = createVocabList(listOPosts)                               # 创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))             # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))        # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']                                 # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))              # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')                                        # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')                                       # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']                                       # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))              # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')                                        # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')                                       # 执行分类并打印分类结果