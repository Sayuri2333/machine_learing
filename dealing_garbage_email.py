import re
from basic_bayes import classifyNB
from basic_bayes import trainNB0
from basic_bayes import setOfWords2Vec
import random
import numpy as np
"""
函数说明:接收一个大字符串并将其解析为字符串列表

Parameters:
    无
Returns:
    无

"""


def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写


"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表

"""


def createVocabList(dataSet):
    vocabSet = set([])                      # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型

"""


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)                                        #创建一个其中所含元素都为0的向量
    for word in inputSet:                                                #遍历每个词条
        if word in vocabList:                                            #如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec


"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
"""


def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):                                                  # 遍历25个txt文件
        wordList = textParse(open('spam/%d.txt' % i, 'r').read())     # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                                                 # 标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('ham/%d.txt' % i, 'r').read())      # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)                                                 # 标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)                                    # 创建词汇表，不重复
    trainingSet = list(range(50)); testSet = []                             # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):                                                     # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        # random.uniform(x, y)随机生成一个实数位于(x, y)范围内
        randIndex = int(random.uniform(0, len(trainingSet)))                # 随机选取索引值
        testSet.append(trainingSet[randIndex])                              # 添加测试集的索引值
        del(trainingSet[randIndex])                                         # 在训练集列表中删除添加到测试集的索引值
    trainMat = []; trainClasses = []                                        # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:                                            # 遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                            # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型
    errorCount = 0                                                          # 错误分类计数
    for docIndex in testSet:                                                # 遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])           # 测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    #如果分类错误
            errorCount += 1                                                 #错误计数加1
            print("分类错误的测试集：", docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))


spamTest()



