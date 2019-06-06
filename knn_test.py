import numpy as np
import operator
"""
函数说明:创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签

"""

def createDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels
if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 打印数据集
    print(group)
    print(labels)

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

"""


def classify0(inX, dataset, labels, k):
    # inX为输入判别向量 dataset为数据集矩阵 labels为标签向量 k为k值
    # numpy函数shape方法返回矩阵行列数,shape[0]返回行数 shape[1]返回列数
    dataSetsize = dataset.shape[0]
    # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    # 就是把判别向量扩充成矩阵来和数据集矩阵进行对应运算
    diffmat = np.tile(inX, (dataSetsize, 1)) - dataset
    # 二维特征相减后平方
    sqDiffMat = diffmat ** 2
    # numpy.sum方法为矩阵内部元素相加 axis=0 为整列元素相加 axis=1 为整行元素相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开平方, 计算出距离
    distances = sqDistances ** 0.5
    # numpy.argsort()方法 返回distances中元素从小到大排列的索引值
    sortedDistIndices = distances.argsort()
    # 创建字典记录类别以及次数
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.setdefault(voteIlabel, 0) + 1
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
        # sorted()返回数组
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # KNN分类
    test_class = classify0(test, group, labels, 3)
    # 输出结果
    print(test_class)
