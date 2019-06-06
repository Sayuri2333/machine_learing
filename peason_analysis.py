from math import sqrt
from sympy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'simsun.ttc', size=14)
def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den


hindex_list = pd.read_csv('h.csv', encoding='utf-8')
hindex_dianzan_list = hindex_list['点赞h指数']
hindex_pinglun_list = hindex_list['评论h指数']
hindex_zhuanfa_list = hindex_list['转发h指数']
# weight_list = [0.33, 0.33, 0.34]
# hindex_total_list = weight_list[0] * hindex_dianzan_list + weight_list[1] * hindex_pinglun_list + weight_list[2] * hindex_zhuanfa_list
# print(hindex_total_list)
# regression_list = [corrcoef(list(hindex_dianzan_list), list(hindex_total_list)), corrcoef(list(hindex_pinglun_list), list(hindex_total_list)), corrcoef(list(hindex_zhuanfa_list), list(hindex_total_list))]
# print(regression_list)
# print(regression_list[0] * regression_list[1] * regression_list[2])

two_list = []
four_list = []
six_list = []
eight_list = []
max_result = 0
for x in range(0, 100):
    if x == 33:
        print("x")
    index = 100 - x
    x = x / 100
    for y in range(0, index):
        y = y / 100
        weight_list = [x, y, 1 - x - y]
        hindex_total_list = weight_list[0] * hindex_dianzan_list + weight_list[1] * hindex_pinglun_list + weight_list[2] * hindex_zhuanfa_list
        regression_list = [corrcoef(list(hindex_dianzan_list), list(hindex_total_list)),
                           corrcoef(list(hindex_pinglun_list), list(hindex_total_list)),
                           corrcoef(list(hindex_zhuanfa_list), list(hindex_total_list))]
        result = regression_list[0] * regression_list[1] * regression_list[2]
        if 0.6 < result <= 0.75:
            six_list.append((x, y))
        elif 0.75 < result < 0.8:
            eight_list.append((x, y))
        if max_result < result:
            max_result = result
            max_list = weight_list



A = np.matrix(six_list)
print(A)
plt.scatter(x=A[:, 0].tolist(), y=A[:, 1].tolist(), color='red', s=15, alpha=.5)
A = np.matrix(eight_list)
print(A)
plt.scatter(x=A[:, 0].tolist(), y=A[:, 1].tolist(), color='blue', s=15, alpha=.5)
plt.xlabel(u'权重1的取值', FontProperties=font)
plt.ylabel(u'权重2的取值', FontProperties=font)
print(max_result)
print(max_list)
plt.savefig('fig.png')
