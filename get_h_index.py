import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def get_all_list():
    file = open('B:\machine_learning\浙江大学.csv', 'r', encoding='utf-8')
    df = pd.read_csv(file, header=None, sep=',', names=['time', 'title', 'zhuanfa', 'pinglun', 'dianzan'], dtype=str)
    zhuanfa_list = []
    for i in df['zhuanfa'].values:
        if i == i:
            if i != '字段3' and i != '转发':
                zhuanfa_list.append(int(i))
            elif i == '转发':
                zhuanfa_list.append(0)
        else:
            zhuanfa_list.append(0)
    sorted_zhuanfa_list = sorted(zhuanfa_list, reverse=True)
    pinglun_list = []
    print(df['pinglun'].values)
    for i in df['pinglun'].values:
        if i == i:
            if i != '字段4' and i != '评论':
                pinglun_list.append(int(i))
            elif i == '评论':
                pinglun_list.append(0)
        else:
            pinglun_list.append(0)
    sorted_pinglun_list = sorted(pinglun_list, reverse=True)
    dianzan_list = []
    for i in df['dianzan'].values:
        if i == i:
            if i != '字段5' and i != '点赞':
                dianzan_list.append(int(i))
            elif i == '点赞':
                dianzan_list.append(0)
        else:
            dianzan_list.append(0)
    sorted_dianzan_list = sorted(dianzan_list, reverse=True)
    index_list = [i for i in range(len(sorted_dianzan_list))]
    return index_list, sorted_dianzan_list, sorted_zhuanfa_list, sorted_pinglun_list, dianzan_list, zhuanfa_list, pinglun_list


index_list, sorted_dianzan_list, sorted_zhuanfa_list, sorted_pinglun_list, dianzan_list, zhuanfa_list, pinglun_list = get_all_list()


def counting_H_index(index_list, other_list):
    for index, number in zip(index_list, other_list):
        if index >= number:
            return index


print('---浙江大学---')
print('点赞H指数: ' + str(counting_H_index(index_list, sorted_dianzan_list)))
print('评论H指数: ' + str(counting_H_index(index_list, sorted_pinglun_list)))
print('转发H指数: ' + str(counting_H_index(index_list, sorted_zhuanfa_list)))


