import pandas as pd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

def get_all_list():
    file = open('B:\machine_learning\武汉大学.csv', 'r', encoding='utf-8')
    df = pd.read_csv(file, header=None, sep=',', names=['time', 'title', 'zhuanfa', 'pinglun', 'dianzan'])
    zhuanfa_list = []
    for i in df['zhuanfa'].values:
        if i != '字段3' and i != '转发':
            zhuanfa_list.append(int(i))
        elif i == '转发':
            zhuanfa_list.append(0)
    sorted_zhuanfa_list = sorted(zhuanfa_list, reverse=True)
    pinglun_list = []
    print(df['pinglun'].values)
    for i in df['pinglun'].values:
        if i != '字段4' and i != '评论':
            pinglun_list.append(int(i))
        elif i == '评论':
            pinglun_list.append(0)
    sorted_pinglun_list = sorted(pinglun_list, reverse=True)
    dianzan_list = []
    for i in df['dianzan'].values:
        if i != '字段5' and i != '赞':
            dianzan_list.append(int(i))
        elif i == '赞':
            dianzan_list.append(0)
    sorted_dianzan_list = sorted(dianzan_list, reverse=True)
    index_list = [i for i in range(len(sorted_dianzan_list))]
    return index_list, sorted_dianzan_list, sorted_zhuanfa_list, sorted_pinglun_list, dianzan_list, zhuanfa_list, pinglun_list


index_list, sorted_dianzan_list, sorted_zhuanfa_list, sorted_pinglun_list, dianzan_list, zhuanfa_list, pinglun_list = get_all_list()
font = FontProperties(fname=r'B:\machine_learning\simsun.ttc', size=14)
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(19.2, 10.8))
axs[0][0].scatter(x=index_list, y=sorted_dianzan_list, color='red', s=15, alpha=.5)
axs[0][0].scatter(x=index_list, y=sorted_zhuanfa_list, color='green', s=15, alpha=.5)
axs[0][0].scatter(x=index_list, y=sorted_pinglun_list, color='blue', s=15, alpha=.5)
axs0_title_text = axs[0][0].set_title(u'所有微博按照评论点赞转发数排序', FontProperties=font)
axs0_xlabel_text = axs[0][0].set_xlabel(u'点赞/评论/转发降序排序位次', FontProperties=font)
axs0_ylabel_text = axs[0][0].set_ylabel(u'点赞/评论/转发数', FontProperties=font)
plt.setp(axs0_title_text, size=9, weight='bold', color='red')
plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

numlist = np.zeros((len(dianzan_list), 3), dtype='int32')
numlist[:, 0] = dianzan_list
numlist[:, 1] = zhuanfa_list
numlist[:, 2] = pinglun_list

sorted_pinglun_numlist = numlist[np.lexsort(-numlist.T)]
axs[0][1].scatter(x=index_list, y= sorted_pinglun_numlist[:, 2], color='blue', s=15, alpha=.5)
axs[0][1].scatter(x=index_list, y= sorted_pinglun_numlist[:, 1], color='green', s=15, alpha=.5)
axs[0][1].scatter(x=index_list, y= sorted_pinglun_numlist[:, 0], color='red', s=15, alpha=.5)
axs1_title_text = axs[0][1].set_title(u'每条微博按照评论数排序', FontProperties=font)
axs1_xlabel_text = axs[0][1].set_xlabel(u'评论降序排序位次', FontProperties=font)
axs1_ylabel_text = axs[0][1].set_ylabel(u'点赞/评论/转发数', FontProperties=font)
plt.setp(axs1_title_text, size=9, weight='bold', color='red')
plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

sorted_dianzan_numlist = numlist[np.lexsort(-numlist[:, ::-1].T)]
axs[1][0].scatter(x=index_list, y= sorted_dianzan_numlist[:, 2], color='blue', s=15, alpha=.5)
axs[1][0].scatter(x=index_list, y= sorted_dianzan_numlist[:, 1], color='green', s=15, alpha=.5)
axs[1][0].scatter(x=index_list, y= sorted_dianzan_numlist[:, 0], color='red', s=15, alpha=.5)
axs2_title_text = axs[1][0].set_title(u'每条微博按照点赞数排序', FontProperties=font)
axs2_xlabel_text = axs[1][0].set_xlabel(u'点赞降序排序位次', FontProperties=font)
axs2_ylabel_text = axs[1][0].set_ylabel(u'点赞/评论/转发数', FontProperties=font)
plt.setp(axs2_title_text, size=9, weight='bold', color='red')
plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

sorted_zhuanfa_numlist = numlist[np.lexsort(-numlist.T[np.array([0, 2, 1])])]
print(sorted_zhuanfa_numlist)
axs[1][1].scatter(x=index_list, y= sorted_zhuanfa_numlist[:, 2], color='blue', s=15, alpha=.5)
axs[1][1].scatter(x=index_list, y= sorted_zhuanfa_numlist[:, 1], color='green', s=15, alpha=.5)
axs[1][1].scatter(x=index_list, y= sorted_zhuanfa_numlist[:, 0], color='red', s=15, alpha=.5)
axs3_title_text = axs[1][1].set_title(u'每条微博按照转发数排序', FontProperties=font)
axs3_xlabel_text = axs[1][1].set_xlabel(u'转发降序排序位次', FontProperties=font)
axs3_ylabel_text = axs[1][1].set_ylabel(u'点赞/评论/转发数', FontProperties=font)
plt.setp(axs3_title_text, size=9, weight='bold', color='red')
plt.setp(axs3_xlabel_text, size=7, weight='bold', color='black')
plt.setp(axs3_ylabel_text, size=7, weight='bold', color='black')

dianzan = mlines.Line2D([], [], color='red', marker='.', markersize=6, label=u'like')
zhuanfa = mlines.Line2D([], [], color='green', marker='.', markersize=6, label=u'share')
pinglun = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label=u'comment')

axs[0][0].legend(handles=[dianzan, zhuanfa, pinglun])
axs[0][1].legend(handles=[dianzan, zhuanfa, pinglun])
axs[1][0].legend(handles=[dianzan, zhuanfa, pinglun])
axs[1][1].legend(handles=[dianzan, zhuanfa, pinglun])

print('haha')
plt.savefig('武汉大学.png')