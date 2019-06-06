import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
import numpy

df = pd.read_excel('C:\\Users\\HASEE\\Desktop\\YOUTUBE\\sayuri2333\\kizuna_ai\\kizuna_ai.xlsx', sheetname='sheet1')
df2 = pd.read_excel('C:\\Users\\HASEE\\Desktop\\sayuri2333\\kizuna_ai\\tokino_sora.xlsx', sheetname='sheet1')
guanlan_list = []
for i in df['观看人数'].values:
    guanlan_list.append(int(i) /10000)
dianzanbi_list = []
for i in df['点赞比'].values:
    dianzanbi_list.append(i)
guanlan_list_siro = []
for i in df2['观看人数'].values:
    guanlan_list_siro.append(int(i) / 10000)
dianzanbi_list_siro = []
for i in df2['点赞比'].values:
    dianzanbi_list_siro.append(i)
plt.scatter(x=dianzanbi_list, y=guanlan_list, color='red', s=15, alpha=.5)
plt.scatter(x=dianzanbi_list_siro, y=guanlan_list_siro, color='blue', s=15, alpha=.5)
font = FontProperties(fname=r'B:\machine_learning\simsun.ttc', size=10)
plt.xlabel("点赞/反对百分比", FontProperties=font)
plt.ylabel("观看次数(万)", FontProperties=font)
plt.savefig('youtuber.png')