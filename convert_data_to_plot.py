from matplotlib.font_manager import *
import pandas as pd
import matplotlib.pyplot as plt

font = FontProperties(fname=r'simsun.ttc')
df = pd.DataFrame()
df = pd.read_csv('Hindex.csv')
print(df)
print(list(df.ix[2]))
labels = []
data = []
x = ['H-like', 'H-comment', 'H-share']
print(x)
for i in list(df.index):
     labels.append(list(df.ix[i])[0])
     data.append(list(df.ix[i])[1: ])


# plt.plot(x, data[0], 'aliceblue', label=labels[0])
# plt.grid()
# plt.show()
color_list = ['b', 'c', 'g', 'm', 'r', 'k', 'y']
l = []
for i in list(df.index):
    plt.plot(x, data[i], color_list[i], linewidth=2.0, alpha=0.5, label=labels[i])
plt.legend(prop=font)
plt.grid()
plt.savefig('hindex.png')