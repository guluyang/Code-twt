import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, rc
import pandas as pd
import re

rc('text', usetex=True)  # 文本使用latex模式
rc('font', size=12)  # 全局字体大小
rc('axes', labelsize=12)  # 坐标轴名字体大小
rc('text.latex', preamble=r'\usepackage{times}')

plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')

# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }

path = '../results/RDMIL-F/tiger+.txt'
name = 'Fill_F_' + path.split('/')[-1].split('.')[0] + '.pdf'
plt.figure(figsize=(6, 5))
df = pd.read_csv(path, header=None)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# print(df)
acc_means = df[1].apply(lambda x: float(re.split(r'[$\\]', x)[1])).values
acc_stds = df[1].apply(lambda x: float(re.split(r'[{}]', x)[1])).values
print('acc means: ', acc_means)
print('acc stds: ', acc_stds)
f1_means = df[2].apply(lambda x: float(re.split(r'[$\\]', x)[1])).values
f1_stds = df[2].apply(lambda x: float(re.split(r'[{}]', x)[1])).values
print('f1 means: ', f1_means)
print('f1 stds: ', f1_stds)
# exit(0)
iters = np.arange(9)
acc_r1 = list(map(lambda x: x[0] - x[1], zip(acc_means, acc_stds)))  # 下方差
acc_r2 = list(map(lambda x: x[0] + x[1], zip(acc_means, acc_stds)))  # 上方差
f1_r1 = list(map(lambda x: x[0] - x[1], zip(f1_means, f1_stds)))  # 下方差
f1_r2 = list(map(lambda x: x[0] + x[1], zip(f1_means, f1_stds)))  # 上方差
# acc
acc_color = palette(0)
plt.plot(iters, acc_means, color=acc_color, label='Accuracy', linewidth=3.0)
plt.fill_between(iters, acc_r1, acc_r2, color=acc_color, alpha=0.2)
# f1
f1_color = palette(1)
plt.plot(iters, f1_means, color=f1_color, label='F1-score', linewidth=3.0)
plt.fill_between(iters, f1_r1, f1_r2, color=f1_color, alpha=0.2)
plt.xticks(iters, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
max_ = np.max(acc_r2 + f1_r2)
min_ = np.min(acc_r1 + f1_r1)
# max_ = 0.91
# min_ = 0.76
print(min_, max_)
# max_ = np.around(max_, 1) if np.abs(max_ - (max_ // 0.1) * 0.1) >= 0.05 else (np.around(max_, 1) + 0.1)
# min_ = np.around(min_, 1) if np.abs(min_ - (min_ // 0.1) * 0.1) < 0.05 else (np.around(min_, 1) - 0.1)
print(min_, max_)
plt.yticks(np.linspace(min_, max_, 5), np.around(np.linspace(min_, max_, 5), 2))
plt.legend(loc='lower right', fontsize=18)
plt.xlabel(r'$r_p$', fontsize=24)
# plt.ylim(np.around(np.min(acc_r1 + f1_r1), decimals=1), np.around(np.max(acc_r2 + f1_r2), decimals=1))
# print(np.around(np.min(acc_r1 + f1_r1), decimals=1), np.around(np.max(acc_r2 + f1_r2), decimals=1))
# plt.axis([0.1, 0.9, 0.55, 0.95])
plt.savefig(name, dpi=120, bbox_inches='tight')
plt.show()



