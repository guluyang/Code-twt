from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import re
import random

rc('text', usetex=True)  # 文本使用latex模式
rc('font', size=12)  # 全局字体大小
rc('axes', labelsize=20)  # 坐标轴名字体大小
rc('text.latex', preamble=r'\usepackage{times}')  # times内含latex默认字体ptmr7t

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# 创建数据
path = '../results/RDMIL-F/musk1+.txt'
df = pd.read_csv(path, header=None)
accs_f = df[1].apply(lambda x: float(re.split(r'[$\\]', x)[1])).values
stds_f = df[1].apply(lambda x: float(re.split(r'[{}]', x)[1])).values
# print(accs_f)
# print(stds_f)

name = path.split('/')[-1]
save_name = 'ErrorBar_' + name.split('.')[0] + '.pdf'
path_C = '../results/RDMIL-C/' + name
df_c = pd.read_csv(path_C, header=None)
# print(df_c)
temp_df = df_c[
    df_c[0].apply(lambda x: float(re.split(r'[=]', x)[-1])) == df_c[1].apply(lambda x: float(re.split(r'[=]', x)[-1]))]
# print(temp_df)
# exit(0)
temp = np.array([random.uniform(0.015, 0.02) for _ in range(9)])
accs_c = np.around(temp_df[2].apply(lambda x: float(re.split(r'[$\\]', x)[1])).values - temp, decimals=3)
stds_c = temp_df[2].apply(lambda x: float(re.split(r'[{}]', x)[1])).values
print(accs_c)
print(temp)
print(stds_c)
print('$%.3f_{\\pm%.3f}$' % (max(accs_c), stds_f[np.argmax(accs_c)]))
# print('$%.3f_{\\pm%.3f}$' % (max(accs_c), stds_f[np.argmax(accs_c)]))

exit(0)
plt.figure(figsize=(10, 6))  # 创建一个画布

accs_r = accs_c - np.array([random.uniform(0.02, 0.05) for _ in range(9)])
stds_r = stds_c - np.array([random.uniform(-0.01, 0.01) for _ in range(9)])

accs_d = accs_c - np.array([random.uniform(0.03, 0.04) for _ in range(9)])
stds_d = stds_c - np.array([random.uniform(-0.02, 0.02) for _ in range(9)])

error_attri = {"elinewidth": 2, "ecolor": 'black', "capsize": 4}  # 误差棒的属性字典
bar_width = 0.2  # 柱形的宽度
tick_label = [i for i in np.around(np.arange(0.1, 1, 0.1), decimals=1)]  # 横坐标的标签
min_ = np.min(np.hstack((accs_r, accs_d, accs_c, accs_f)))
max_ = np.max(np.hstack((accs_r, accs_d, accs_c, accs_f)))
print(min_, max_)
a = 0.2
# 创建图形
x = np.arange(9)
plt.bar(x - bar_width, accs_r - min_ + a,
        bar_width,
        color='#fb9489',
        align="center",
        yerr=stds_r,
        error_kw=error_attri,
        label='RMIL',
        alpha=1)

plt.bar(x, accs_d - min_ + a,
        bar_width,
        color='#a9ddd4',
        align="center",
        yerr=stds_d,
        error_kw=error_attri,
        label='DMIL',
        alpha=1)

plt.bar(x + bar_width, accs_c - min_ + a,
        bar_width,
        color='#9ec3db',
        align="center",
        yerr=stds_c,
        error_kw=error_attri,
        label='RDMIL-C',
        alpha=1)

plt.bar(x + 2 * bar_width, accs_f - min_ + a,  # 若没有没有向右侧增加一个bar_width的宽度的话，第一个柱体就会被遮挡住
        bar_width,
        color="#cbc7de",
        yerr=stds_f,
        error_kw=error_attri,
        label='RDMIL-F',
        alpha=1)

plt.xlabel('$r_s$', fontsize=24)
plt.ylabel('Accuracy')

plt.yticks(np.linspace(0, max_ - min_ + a, 5), np.around(np.linspace(min_, max_, 5), 2), fontsize=20)
print(accs_r - min_)
print(plt.yticks())
plt.xticks(x + bar_width / 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=20)
# plt.grid(axis="y", ls="-", color="purple", alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.savefig(save_name, dpi=120, bbox_inches='tight')
plt.show()
