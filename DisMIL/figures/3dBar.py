import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 有时没这句会报错
import seaborn as sns
import cv2
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import rc
import pandas as pd
import re

rc('text', usetex=True)  # 文本使用latex模式
rc('font', size=12)  # 全局字体大小
rc('axes', labelsize=20)  # 坐标轴名字体大小
rc('text.latex', preamble=r'\usepackage{times}')  # times内含latex默认字体ptmr7t


def RGB_to_Hex(tmp):
    rgb = [tmp[0], tmp[1], tmp[2]]  # 将RGB格式划分开来
    strs = '#'
    for i in rgb:
        num = int(i)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()
    return strs


r_1_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
r_2_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# 行为x轴, 列为y
#                 r_R=0.1,  r_R=0.3, r_R=0.5, r_R=0.7, r_R=0.9
new_SBU_inter = [[0.85636, 0.85642, 0.85602, 0.85586, 0.85437],  # r_D = 0.1
                 [0.86050, 0.86086, 0.86110, 0.86097, 0.86069],  # r_D = 0.3
                 [0.86266, 0.86332, 0.86338, 0.86318, 0.86237],  # r_D = 0.5
                 [0.86419, 0.86423, 0.86443, 0.86394, 0.86365],  # r_D = 0.7
                 [0.86421, 0.86485, 0.86450, 0.86408, 0.86275],  # r_D = 0.9
                 ]
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
path = '../results/RDMIL-C/musk1+.txt'
name = '3DBar_C_' + path.split('/')[-1].split('.')[0] + '.pdf'
df = pd.read_csv(path, header=None)
data = []
for i in np.arange(0.1, 1, 0.1):
    print(str(np.around(i, decimals=1)))
    temp_df = df[df[0] == 'rep_ratio=' + str(np.around(i, decimals=1))]
    col = np.reshape(temp_df[2].apply(lambda x: float(re.split(r'[$\\]', x)[1])).values, (-1, 1))
    # print(col)
    data.append(col)
data = np.hstack(data)
print(data)

# data = data - data.min()
# print(data)

x = r_1_list
y = r_2_list
x_tickets = [str(_x) for _x in r_1_list]
y_tickets = [str(_x) for _x in r_2_list]
xx, yy = np.meshgrid(x, y)

# 颜色
# cmap = sns.cubehelix_palette(n_colors=1, start=-0.5, rot=0.1, gamma=0.8, as_cmap=True)
cmap = sns.cubehelix_palette(n_colors=1, start=-0.5, rot=0.1, gamma=1, as_cmap=True)

norm = Normalize(vmin=data.min(), vmax=data.max())
data = norm(data)
rgbs = cv2.cvtColor(cmap(data).astype(np.float32), cv2.COLOR_RGBA2RGB) * 255
rgb_16 = []  # 16进制颜色
for i in range(len(rgbs)):
    temp = []
    for j in range(len(rgbs[0])):
        temp.append(RGB_to_Hex(rgbs[i, j]))
    rgb_16.append(temp)
rgb_16 = np.asarray(rgb_16)
xx_flat, yy_flat, acc_flat, rgb_16 = xx.ravel(), yy.ravel(), data.ravel(), rgb_16.ravel()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
print(xx_flat)
print(yy_flat)
# exit(0)
ax.bar3d(xx_flat, yy_flat, 0, 0.04, 0.04, acc_flat,
         color=rgb_16,  # 颜色
         edgecolor="white",  # 黑色描边
         shade=False)  # 加阴影

ax.set_xticks(x)
ax.set_xticklabels(x_tickets)
ax.set_yticks(y)
ax.set_yticklabels(y_tickets)
ax.axes.zaxis.set_ticks([])  # z轴不显示

# 坐标轴名
# plt.tick_params(labelsize=12)  # 调整坐标值大小
ax.set_xlabel(r"$r_r$")
ax.set_ylabel(r"$r_d$")
# ax.set_zlabel("$Intersection$")

# ax.view_init(45)  # (上下, 左右) 调整视角
plt.savefig(name, dpi=120, bbox_inches='tight')
plt.show()
