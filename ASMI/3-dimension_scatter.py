##画个简单三维图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure().add_subplot(111, projection='3d')
# 基于ax变量绘制三维图
# xs表示x方向的变量
# ys表示y方向的变量
# zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
# m表示点的形式，o是圆形的点，^是三角形（marker)
# c表示颜色（color for short）
# l
xs = [16, 16, 16, 16, 16, 16, 16, 16, 16,
      32, 32, 32, 32, 32, 32, 32, 32, 32,
      64, 64, 64, 64, 64, 64, 64, 64, 64,
      128, 128, 128, 128, 128, 128, 128, 128, 128,
      256, 256, 256, 256, 256, 256, 256, 256, 256,
      512, 512, 512, 512, 512, 512, 512, 512, 512,
      1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
# p
ys = [1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      1, 2, 3, 4, 5, 6, 7, 8, 9]
# 准确率
zs = [91.8, 91.2, 94.2, 93.4, 92.8, 92.6, 92.8, 94.4, 94.0,
      90.6, 94.2, 92.2, 90.6, 94.2, 92.8, 93.8, 94.2, 94.4,
      92.4, 91.4, 92.4, 94.2, 93.6, 93.8, 92.2, 93.4, 93.8,
      93.0, 92.0, 93.8, 93.2, 93.8, 94.2, 93.2, 94.0, 94.6,
      92.6, 94.0, 92.8, 93.8, 92.8, 94.2, 92.8, 94.8, 93.2,
      93.6, 94.0, 94.0, 95.8, 94.0, 94.0, 94.2, 93.8, 94.0,
      93.2, 95.0, 94.6, 95.4, 93.8, 93.8, 94.0, 94.4, 93.2]
ax.scatter(xs, ys, zs, c='b', marker='o')

# 设置坐标轴数组
# ax.set_xlim3d(16, 1024)
ax.set_ylim3d(1, 9)

# 设置坐标轴名字
ax.set_xlabel(r'$\mathtt{l}$', fontsize=15)
ax.set_ylabel(r'$q$', fontsize=15)
ax.set_zlabel('Accuracy', fontsize=15)
plt.title('musk1', y=-0.1, fontsize=15)
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.fontset'] = 'stix'
# 背景透明
plt.style.use('classic')
# 保存为svg矢量文件
plt.savefig('main/figures/musk1.svg', format='svg')
# 显示图像
plt.show()

# ax = plt.figure().add_subplot(111, projection='3d')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')