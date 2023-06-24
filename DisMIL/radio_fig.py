import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'  # 将字体设置为黑体'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

labels = np.array(['Musk1', 'Musk2', 'Elephant', 'Fox', 'Tiger', 'News.aa', 'News.cg', 'News.ss'])
dataLenth = 8  # 数据长度
data_ori = np.array([0.928, 0.922, 0.816, 0.655, 0.803, 0.862, 0.817, 0.835])
angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)  # 根据数据长度平均分割圆周长

data_pol = np.array([0.846, 0.892, 0.796, 0.611, 0.717, 0.818, 0.814, 0.772])
data_rbf = np.array([0.905, 0.844, 0.657, 0.537, 0.766, 0.861, 0.808, 0.803])
data_sig = np.array([0.810, 0.507, 0.420, 0.415, 0.575, 0.680, 0.635, 0.712])

# 闭合
data_ori = np.concatenate((data_ori, [data_ori[0]]))
angles = np.concatenate((angles, [angles[0]]))

data_pol = np.concatenate((data_pol, [data_pol[0]]))
data_rbf = np.concatenate((data_rbf, [data_rbf[0]]))
data_sig = np.concatenate((data_sig, [data_sig[0]]))


labels = np.concatenate((labels, [labels[0]]))  # 对labels进行封闭

fig = plt.figure(facecolor="white")  # facecolor 设置框体的颜色
ax = plt.subplot(111, polar=True)  # 将图分成1行1列，画出位置1的图；设置图形为极坐标图
con_data = np.vstack((data_ori, data_pol, data_rbf, data_sig))
ax.set_rlim(0, con_data.max() + 0.01)
plt.plot(angles, data_ori, '-', linewidth=2, color='#78BF65', label='MDK')
plt.plot(angles, data_pol, '--', linewidth=2, color='#D95959', label='Poly')
plt.plot(angles, data_rbf, '-.', linewidth=2, color='#F2762E', label='RBF')
plt.plot(angles, data_sig, ':', linewidth=2, color='#006060', label='Sigmoid')
# plt.fill(angles, data, facecolor='g', alpha=0.25)  # 填充两条线之间的色彩，alpha为透明度
plt.thetagrids(angles * 180 / np.pi, labels)  # 做标签
ax.set_theta_zero_location('N')
# plt.figtext(0.52,0.95,'雷达图',ha='center')   #添加雷达图标题
plt.grid(True)
plt.legend(loc='lower center', ncol=4)
plt.savefig('figures/radio.pdf', pad_inches=0, bbox_inches='tight')
plt.show()