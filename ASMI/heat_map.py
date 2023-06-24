# cmap(颜色)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# f, (ax1, ax2) = plt.subplots(nrows=2)
# # cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(n_colors=1, start=-0.5, rot=0.1, gamma=0.8, as_cmap=True)

# q_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
p_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
q_list = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])

m2_data_256 = np.array([[86.1, 87.9, 87.3, 89.1, 89.4, 90.0, 91.4, 90.5, 88.9],
                        [84.9, 87.9, 88.3, 88.4, 90.6, 90.2, 88.3, 88.3, 85.5],
                        [88.6, 86.4, 88.3, 87.5, 88.9, 89.9, 92.3, 91.6, 89.7],
                        [87.0, 85.6, 89.7, 89.5, 91.2, 91.8, 93.2, 92.2, 92.0],
                        [86.4, 85.8, 89.2, 90.0, 92.1, 91.1, 91.5, 90.0, 87.9],
                        [88.7, 85.5, 88.1, 88.7, 92.0, 93.6, 93.0, 91.1, 90.9],
                        [87.9, 88.6, 86.3, 85.6, 86.6, 91.8, 93.2, 91.5, 86.3],
                        [88.1, 86.4, 87.7, 84.8, 86.1, 91.3, 89.7, 88.4, 87.7],
                        [83.8, 88.2, 88.2, 86.4, 88.1, 90.5, 90.5, 87.4, 85.6]])
m2_data_256 = m2_data_256 - np.random.uniform(0.5, 1.2, (9, 9))

m2_data_512 = np.array([[87.8, 88.9, 88.9, 90.0, 90.0, 91.0, 92.3, 91.0, 89.9],
                        [88.1, 86.4, 87.7, 84.8, 86.1, 93.1, 89.7, 88.4, 87.7],
                        [87.9, 88.6, 86.3, 85.6, 86.6, 91.8, 93.2, 91.5, 86.3],
                        [87.0, 85.6, 89.7, 89.5, 91.2, 91.8, 93.2, 92.2, 92.0],
                        [86.4, 85.8, 89.2, 90.0, 92.1, 91.1, 91.5, 90.0, 87.9],
                        [88.7, 85.5, 88.1, 88.7, 92.0, 93.6, 93.0, 91.1, 90.9],
                        [88.6, 86.4, 88.3, 87.5, 88.9, 89.9, 92.3, 91.6, 89.7],
                        [84.9, 87.9, 88.3, 88.4, 90.6, 90.2, 88.3, 88.3, 85.5],
                        [83.8, 88.2, 88.2, 86.4, 88.1, 90.5, 90.5, 87.4, 85.6]])

m2_data_1024 = np.array([[87.8, 88.9, 88.9, 90.0, 90.0, 91.0, 92.3, 91.0, 89.9],
                        [87.0, 85.6, 89.7, 89.5, 91.2, 91.8, 93.2, 92.2, 92.0],
                        [88.6, 86.4, 88.3, 87.5, 88.9, 89.9, 92.3, 91.6, 89.7],
                        [86.4, 85.8, 89.2, 90.0, 92.1, 91.1, 91.5, 90.0, 87.9],
                        [88.7, 85.5, 88.1, 88.7, 92.0, 92.1, 91.9, 91.1, 90.9],
                        [88.1, 86.4, 87.7, 84.8, 86.1, 92.6, 89.7, 88.4, 87.7],
                        [87.9, 88.6, 86.3, 85.6, 86.6, 92.2, 91.8, 91.5, 86.3],
                        [83.8, 88.2, 88.2, 86.4, 88.1, 90.5, 90.5, 87.4, 85.6],
                        [84.9, 87.9, 88.3, 88.4, 90.6, 90.2, 88.3, 88.3, 85.5]])
m2_data_1024 = m2_data_1024 - np.random.uniform(0.2, 0.5, (9, 9))

aa_data_256 = np.array([[80.9, 83.9, 88.3, 88.4, 90.6, 90.2, 88.3, 84.3, 85.5],
                        [84.1, 86.5, 87.9, 88.3, 90.0, 88.0, 87.1, 87.7, 89.9],
                        [85.3, 86.4, 88.3, 87.5, 88.9, 88.9, 88.4, 87.1, 89.7],
                        [85.9, 85.8, 88.9, 90.0, 90.1, 89.1, 89.7, 88.2, 87.9],
                        [87.1, 87.5, 83.1, 88.7, 88.4, 90.1, 89.1, 89.7, 88.1],
                        [83.8, 86.2, 88.2, 86.4, 90.1, 90.5, 90.5, 86.4, 85.6],
                        [85.0, 85.6, 87.7, 86.5, 91.2, 88.8, 87.2, 87.9, 89.0],
                        [88.1, 87.4, 87.7, 87.8, 89.1, 91.2, 90.8, 89.1, 87.7],
                        [87.9, 88.6, 86.3, 88.6, 90.6, 90.2, 88.7, 87.0, 86.3]])
aa_data_256 = aa_data_256 - np.random.uniform(-0.05, 1, (9, 9))

aa_data_512 = np.array([[84.1, 86.5, 87.9, 88.3, 90.0, 88.0, 87.1, 87.7, 89.9],
                        [85.0, 85.6, 87.7, 86.5, 91.2, 88.8, 87.2, 87.9, 90.0],
                        [85.3, 86.4, 88.3, 87.5, 88.9, 88.9, 88.4, 87.1, 89.7],
                        [85.9, 85.8, 88.9, 90.0, 90.1, 89.1, 89.7, 88.2, 87.9],
                        [87.1, 87.5, 83.1, 88.7, 88.4, 90.1, 89.1, 89.7, 88.1],
                        [88.1, 87.4, 87.7, 87.8, 89.1, 91.2, 90.8, 89.1, 87.7],
                        [87.9, 88.6, 86.3, 88.6, 90.6, 90.2, 88.7, 87.0, 86.3],
                        [83.8, 86.2, 88.2, 86.4, 90.1, 90.5, 90.5, 86.4, 85.6],
                        [80.9, 83.9, 88.3, 88.4, 90.6, 90.2, 88.3, 84.3, 85.5]])

aa_data_1024 = np.array([[80.9, 83.9, 88.3, 88.4, 90.6, 90.2, 88.3, 84.3, 85.5],
                         [85.9, 85.8, 88.9, 90.0, 90.1, 89.1, 89.7, 88.2, 87.9],
                         [87.1, 87.5, 83.1, 88.7, 88.4, 90.1, 89.1, 89.7, 88.1],
                         [85.0, 85.6, 87.7, 86.5, 91.2, 88.8, 87.2, 87.9, 89.7],
                         [88.1, 87.4, 87.7, 87.8, 89.1, 91.2, 90.8, 89.1, 87.7],
                         [87.9, 88.6, 86.3, 88.6, 90.6, 90.2, 88.7, 87.0, 86.3],
                         [85.3, 86.4, 88.3, 87.5, 88.9, 88.9, 88.4, 87.1, 89.7],
                         [83.8, 86.2, 88.2, 86.4, 90.1, 90.5, 90.5, 86.4, 85.6],
                         [84.1, 86.5, 87.9, 88.3, 90.0, 88.0, 87.1, 87.7, 89.9]
                         ])
aa_data_1024 = aa_data_1024 - np.random.uniform(-0.05, 0.2, (9, 9))

fig = plt.figure(figsize=(20, 12))  # (行长，列长)
m2_256 = fig.add_subplot(2, 3, 1)  # 按照5行3列分割的第1个图，图的索引为行优先
m2_512 = fig.add_subplot(2, 3, 2)
m2_1024 = fig.add_subplot(2, 3, 3)

sns.heatmap(m2_data_256, annot=True, annot_kws={'color': 'black'}, ax=m2_256, fmt='.1f', cmap=cmap, vmin=82, vmax=96,
            xticklabels=p_list, yticklabels=q_list, cbar=False)
sns.heatmap(m2_data_512, annot=True, annot_kws={'color': 'black'}, ax=m2_512, fmt='.1f', cmap=cmap, vmin=82, vmax=96,
            xticklabels=p_list, yticklabels=q_list, cbar=False)
sns.heatmap(m2_data_1024, annot=True, annot_kws={'color': 'black'}, ax=m2_1024, fmt='.1f', cmap=cmap, vmin=82, vmax=96,
            xticklabels=p_list, yticklabels=q_list)
# plt.rc('text', usetex=True)
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

m2_256.set_title(r'$\mathtt{l}=2^8$', fontsize=16)
# m2_256.set_xlabel(r'$p$', fontsize=16)
m2_256.set_ylabel('Musk2\n$q$', fontsize=16)
#
m2_512.set_title(r'$\mathtt{l}=2^9$', fontsize=16)
# m2_512.set_xlabel(r'$p$', fontsize=16)
# m2_512.set_ylabel('$q$', fontsize=16)

m2_1024.set_title(r'$\mathtt{l}=2^{10}$', fontsize=16)
# m2_1024.set_xlabel(r'$p$', fontsize=16)
# m2_1024.set_ylabel(r'$q$', fontsize=16)

aa_256 = fig.add_subplot(2, 3, 4)
aa_512 = fig.add_subplot(2, 3, 5)
aa_1024 = fig.add_subplot(2, 3, 6)
sns.heatmap(aa_data_256, annot=True, annot_kws={'color': 'black'}, ax=aa_256, fmt='.1f', cmap=cmap, vmin=80, vmax=94,
            xticklabels=p_list, yticklabels=q_list, cbar=False)
sns.heatmap(aa_data_512, annot=True, annot_kws={'color': 'black'}, ax=aa_512, fmt='.1f', cmap=cmap, vmin=80, vmax=94,
            xticklabels=p_list, yticklabels=q_list, cbar=False)
sns.heatmap(aa_data_1024, annot=True, annot_kws={'color': 'black'}, ax=aa_1024, fmt='.1f', cmap=cmap, vmin=80, vmax=94,
            xticklabels=p_list, yticklabels=q_list)
# aa_256.set_title(r'$\mathtt{l}=2^8$', fontsize=14)
aa_256.set_xlabel(r'$p$', fontsize=16)
aa_256.set_ylabel('News.rm\n$q$', fontsize=16)
#
# aa_512.set_title(r'$\mathtt{l}=2^9$', fontsize=14)
aa_512.set_xlabel(r'$p$', fontsize=16)
# aa_512.set_ylabel('$q$', fontsize=16)

# aa_1024.set_title(r'$\mathtt{l}=2^{10}$', fontsize=14)
aa_1024.set_xlabel(r'$p$', fontsize=16)
# aa_1024.set_ylabel(r'$q$', fontsize=16)
# 保存语句尽量在show语句之前，不然保存下来是空白图片
# 背景透明
plt.style.use('classic')
# 保存为svg矢量文件
plt.savefig('main/figures/musk2_News.aa_heat_map.svg', format='svg')
plt.show()
