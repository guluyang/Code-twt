import numpy as np
from DUtils import load_data, get_index
import os
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from scipy.io import loadmat


class DisMIL:
    def __init__(self, path, dis_measure, bag_sigma, bag_ratio, n, k):
        self.path = path
        self.dataset_name = path.split('/')[-1].split('.')[0]
        self.dis_measure = dis_measure
        self.bag_sigma = bag_sigma
        self.bag_ratio = bag_ratio
        self.n, self.k = n, k
        # bag_ins_idx记录每个包包含的实例，形式为: bag_idx [head_ins_idx, tail_ins_idx + 1] +1是为了方便取实例，因为左闭右开
        self.bags, self.labels, self.ins, self.bag_ins_idx = load_data(path)
        self.bags = np.array(self.bags, dtype=object)
        self.num_bags = len(self.labels)
        self.bag_dis_matrix = self.get_bag_matrix()


    def get_bag_matrix(self):
        dis_path = '../Distance/' + self.dataset_name + '_' + self.dis_measure + '.npy'
        if os.path.exists(dis_path):
            # print('Load computed bag dis matrix')
            return np.load(dis_path)
        else:
            print('computing distance matrix')
            dis_matrix = np.zeros((self.num_bags, self.num_bags))
            if self.dis_measure == 'ave_h':
                for i in tqdm(range(self.num_bags), desc='computing...'):
                    for j in range(i, self.num_bags):
                        dis_matrix[i, j] = self.ave_hausdorff(i, j)
                        dis_matrix[j, i] = self.ave_hausdorff(i, j)
            np.save(dis_path, dis_matrix)
            return dis_matrix

    def ave_hausdorff(self, i, j):
        if i == j:
            return 0
        temp_1 = np.sum(np.min(eucl(self.bags[i], self.bags[j]), axis=1))
        temp_2 = np.sum(np.min(eucl(self.bags[j], self.bags[i]), axis=1))
        result = (temp_1 + temp_2) / (self.bags[i].shape[0] + self.bags[j].shape[0])
        return result

    def compute_representation(self, matrix, r=0.2):
        """返回每一个包的lambda值"""
        d_c = r * np.max(matrix)
        num_bags = matrix.shape[0]
        # 计算所有包的局部密度
        local_density_list = np.sum(1 / (np.exp(np.power(matrix / d_c, 2))), axis=1)
        dis_to_master_list = np.zeros(num_bags)
        for i in range(num_bags):
            temp_idx = np.argwhere(local_density_list > local_density_list[i])  # 所有密度比i大的索引
            if len(temp_idx) == 0:  # 如果局部密度最大，则其dis_to_master也设置为最大
                dis_to_master_list[i] = np.max(matrix)
                continue
            temp_dis = matrix[i][temp_idx]  # 与密度比i大的样本的距离
            dis_to_master_list[i] = np.min(temp_dis)
        dis_to_master_list = np.array(dis_to_master_list)
        lambda_list = local_density_list * dis_to_master_list
        return lambda_list

    def run(self, tr_idx, te_idx):
        tr_dis_matrix = self.bag_dis_matrix[tr_idx]
        tr_dis_matrix = tr_dis_matrix[:, tr_idx]

        tr_labels, te_labels = self.labels[tr_idx], self.labels[te_idx]
        temp = np.where(tr_labels > 0, 1, -1).reshape(1, -1)
        Q = temp.T @ temp  # 标签相同为1，标签不同为-1
        # 标签相同的包对数(|A|)
        label_sim_num = np.argwhere(Q == 1).shape[0]
        # 标签不同的包对数(|B|)
        label_dissim_num = np.argwhere(Q == -1).shape[0]
        Q = np.where(Q == 1, -1 / label_sim_num, 1 / label_dissim_num)
        # Q = Q * len(tr_idx) ** 2  # 数值太小，乘以总的包对数(一个尝试)
        Q = Q / np.exp(tr_dis_matrix)
        # np.set_printoptions(threshold=np.inf)
        D = np.diag(np.sum(Q, axis=0))
        L = D - Q
        del D, Q
        # 根据公式选择包
        all_bag_map_vector = np.exp(- tr_dis_matrix ** 2 / self.bag_sigma ** 2)
        # all_bag_map_vector = tr_dis_matrix
        bag_score = np.diagonal(all_bag_map_vector.T @ L @ all_bag_map_vector)
        bag_representation = self.compute_representation(tr_dis_matrix, r=0.2)
        bag_score = bag_score * bag_representation  # 添加了代表性的要好一点
        if self.bag_ratio == 1:
            bag_idx = np.argsort(bag_score)
        else:
            bag_idx = np.argsort(bag_score)[- int(np.floor(len(tr_idx) * self.bag_ratio)):]
        # 训练包基于所选包的映射向量
        tr_vec = np.exp(-self.bag_dis_matrix ** 2 / self.bag_sigma ** 2)[tr_idx, :][:, bag_idx]
        # 测试包基于所选包的映射向量
        te_vec = np.exp(-self.bag_dis_matrix ** 2 / self.bag_sigma ** 2)[te_idx, :][:, bag_idx]
        # 用内积计算向量相似度, 与直接使用linear核的SVM等价
        # tr_sim = tr_vec @ tr_vec.T + 1
        # tr_te_sim = te_vec @ tr_vec.T + 1
        # 调用SVM训练
        svm = SVC(kernel='linear')
        svm.fit(tr_vec, tr_labels)
        pred_labels = svm.predict(te_vec)
        acc = np.sum(pred_labels == te_labels) / len(te_idx)
        print('降维前准确率', acc)
        f1 = f1_score(te_labels, pred_labels, zero_division=True)
        # 降维画决策边界图
        # 先降维，再重新训练，再获得svm.coef_画出二维决策曲线(支持向量也要标出)
        pca = PCA(n_components=2)
        pca_tr_vec = pca.fit_transform(tr_vec)
        pca_te_vec = pca.transform(te_vec)
        pca_svm = SVC(kernel='linear')
        pca_svm.fit(pca_tr_vec, tr_labels)
        pca_acc = pca_svm.score(pca_te_vec, te_labels)
        print('降维后准确率:', pca_acc)
        print(pca_svm.coef_)  # 权重
        print(pca_svm.intercept_)  # 偏置
        po_idx = np.where(te_labels == 1)
        ne_idx = np.where(te_labels == 0)
        # 红色代表正
        plt.scatter(x=pca_te_vec[po_idx][:, 0], y=pca_te_vec[po_idx][:, 1], c='#F23030', label='Positive', s=66)
        # 绿色代表负
        plt.scatter(x=pca_te_vec[ne_idx][:, 0], y=pca_te_vec[ne_idx][:, 1], c='#1D594E', label='Negative', s=66)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 在最大值和最小值之间形成30个规律的数据
        axisx = np.linspace(xlim[0], xlim[1], 50)
        axisy = np.linspace(ylim[0], ylim[1], 50)
        # 使用meshgrid函数将两个一维向量转换为特征矩阵
        axisy, axisx = np.meshgrid(axisy, axisx)
        xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
        # 建模，通过fit计算出对应的决策边界
        Z = pca_svm.decision_function(xy).reshape(axisx.shape)
        # 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
        ax.contour(axisx, axisy, Z
                   , colors="k"
                   , levels=[-1, 0, 1]  # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
                   , alpha=1  # 透明度
                   , linestyles=['--', '-', '--'])
        ax.set_xlim(xlim)  # 设置x轴取值
        ax.set_ylim(ylim)
        ax.set_facecolor('lightgray')  # 设置视图背景颜⾊

        # ax.set_xlim((xlim[0]*2, xlim[1]*2))
        # ax.set_ylim((ylim[0]*2, ylim[1]*2))
        plt.legend(loc='upper right', prop={'size': 16})
        filename = path.split('/')[-1].split('.')[0] + '_' + str(bag_ratio) + '_' + str(np.round(acc, 3)) + '_' + str(np.round(pca_acc, 3)) + '.pdf'
        plt.savefig('figures/' + filename, pad_inches=0, bbox_inches='tight')
        plt.show()
        return np.sum(pred_labels == te_labels) / len(te_idx), f1


if __name__ == '__main__':
    """用于降维后画图的正式版"""
    path = '../Data/Text(sparse)/normalized/alt_atheism+.mat'
    dis_measure = 'ave_h'
    bag_ratio = 0.9
    n, k = 10, 10
    bag_sigma = 0.01
    num_bag = len(loadmat(path)['data'])
    tr_idx_list, te_idx_list = get_index(num_bags=num_bag, para_k=2, seed=66)
    acc, f1 = DisMIL(path, dis_measure, bag_sigma, bag_ratio, n, k).run(tr_idx_list[0], te_idx_list[0])