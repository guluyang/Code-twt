import numpy as np
from DUtils import load_data, get_index
import os
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time


class DisMIL:
    def __init__(self, path, dis_measure, bag_sigma, bag_ratio, n, k, kernel):
        self.path = path
        self.dataset_name = path.split('/')[-1].split('.')[0]
        self.dis_measure = dis_measure
        self.bag_sigma = bag_sigma
        self.bag_ratio = bag_ratio
        self.n, self.k = n, k
        self.kernel = kernel
        # print(self.dataset_name + '|' + self.kernel)
        # bag_ins_idx记录每个包包含的实例，形式为: bag_idx [head_ins_idx, tail_ins_idx + 1] +1是为了方便取实例，因为左闭右开
        self.bags, self.labels, self.ins, self.bag_ins_idx = load_data(path)
        self.bags = np.array(self.bags, dtype=object)
        self.num_bags = len(self.labels)
        self.bag_dis_matrix = self.get_bag_matrix()

    def one_cv(self):
        tr_idx_list, te_idx_list = get_index(num_bags=self.num_bags, para_k=self.k, seed=None)
        acc_list, f1_list = [], []
        for i in range(self.k):
            acc, f1 = self.run(tr_idx_list[i], te_idx_list[i])
            acc_list.append(acc)
            f1_list.append(f1)
        return np.mean(acc_list), np.mean(f1_list)

    def n_cv(self):
        acc_list, f1_list = [], []
        for i in range(self.n):
            acc, f1 = self.one_cv()
            acc_list.append(acc)
            f1_list.append(f1)
        return np.mean(acc_list), np.std(acc_list, ddof=1), np.mean(f1_list), np.std(f1_list, ddof=1)

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
        svm = SVC(kernel=self.kernel)
        svm.fit(tr_vec, tr_labels)
        pred_labels = svm.predict(te_vec)
        # print(np.sum(pred_labels == te_labels) / len(te_idx))
        f1 = f1_score(te_labels, pred_labels, zero_division=True)
        return np.sum(pred_labels == te_labels) / len(te_idx), f1

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


if __name__ == '__main__':
    """正式版(用于换线性核的消融实验)"""
    path = '../Data/Text(sparse)/normalized/alt_atheism+.mat'
    dis_measure = 'ave_h'
    bag_sigma = 0.01
    bag_ratio = 0.1
    n, k = 10, 10
    kernels = ['poly', 'rbf', 'sigmoid']  # 可选'linear', 'poly', 'rbf', 'sigmoid'
    print(path.split('/')[-1])
    for kernel in kernels:
        acc, acc_std, f1, f1_std = DisMIL(path, dis_measure, bag_sigma, bag_ratio, n, k, kernel).n_cv()
        print(kernel + ': ', end='')
        print('acc: $%.3f_{\\pm%.3f}$' % (acc, acc_std))
