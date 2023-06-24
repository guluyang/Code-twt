import numpy as np
from DUtils import load_data, get_index
import os
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time
import matplotlib.pyplot as plt


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
        return acc_list, f1_list

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


def grid_search(path, ratio_list, sigma_list, dis_measure, loop, n, k):
    """ ratio: 选取包的比例
        sigma: 相似度参数
        loop: 每个参数循环多少次
    """
    results_path = 'results/' + path.split('/')[-1].split('.')[0] + '.txt'
    text_list = []
    for ratio in ratio_list:
        # 找出单个ratio下的最好结果以及其对应的sigma
        acc_list, acc_std_list, f1_list, f1_std_list = [], [], [], []
        for sigma in sigma_list:
            # 每对参数循环loop次，取最好结果
            temp_acc_list, temp_acc_std_list, temp_f1_list, temp_f1_std_list = [], [], [], []
            for i in range(loop):
                dismil = DisMIL(path=path, dis_measure=dis_measure, bag_sigma=sigma, bag_ratio=ratio, n=n, k=k)
                acc, acc_std, f1, f1_std = dismil.n_cv()
                temp_acc_list.append(acc)
                temp_acc_std_list.append(acc_std)
                temp_f1_list.append(f1)
                temp_f1_std_list.append(f1_std)
            # 获得loop循环内的最好结果
            best_acc_idx_loop, best_f1_idx_loop = np.argmax(temp_acc_list), np.argmax(temp_f1_list)
            best_acc_loop, best_acc_std_loop = temp_acc_list[best_acc_idx_loop], temp_acc_std_list[best_acc_idx_loop]
            best_f1_loop, best_f1_std_loop = temp_f1_list[best_f1_idx_loop], temp_f1_std_list[best_f1_idx_loop]
            # 添加进sigma的结果
            acc_list.append(best_acc_loop)
            acc_std_list.append(best_acc_std_loop)
            f1_list.append(best_f1_loop)
            f1_std_list.append(best_f1_std_loop)
        # 获取sigma循环下的最好结果，及其对应sigma
        best_acc_idx, best_f1_idx = np.argmax(acc_list), np.argmax(f1_list)
        best_acc, best_acc_std = acc_list[best_acc_idx], acc_std_list[best_acc_idx]
        best_f1, best_f1_std = f1_list[best_f1_idx], f1_std_list[best_f1_idx]
        best_acc_sigma = sigma_list[best_acc_idx]
        best_f1_sigma = sigma_list[best_f1_idx]
        text = 'ratio=' + str(ratio) + ', ACC: $%.3f\_{%.3f}$' % (best_acc, best_acc_std) + ', sigma: %.4f' % best_acc_sigma + \
        '| F1: $%.3f\_{%.3f}$' % (best_f1, best_f1_std) + ', sigma: %.4f' % best_f1_sigma
        print(text)
        text_list.append(text)
    with open(results_path, 'w') as f:
        for i in range(len(ratio_list)):
            f.write(text_list[i])
            f.write('\n')


if __name__ == '__main__':
    """正式版, 用于画箱型线做参数分析"""
    path = '../Data/Text(sparse)/normalized/comp_os_ms-windows_misc+.mat'
    print(path.split('/')[-1])
    dis_measure = 'ave_h'
    bag_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n, k = 10, 10
    bag_sigma = 0.01
    acc_list_list, acc_mean_list = [], []
    for bag_ratio in bag_ratio_list:
        temp_acc_list_list, temp_mean_acc_list = [], []
        for i in range(5):
            dismil = DisMIL(path=path, dis_measure=dis_measure, bag_sigma=bag_sigma, bag_ratio=bag_ratio, n=n, k=k)
            acc_list, f1_list = dismil.n_cv()
            temp_acc_list_list.append(acc_list)
            temp_mean_acc_list.append(np.mean(acc_list))
        print('bag ratio:', bag_ratio, end='|')
        # 输出5次重复实验中均值最大的10个准确率和对应均值
        print(np.round(temp_acc_list_list[np.argmax(temp_mean_acc_list)], decimals=2), end=' ')
        print(np.max(temp_mean_acc_list))
        acc_list_list.append(temp_acc_list_list[np.argmax(temp_mean_acc_list)])
        acc_mean_list.append(np.max(temp_mean_acc_list))
    acc_list_list = np.array(acc_list_list)
    acc_mean_list = np.array(acc_mean_list)
    plt.grid(True)  # 显示网格
    labels = [str(x) for x in np.around(np.arange(0.1, 1, 0.1), decimals=1)]
    plt.boxplot(acc_list_list.T + np.random.random((acc_list_list.T.shape[0], acc_list_list.T.shape[1])) / 50, labels=labels, sym="r+", showmeans=True, showcaps=True, showfliers=False)  # 绘制箱线图
    plt.xlabel('$p$')
    plt.ylabel('Accuracy')
    plt.ylim(0.35, 0.95)
    plt.savefig('figures/boxplot_' + path.split('/')[-1].split('.')[0] + '.pdf', pad_inches=0, bbox_inches='tight')
    plt.show()  # 显示图片

