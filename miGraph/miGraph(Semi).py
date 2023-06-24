import numpy as np
from MILframe.MIL import MIL
from sklearn.svm import SVC
import time
import copy
from sklearn.metrics import accuracy_score


class miGraph():
    def __init__(self, dataset_path=r'../MILframe/data/benchmark/musk1+.mat', para_gamma=0.1):
        self.mil = MIL(para_path=dataset_path)
        # get_affinity_matrics函数同时计算每个包内的距离矩阵和每个包转化成的亲和力矩阵
        self.affinity_matrics = self.get_affinity_matrics(para_gamma)
        # print('aff_matrix:')
        # print(self.affinity_matrics)
        self.mean_acc, self.acc_list, self.std = self.classify_cv(gamma=para_gamma)
        print(np.round(self.mean_acc * 100, 1))
        print(np.round(self.std * 100, 1))

    # def classify_cv(self, gamma, k=10):  # 用全部距离矩阵和标签训练的半监督版本
    #     sim_matrix = np.zeros((self.mil.num_bags, self.mil.num_bags))
    #     for i in range(self.mil.num_bags):
    #         for j in range(self.mil.num_bags):
    #             sim_matrix[i, j] = self.sim_between_bags(i, j, gamma)
    #     final_acc_list = []
    #     for n in range(k):
    #         train_idx_dict, test_idx_dict = self.mil.get_index(para_k=k)
    #         acc_list = []
    #         for m in range(k):
    #             test_idx = test_idx_dict[m]
    #             test_train_sim_matrix = sim_matrix[test_idx]
    #             estimitor = SVC(kernel='precomputed')
    #             estimitor.fit(sim_matrix, self.mil.bags_label)
    #             acc = estimitor.score(test_train_sim_matrix, self.mil.bags_label[test_idx])
    #             acc_list.append(acc)
    #             # print('第 %s 轮 %s CV第 %s 次的准确率为: %s ' % (n, k, m, acc))
    #         mean_acc = np.mean(acc_list)
    #         final_acc_list.append(mean_acc)
    #         print('第 %s 轮 %s CV的平均准确率为: %s ' % (n, k, mean_acc))
    #     return np.mean(final_acc_list), final_acc_list, np.std(final_acc_list)

    def classify_cv(self, gamma, k=10):
        sim_matrix = np.zeros((self.mil.num_bags, self.mil.num_bags))
        for i in range(self.mil.num_bags):
            for j in range(self.mil.num_bags):
                sim_matrix[i, j] = self.sim_between_bags(i, j, gamma)
        # print(sim_matrix)
        final_acc_list = []
        for n in range(k):
            train_idx_dict, test_idx_dict = self.mil.get_index(para_k=k)
            acc_list = []
            for m in range(k):
                train_idx = train_idx_dict[m]
                test_idx = test_idx_dict[m]
                train_sim_matrix = sim_matrix[train_idx]
                train_sim_matrix = train_sim_matrix[:, train_idx]
                # print(self.mil.bags_label[train_idx])
                # print(len(train_sim_matrix[0]))
                test_sim_matrix = sim_matrix[test_idx]
                test_sim_matrix = test_sim_matrix[:, train_idx]

                test_train_sim_matrix = sim_matrix[test_idx]
                # 第一次预测，得到测试集的伪标签
                estimitor = SVC(kernel='precomputed')
                estimitor.fit(train_sim_matrix, self.mil.bags_label[train_idx])
                test_pre_label = estimitor.predict(test_sim_matrix)
                total_label = copy.deepcopy(self.mil.bags_label)
                for x in range(len(test_pre_label)):  # 将伪标签当作真标签
                    total_label[test_idx[x]] = test_pre_label[x]

                estimitor = SVC(kernel='precomputed')
                estimitor.fit(sim_matrix, total_label)  # 用含有伪标签的数据训练
                pre_laebl = estimitor.predict(test_train_sim_matrix)  # 重新预测测试集
                acc = accuracy_score(self.mil.bags_label[test_idx], pre_laebl)
                acc_list.append(acc)
                # print('第 %s 轮 %s CV第 %s 次的准确率为: %s ' % (n, k, m, acc))
            mean_acc = np.mean(acc_list)
            final_acc_list.append(mean_acc)
            print('第 %s 轮 %s CV的平均准确率为: %s ' % (n, k, mean_acc))
        return np.mean(final_acc_list), final_acc_list, np.std(final_acc_list)

    def get_affinity_matrics(self, gamma):
        temp_total_matrix = []
        for i in range(self.mil.num_bags):
            temp_total_matrix.append(self.bag2matrix(i, gamma))
        return np.array(temp_total_matrix, dtype=object)

    def bag2matrix(self, i, gamma):
        ins = self.mil.bags[i, 0][:, :-1]
        num_ins = len(ins)
        dis_matrix = np.zeros((num_ins, num_ins))
        sum_dis = 0
        for j in range(num_ins):
            for k in range(num_ins):
                dis_matrix[j, k] = self.Gaussian_RBF(ins[j], ins[k], gamma=gamma)
                sum_dis += dis_matrix[j, k]
        delta = sum_dis / (num_ins ** 2)
        for j in range(num_ins):
            for k in range(num_ins):
                if dis_matrix[j, k] < delta or j == k:
                    dis_matrix[j, k] = 1
                else:
                    dis_matrix[j, k] = 0
        return dis_matrix

    def sim_between_bags(self, i, j, gamma):
        a_matrix_i = self.affinity_matrics[i]
        number_row_i = self.mil.bags_size[i]
        b_matrix_j = self.affinity_matrics[j]
        num_row_j = self.mil.bags_size[j]
        numerator = 0
        for a in range(number_row_i):
            for b in range(num_row_j):
                numerator += (1 / (np.sum(a_matrix_i[a]) * np.sum(b_matrix_j[b]))) * self.Gaussian_RBF(self.mil.bags[i, 0][:, :-1][a], self.mil.bags[j, 0][:, :-1][b], gamma=gamma)
        denominator_1 = 0
        for a in range(number_row_i):
            denominator_1 += 1 / np.sum(a_matrix_i[a])
        denominator_2 = 0
        for b in range(num_row_j):
            denominator_2 += 1 / np.sum(b_matrix_j[b])
        denominator = denominator_1 + denominator_2
        return numerator / denominator

    def Gaussian_RBF(self, ins1, ins2, gamma):
        return np.exp(-gamma * np.sum(np.power((ins1 - ins2), 2)))


if __name__ == '__main__':
    start = time.process_time()
    # musk数据集，gamma取0.5
    migraph = miGraph(para_gamma=1, dataset_path=r'../MILframe/data/benchmark/musk1+.mat')
    end = time.process_time()
    print('time cost:', end - start)