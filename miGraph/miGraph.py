import numpy as np
from MILframe.MIL import MIL
from sklearn.svm import SVC
import time
import os
from sklearn.metrics import f1_score


class miGraph():
    def __init__(self, dataset_path=r'../MILframe/data/benchmark/musk1+.mat', para_gamma=0.1):
        self.mil = MIL(para_path=dataset_path)
        # get_affinity_matrics函数同时计算每个包内的距离矩阵和每个包转化成的亲和力矩阵
        self.affinity_matrics = self.get_affinity_matrics(para_gamma)
        # print(self.affinity_matrics)
        self.mean_acc, self.acc_list, self.std, self.f1, self.f1_std = self.classify_cv(para_gamma)
        # print('mean acc: ', self.mean_acc)
        # print('std: ', self.std)
        # print(dataset_path.split('/')[-1], end=': ')
        # print('acc: $%.3f_{\\pm%.3f}$' % (self.mean_acc, self.std), end=' | ')

    def classify_cv(self, gamma, k=10):
        sim_matrix = np.zeros((self.mil.num_bags, self.mil.num_bags))
        for i in range(self.mil.num_bags):
            for j in range(self.mil.num_bags):
                sim_matrix[i, j] = self.sim_between_bags(i, j, gamma)
        # print(sim_matrix)
        final_acc_list = []
        final_f1_list = []
        for n in range(k):
            train_idx_dict, test_idx_dict = self.mil.get_index(para_k=k)
            acc_list = []
            f1_list = []
            for m in range(k):
                train_idx = train_idx_dict[m]
                test_idx = test_idx_dict[m]
                train_sim_matrix = sim_matrix[train_idx]
                train_sim_matrix = train_sim_matrix[:, train_idx]
                # print(self.mil.bags_label[train_idx])
                # print(len(train_sim_matrix[0]))
                test_sim_matrix = sim_matrix[test_idx]
                test_sim_matrix = test_sim_matrix[:, train_idx]
                estimitor = SVC(kernel='precomputed')
                estimitor.fit(train_sim_matrix, self.mil.bags_label[train_idx])
                pred = estimitor.predict(test_sim_matrix)
                acc = sum(pred == self.mil.bags_label[test_idx]) / len(pred)
                acc_list.append(acc)
                f1_list.append(f1_score(self.mil.bags_label[test_idx], pred, zero_division=0, average='micro'))
                # print('第 %s 轮 %s CV第 %s 次的准确率为: %s ' % (n, k, m, acc))
            mean_acc = np.mean(acc_list)
            final_acc_list.append(mean_acc)
            final_f1_list.append(np.mean(f1_list))
            # print('第 %s 轮 %s CV的平均准确率为: %s ' % (n, k, mean_acc))
        return np.float(np.mean(final_acc_list)), final_acc_list, np.float(np.std(final_acc_list, ddof=1)), \
               np.float(np.mean(final_f1_list)), np.float(np.std(final_f1_list, ddof=1))

    def get_affinity_matrics(self, gamma):
        temp_total_matrix = []
        for i in range(self.mil.num_bags):
            temp_total_matrix.append(self.bag2matrix(i, gamma))
        return temp_total_matrix

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
    # start = time.process_time()
    # path = '../MILframe/Corel/ten_corel_2/90horse_95bird_b_230+.mat'
    # migraph = miGraph(para_gamma=1, dataset_path=path)
    # end = time.process_time()
    # print('time cost:', end - start)
    path_list = ['../MILframe/Benchmark/musk1+.mat',
                 '../MILframe/Benchmark/musk2+.mat',
                 '../MILframe/Benchmark/elephant+.mat',
                 '../MILframe/Benchmark/fox+.mat',
                 '../MILframe/Benchmark/tiger+.mat']
    dic = '../MILframe/Text(sparse)/normalized/'
    path_list_2 = os.listdir(dic)
    # for i in path_list_2:
    #     path_list.append(dic + i)
    for path in path_list:
        start = time.process_time()
        print(path.split('/')[-1].split('.')[0] + ': ')
        migraph = miGraph(para_gamma=1, dataset_path=path)
        print('Acc: $%.3f_{\\pm%.3f}$, F1: $%.3f_{\\pm%.3f}$' % (
            migraph.mean_acc, migraph.std, migraph.f1, migraph.f1_std))
        end = time.process_time()
        print('time cost:', end - start)
