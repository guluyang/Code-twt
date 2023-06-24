from sklearn.mixture import GaussianMixture
from MILframe import MIL
import time
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import euclidean_distances as eucl
import os


def compute_discer(vectors, labels):
    positive_vectors, negative_vectors = [], []
    for i in range(len(vectors)):
        if labels[i] == 1:
            positive_vectors.append(vectors[i])
        elif labels[i] == 0:
            negative_vectors.append(vectors[i])
    positive_vectors = np.array(positive_vectors)
    negative_vectors = np.array(negative_vectors)
    # 均值向量
    positive_mean = np.mean(positive_vectors, axis=0)
    negative_mean = np.mean(negative_vectors, axis=0)
    # 平均距离
    positive_dis = np.mean(eucl(positive_vectors), axis=None)
    negative_dis = np.mean(eucl(negative_vectors), axis=None)
    fenmu = positive_dis + negative_dis
    return eucl([positive_mean], [negative_mean])[0][0] / fenmu #  if fenmu > 1e-3 else 1e-3


class miFV():
    def __init__(self, dataset_path='../MILframe/data/benchmark/musk1.mat', k=2, dr=None):
        self.mil = MIL.MIL(dataset_path) if dr is None else MIL.MIL(dataset_path, dr=dr)
        self.dr = dr
        self.K = k
        self.gmm = GaussianMixture(n_components=k, covariance_type='diag')
        if dr is None:
            self.gmm.fit(self.mil.ins)
            self.weights = self.gmm.weights_  # 各个分布的权重
            self.means = self.gmm.means_  # 各个分布的均值向量
            self.covars = self.gmm.covariances_  # 各个分布的协方差矩阵
            self.D = self.K * self.mil.dimensions
        elif dr is not None:
            self.gmm.fit(self.mil.dr_ins)
            self.weights = self.gmm.weights_  # 各个分布的权重
            self.means = self.gmm.means_  # 各个分布的均值向量
            self.covars = self.gmm.covariances_  # 各个分布的协方差矩阵
            self.D = self.K * self.mil.dr_ins_len
        self.trs_vector = self.get_trs_vector()
        self.acc, self.acc_std, self.f1, self.f1_std, self.discer, self.discer_std = self.classfier_cv()

    def get_trs_vector(self):
        temp_trs_vector_matrix = []
        # for i in tqdm(range(self.mil.num_bags)):
        for i in range(self.mil.num_bags):
            temp_matrix = []
            for k in range(self.K):
                vector = []
                temp1 = self.f_xi_wk(i, k)
                temp2 = self.f_xi_uk(i, k)
                temp3 = self.f_xi_sigmak(i, k)
                vector = np.append(vector, temp1)
                vector = np.append(vector, temp2)
                vector = np.append(vector, temp3)  # 此时的vector为最开始的fv向量
                temp_matrix.append(vector)
            temp_vector = np.sum(temp_matrix, axis=0)  # 每个包转化成向量
            temp_vector = np.multiply(np.sqrt(np.abs(temp_vector)), np.sign(temp_vector))  # 带符号的开方
            temp_vector = temp_vector / np.sqrt(np.sum(np.power(temp_vector, 2)))  # 除以二范数
            temp_vector = np.append(temp_vector, self.mil.bags_label[i])
            temp_trs_vector_matrix.append(temp_vector)
        return np.array(temp_trs_vector_matrix)

    def classfier_cv(self, classfier='svm', k_for_cv=10):
        global estimator
        if classfier == 'knn':
            estimator = KNeighborsClassifier(n_neighbors=3)
        elif classfier == 'svm':
            estimator = SVC(kernel='poly')
        total_sum = 0
        total_f1_sum = 0
        temp_accuracy_list = []
        total_f1_list = []
        cv10_precision_list = []
        cv10_discer_list = []
        # for i in tqdm(range(0, k_for_cv)):
        for i in range(0, k_for_cv):
            train_index, test_index = self.mil.get_index(para_k=k_for_cv)
            temp_sum = 0
            temp_f1_list = []
            cv_discer_list = []
            for index in range(k_for_cv):  # 一轮CV
                x_train = self.trs_vector[train_index[index], :-1]
                y_train = self.trs_vector[train_index[index], -1]

                x_test = self.trs_vector[test_index[index], :-1]
                y_test = self.trs_vector[test_index[index], -1]
                discer = 0
                temp_y_test = np.where(y_test > 0, 1, 0)
                if np.sum(temp_y_test) != 0 and np.sum(temp_y_test) != len(temp_y_test) and len(temp_y_test) > 2:
                    # print(temp_y_test)
                    discer = compute_discer(x_test, temp_y_test)

                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(x_test)
                # print(y_pred.mean())

                f1 = f1_score(y_test, y_pred, zero_division=0, average='micro')
                temp_f1_list.append(f1)
                score = estimator.score(x_test, y_test)
                temp_sum += score
                cv_discer_list.append(discer)

            # print(cv_discer_list)
            temp_accuracy = temp_sum / k_for_cv
            total_sum += temp_accuracy
            temp_accuracy_list.append(temp_accuracy)
            total_f1_list.append(np.mean(temp_f1_list))
            cv10_discer_list.append(np.mean(cv_discer_list))
            # print("第 %s 次 %s CV 的平均准确度为 %s" % (i, k_for_CV, temp_accuracy))
        accuracy = total_sum / k_for_cv

        # print("%s 倍交叉验证的平均准确度为 %s" % (k_for_CV, accuracy))
        return accuracy, np.std(temp_accuracy_list, ddof=1), np.mean(total_f1_list), np.std(total_f1_list, ddof=1), \
               float(np.mean(cv10_discer_list)), float(np.std(cv10_discer_list))

    def gamma(self, k, x_gamma):
        return self.gmm.predict_proba(x_gamma.reshape(1, -1))[0, k]  # reshape(1, -1)在单向量外套一对中括号

    def f_xi_wk(self, i, k):
        temp_sum = 0
        for j in range(self.mil.bags_size[i]):
            x_ij = []
            if self.dr is None:
                x_ij = self.mil.bags[i, 0][:, :-1][j]
            elif self.dr is not None:
                x_ij = self.mil.dr_bags[i][j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) - self.weights[k]
        return 1 / np.sqrt(self.weights[k]) * temp_sum

    def f_xi_uk(self, i, k):
        temp_sum = 0
        for j in range(self.mil.bags_size[i]):
            x_ij = []
            if self.dr is None:
                x_ij = self.mil.bags[i, 0][:, :-1][j]
            elif self.dr is not None:
                x_ij = self.mil.dr_bags[i][j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) * ((x_ij - self.means[k]) / self.covars[k])
        return 1 / np.sqrt(self.weights[k]) * temp_sum

    def f_xi_sigmak(self, i, k):
        temp_sum = 0
        for j in range(self.mil.bags_size[i]):
            x_ij = []
            if self.dr is None:
                x_ij = self.mil.bags[i, 0][:, :-1][j]
            elif self.dr is not None:
                x_ij = self.mil.dr_bags[i][j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) * 1 / np.sqrt(2) * (
                    ((x_ij - self.means[k]) ** 2 / self.covars[k] ** 2) - 1)
        return 1 / np.sqrt(self.weights[k]) * temp_sum


if __name__ == '__main__':
    # start = time.process_time()
    # mifv = miFV(dataset_path='../MILframe/Corel/ten_corel_2/90horse_95bird_b_230+.mat', k=2, dr=None)
    #
    # print('Acc: $%.3f_{\\pm%.3f}$' % (mifv.acc, mifv.acc_std))
    # print('F1: $%.1f_{\\pm%.1f}$' % (mifv.f1 * 100, mifv.f1_std * 100))
    # print('Discernibility: $%.2f_{\\pm%.2f}$' % (mifv.discer, mifv.discer_std))
    # end = time.process_time()
    # print('time cost:', (end - start))

    start = time.process_time()

    news_dic = '../MILframe/Text(sparse)/normalized/'
    dataset_name_list = os.listdir(news_dic)
    for name in dataset_name_list:
        print(name.split('.')[0], end=': ')
        acc_list, acc_std_list, f1_list, f1_std_list = [], [], [], []
        for i in range(2):
            smi = miFV(dataset_path=news_dic + name, k=1, dr=None)
            acc_list.append(smi.acc)
            acc_std_list.append(smi.acc_std)
            f1_list.append(smi.f1)
            f1_std_list.append(smi.f1_std)
        max_idx = int(np.argmax(acc_list))
        print('Acc: $%.3f_{\\pm%.3f}$, F1: $%.3f_{\\pm%.3f}$' % (
            acc_list[max_idx], acc_std_list[max_idx], f1_list[max_idx], f1_std_list[max_idx]))

    end = time.process_time()
    print('Time Cost: ', (end - start))
