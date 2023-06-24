from MILframe import MIL
from DP_2 import DP
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class SMMD:
    """
    ratio_for_inbag_dp表示要聚类的簇数与包内总实例数的比值
    """

    def __init__(self, kernel='gaussian', ratio_for_inbag_dp=0.1, classfier='knn',
                 file_path='..\\MILframe\\data\\benchmark\\musk1.mat'):
        self.mil = MIL.MIL(file_path)
        self.trsed_vector_matrix = self.get_trsed_vector_matrix(ratio_for_inbag_dp, kernel)  # 最后一列为标签
        # print(self.trsed_vector_matrix)
        self.accuracy, self.accuracy_list, self.std = self.classfier_cv(classfier=classfier)

    # to be continued......
    def classfier_cv(self, classfier='knn', k_for_cv=10):
        global estimator
        if classfier == 'knn':
            estimator = KNeighborsClassifier(n_neighbors=3)
        elif classfier == 'svm':
            estimator = SVC(kernel='poly')
        total_sum = 0
        temp_accuracy_list = []
        for i in range(0, k_for_cv):
            train_index, test_index = self.mil.get_index(para_k=k_for_cv)
            temp_sum = 0
            for index in range(k_for_cv):  # 一轮CV
                x_train = self.trsed_vector_matrix[train_index[index], :-1]
                y_train = self.trsed_vector_matrix[train_index[index], -1]

                x_test = self.trsed_vector_matrix[test_index[index], :-1]
                y_test = self.trsed_vector_matrix[test_index[index], -1]

                estimator.fit(x_train, y_train)
                score = estimator.score(x_test, y_test)
                temp_sum += score
            temp_accuracy = temp_sum / k_for_cv
            total_sum += temp_accuracy
            temp_accuracy_list.append(temp_accuracy)
            # print("第 %s 次 %s CV 的平均准确度为 %s" % (i, k_for_CV, temp_accuracy))
        accuracy = total_sum / k_for_cv
        # print("%s 倍交叉验证的平均准确度为 %s" % (k_for_CV, accuracy))
        # print('accuracy_list', temp_accuracy_list)
        # print('classfier:', classfier)
        return accuracy, temp_accuracy_list, np.std(temp_accuracy_list, ddof=1)

    def get_trsed_vector_matrix(self, ratio, kernel):
        # print('computing transformed vector matrix........')
        trsed_vecs = []
        for i in range(self.mil.num_bags):
            if self.mil.bags_size[i] == 1:
                trsed_vecs.append(self.mil.bags[i, 0])
            else:
                trsed_vecs.append(self.get_trsed_vector(i, ratio, kernel))
        return np.array(trsed_vecs)

    def get_trsed_vector(self, i, ratio, kernel):
        temp_bag = self.mil.bags[i, 0][:, :-1]
        temp_num_ins = self.mil.bags_size[i]
        dis_matrix = np.zeros((temp_num_ins, temp_num_ins))
        for j in range(temp_num_ins):
            for k in range(temp_num_ins):
                dis_matrix[j, k] = self.dis_between_ins(temp_bag[j], temp_bag[k])
            dp = DP()
            dp.train(matrix=dis_matrix, kernel=kernel, r=ratio)
            lambda_list = dp.lambda_list
            # lambda_list = lambda_list.reshape((1, temp_num_ins))
            temp_vec = np.matmul(lambda_list, temp_bag)
            temp_vec = temp_vec / temp_num_ins
            temp_vec = temp_vec.tolist()
            # print(temp_vec)
            temp_vec.append(self.mil.bags_label[i])
        return temp_vec

    def dis_between_ins(self, ins1, ins2):
        # return np.sqrt((np.sum((ins1 - ins2) ** 2)))
        return np.sqrt(np.sum(np.power((ins1 - ins2), 2)))


if __name__ == '__main__':
    start = time.process_time()
    # tssm = SMMD(ratio_for_inbag_dp=0.5)
    # print(tssm.accuracy)
    acc_list = []
    for i in range(50):
        tssm = SMMD(ratio_for_inbag_dp=0.5)
        print(i)
        acc_list.append(tssm.accuracy)
    print(np.max(acc_list))

    end = time.process_time()
    print('本次运行耗时 %s 秒' % (end - start))
