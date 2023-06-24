from MILframe import MIL, pathes
import numpy as np
from sklearn.cluster import KMeans
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
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
    return eucl([positive_mean], [negative_mean])[0][0] / fenmu if fenmu > 1e-3 else 1e-3


class miVLAD:
    def __init__(self,
                 dataset_path=r'..\MILframe\data\benchmark\musk1.mat',
                 k=5,
                 classifier='svm',
                 k_for_cv=10,
                 dr=None):
        self.dr = dr
        self.mil = MIL.MIL(dataset_path, dr=dr)
        self.bags = self.get_bags()  # 在每个包内实例的后面加上从0开始的id
        # print(self.bags)
        self.codebooks, self.ins_label = self.get_codebooks(k)
        self.trsed_vector = self.get_trsed_vector(k)
        self.acc, self.acc_std, \
        self.f1, self.f1_std = self.classfier_cv(classifier, k_for_cv)

    def classfier_cv(self, classfier='svm', k_for_cv=10):
        global estimator
        if classfier == 'knn':
            estimator = KNeighborsClassifier(n_neighbors=3)
        elif classfier == 'svm':
            estimator = SVC(kernel='poly')
        
        cv10_acc_list = []
        cv10_f1_list = []
        # cv10_recall_list = []
        # cv10_precision_list = []
        # cv10_discer_list = []
        
        for i in range(0, k_for_cv):
            train_index, test_index = self.mil.get_index(para_k=k_for_cv)

            cv_acc_list = []
            cv_f1_list = []
            cv_recall_list = []
            cv_precision_list = []
            cv_discer_list = []
            
            for index in range(k_for_cv):  # 一轮CV
                x_train = self.trsed_vector[train_index[index], :-1]
                y_train = self.trsed_vector[train_index[index], -1]

                x_test = self.trsed_vector[test_index[index], :-1]
                y_test = self.trsed_vector[test_index[index], -1]
                # discer = 0
                # temp_y_test = np.where(y_test > 0, 1, 0)
                # if np.sum(temp_y_test) != 0 and np.sum(temp_y_test) != len(temp_y_test) and len(temp_y_test) > 2:
                #     # print(temp_y_test)
                #     discer = compute_discer(x_test, temp_y_test)

                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(x_test)
                
                acc = estimator.score(x_test, y_test)
                f1 = f1_score(y_pred=y_pred, y_true=y_test, average='weighted', labels=np.unique(y_pred))
                # recall = recall_score(y_test, y_pred, zero_division=0)
                # precision = precision_score(y_test, y_pred)
                
                cv_acc_list.append(acc)
                cv_f1_list.append(f1)
                # cv_recall_list.append(recall)
                # cv_precision_list.append(precision)
                # cv_discer_list.append(discer)
            cv10_acc_list.append(np.mean(cv_acc_list))
            cv10_f1_list.append(np.mean(cv_f1_list))
            # cv10_recall_list.append(np.mean(cv_recall_list))
            # cv10_precision_list.append(np.mean(cv_precision_list))
            # cv10_discer_list.append(np.mean(cv_discer_list))

        return np.mean(cv10_acc_list), np.std(cv10_acc_list), \
            np.mean(cv10_f1_list) * 100, np.std(cv10_f1_list) * 100, \
            # np.mean(cv10_recall_list) * 100, np.std(cv10_recall_list) * 100, \
            # np.mean(cv10_precision_list) * 100, np.std(cv10_precision_list) * 100, \
            # float(np.mean(cv10_discer_list)), float(np.std(cv10_discer_list))

    def get_trsed_vector(self, k):
        temp_trs_vector_matrix = []
        for bag_id in self.bags:
            temp_matrix = []
            if self.dr is None:
                temp_matrix = np.zeros((k, self.mil.dimensions))  # 先用矩阵表示包
            elif self.dr is not None:
                temp_matrix = np.zeros((k, self.mil.dr_ins_len))
            for ins in self.bags[bag_id]:
                ins_id = int(ins[-1])  # 向量的id
                ins_label = self.ins_label[ins_id]  # 这个向量属于哪个中心
                # print(len(ins[:-1]))
                # print(len(self.codebooks[ins_label]))
                # print(len(temp_matrix[ins_label]))
                temp_matrix[ins_label] = np.add(ins[:-1] - self.codebooks[ins_label], temp_matrix[ins_label])

            # 将记录数组延展开
            temp_vector = []
            for i in range(len(temp_matrix)):
                for j in range(len(temp_matrix[i])):
                    temp_vector.append(temp_matrix[i, j])
            # temp_trs_vector_matrix.append(temp_vector)
            temp_vector = np.multiply(np.sqrt(np.abs(temp_vector)), np.sign(temp_vector))
            # 除以二范数
            temp_vector = temp_vector / np.sqrt(np.sum(np.power(temp_vector, 2)))
            # 在temp_vector最后一维加上包的标签
            # temp_vector.append(self.mil.bags_label[bag_id])
            if self.dr is None:
                temp_vector = np.insert(temp_vector, k * self.mil.dimensions, self.mil.bags_label[bag_id])
            elif self.dr is not None:
                temp_vector = np.insert(temp_vector, k * self.mil.dr_ins_len, self.mil.bags_label[bag_id])
            # print(temp_vector)
            temp_trs_vector_matrix.append(temp_vector)
        temp_trs_vector_matrix = np.array(temp_trs_vector_matrix)
        # print(temp_trs_vector_matrix)
        return temp_trs_vector_matrix

    def get_codebooks(self, k):
        km = KMeans(n_clusters=k)
        if self.dr is None:
            km.fit(self.mil.ins)
        elif self.dr is not None:
            km.fit(self.mil.dr_ins)
        return km.cluster_centers_, km.labels_  # 聚类中心坐标, 每个样本所属的簇

    def get_bags(self):
        temp_bags = {}
        if self.dr is None:  # 为None则不降维
            ins_id = 0
            for i in range(self.mil.num_bags):
                temp_bag = []
                for ins in self.mil.bags[i, 0][:, :-1]:
                    new_ins = []
                    for item in ins:
                        new_ins.append(item)
                    new_ins.append(ins_id)
                    ins_id = ins_id + 1
                    temp_bag.append(new_ins)
                temp_bags[i] = np.array(temp_bag)
        elif self.dr is not None:  # 不为None则降维
            ins_id = 0
            for i in range(self.mil.num_bags):
                temp_bag = []
                for ins in self.mil.dr_bags[i]:
                    new_ins = []
                    for item in ins:
                        new_ins.append(item)
                    new_ins.append(ins_id)
                    ins_id = ins_id + 1
                    temp_bag.append(new_ins)
                temp_bags[i] = np.array(temp_bag)
        return temp_bags

    def euclidean(self, ins1, ins2):
        return np.sqrt(np.sum(np.power((ins1 - ins2), 2)))

    def is_groups_value_full(self, groups):
        for i in groups:
            if not groups[i]:
                return False
        return True


if __name__ == '__main__':
    path_li = pathes.get_path_for_GT()
    for path in path_li:
        mivlad = miVLAD(dataset_path=path, k=2, dr=None)
        name = path.split('/')[-1].split('.')[0]
        print('%-25s acc: $%.3f_{\\pm%.3f}$, f1: $%.3f_{\\pm%.3f}$' % (name, mivlad.acc, mivlad.acc_std, mivlad.f1, mivlad.f1_std))

    # start = time.process_time()

    # news_dic = '../MILframe/Text(sparse)/'
    # dataset_name_list = os.listdir(news_dic)
    # for name in dataset_name_list:
    #     print(name.split('.')[0], end=': ')
    #     acc_list, acc_std_list, f1_list, f1_std_list = [], [], [], []
    #     for i in range(2):
    #         mivlad = miVLAD(dataset_path=news_dic + name, k=2, dr=None)
    #         acc_list.append(mivlad.acc)
    #         acc_std_list.append(mivlad.acc_std)
    #         f1_list.append(mivlad.f1)
    #         f1_std_list.append(mivlad.f1_std)
    #     max_idx = int(np.argmax(acc_list))
    #     print('Acc: $%.1f_{\\pm%.1f}$, F1: $%.1f_{\\pm%.1f}$' % (acc_list[max_idx], acc_std_list[max_idx], f1_list[max_idx], f1_std_list[max_idx]))
    #
    # end = time.process_time()
    # print('Time Cost: ', (end - start))

