from MILframe import MIL, pathes
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import euclidean_distances as eucl



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


def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    return round(value * 100) / 100.0


class SMI():
    def __init__(self, file_path='..\\MILframe\\data\\data\\process.mat', dr=None):
        self.mil = MIL.MIL(file_path, dr=dr)
        self.dr = dr
        self.trsed_vector_matrix = self.get_trsed_vector_matrix()
        # print(self.trsed_vector_matrix)
        self.acc, self.acc_std, \
        self.f1, self.f1_std, \
        self.recall, self.recall_std, \
        self.precision, self.precision_std, self.discer, self.discer_std = self.KNN_CV()

    def get_trsed_vector_matrix(self):
        # print('computing transformed vector。。。。。。。。')
        temp_trsed_vector_matrix = []
        # for i in tqdm(range(self.mil.num_bags), desc='Computing Transformed Vectors'):
        for i in range(self.mil.num_bags):
            temp_trsed_vector_matrix.append(self.get_trsed_vector(i))
        return np.array(temp_trsed_vector_matrix)

    def get_trsed_vector(self, i):  # 返回的单示例要带标签。
        instances = self.mil.bags[i, 0][:, :-1]
        vector = np.zeros(self.mil.dimensions)
        if self.dr is not None:  # 如果降维参数不为None，则用降维后的包和向量长度。
            instances = self.mil.dr_bags[i]
            vector = np.zeros(self.mil.dr_ins_len)
        for ins in instances:
            vector = np.add(vector, ins)
        vector = vector / self.mil.bags_size[i]
        vector = vector.tolist()
        vector.append(self.mil.bags_label[i])  # 加上包的标签。
        return np.array(vector)

    def KNN_CV(self, n_for_knn=3, k_for_CV=10):

        cv10_acc_list = []
        cv10_f1_list = []
        cv10_recall_list = []
        cv10_precision_list = []
        cv10_discer_list = []

        for i in range(0, k_for_CV):
            train_index, test_index = self.mil.get_index(para_k=k_for_CV)
            estimator = SVC()

            cv_acc_list = []
            cv_f1_list = []
            cv_recall_list = []
            cv_precision_list = []
            cv_discer_list = []

            for index in range(k_for_CV):  # 一轮CV
                x_train = self.trsed_vector_matrix[train_index[index], :-1]
                y_train = self.trsed_vector_matrix[train_index[index], -1]

                x_test = self.trsed_vector_matrix[test_index[index], :-1]
                y_test = self.trsed_vector_matrix[test_index[index], -1]

                discer = 0
                temp_y_test = np.where(y_test > 0, 1, 0)
                if np.sum(temp_y_test) != 0 and np.sum(temp_y_test) != len(temp_y_test) and len(temp_y_test) > 2:
                    # print(temp_y_test)
                    discer = compute_discer(x_test, temp_y_test)
                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(x_test)
                # y_pred_prob = estimator.predict_proba(x_test)
                # print(y_pred_prob)
                # exit(0)
                acc = estimator.score(x_test, y_test)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                # temp = set(y_test) - set(y_pred)
                # if temp:
                #     print(y_test)
                #     print(y_pred)
                #     exit(0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                precision = precision_score(y_test, y_pred, zero_division=0)
                # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
                # temp_auc = auc(fpr, tpr)

                cv_acc_list.append(acc)
                cv_f1_list.append(f1)
                cv_recall_list.append(recall)
                cv_precision_list.append(precision)
                cv_discer_list.append(discer)

            cv10_acc_list.append(np.mean(cv_acc_list))
            cv10_f1_list.append(np.mean(cv_f1_list))
            cv10_recall_list.append(np.mean(cv_recall_list))
            cv10_precision_list.append(np.mean(cv_precision_list))
            cv10_discer_list.append(np.mean(cv_discer_list))
        return np.mean(cv10_acc_list) * 100, np.std(cv10_acc_list) * 100, \
            np.mean(cv10_f1_list) * 100, np.std(cv10_f1_list) * 100, \
            np.mean(cv10_recall_list) * 100, np.std(cv10_recall_list) * 100, \
            np.mean(cv10_precision_list) * 100, np.std(cv10_precision_list) * 100, \
            float(np.mean(cv10_discer_list)), float(np.std(cv10_discer_list))


if __name__ == '__main__':
    start = time.process_time()
    smi = SMI(file_path='../MILframe/Text(sparse)/normalized/rec_motorcycles+.mat', dr=None)
    print('ins_len:', len(smi.trsed_vector_matrix[0]))  # 因为带标签，所以长度会加1

    print('acc: $%.1f_{\\pm%.1f}$' % (smi.acc, smi.acc_std))
    print('f1: $%.1f_{\\pm%.1f}$' % (smi.f1, smi.f1_std))
    print('recall: $%.1f_{\\pm%.1f}$' % (smi.recall, smi.recall_std))
    print('precision: $%.1f_{\\pm%.1f}$' % (smi.precision, smi.precision_std))
    print('Discernibility: $%.2f_{\\pm%.2f}$'
          % (smi.discer, smi.discer_std))

    end = time.process_time()
    print('本次运行耗时 %s 秒' % (end - start))

    # start = time.process_time()
    #
    # news_dic = '../MILframe/Text(sparse)/unnormalized/'
    # dataset_name_list = os.listdir(news_dic)
    # for name in dataset_name_list:
    #     print(name.split('.')[0], end=': ')
    #     acc_list, acc_std_list, f1_list, f1_std_list = [], [], [], []
    #     for i in range(2):
    #         smi = SMI(file_path=news_dic + name, dr=None)
    #         acc_list.append(smi.acc)
    #         acc_std_list.append(smi.acc_std)
    #         f1_list.append(smi.f1)
    #         f1_std_list.append(smi.f1_std)
    #     max_idx = int(np.argmax(acc_list))
    #     print('Acc: $%.1f_{\\pm%.1f}$, F1: $%.1f_{\\pm%.1f}$' % (
    #     acc_list[max_idx], acc_std_list[max_idx], f1_list[max_idx], f1_std_list[max_idx]))
    #
    # end = time.process_time()
    # print('Time Cost: ', (end - start))
