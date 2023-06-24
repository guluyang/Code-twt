from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat


class miFV:
    def __init__(self, train_bag, train_label, k):
        self.k = k
        self.gmm = GaussianMixture(n_components=k, covariance_type='diag')
        self.train_bag = train_bag
        self.ins = []
        for bag in self.train_bag:
            for ins in bag:
                self.ins.append(ins)
        self.ins = np.array(self.ins)
        self.train_label = train_label

        self.gmm.fit(self.ins)
        self.weights = self.gmm.weights_  # 各个分布的权重
        self.means = self.gmm.means_  # 各个分布的均值向量
        self.covars = self.gmm.covariances_  # 各个分布的协方差矩阵
        self.D = self.k * len(self.ins[0])
        self.svm = SVC()

    def train(self):
        temp_trs_vector_matrix = []
        for i in range(len(self.train_bag)):
            temp_matrix = []
            for k in range(self.k):
                vector = []
                temp1 = self.f_xi_wk(self.train_bag[i], k)
                temp2 = self.f_xi_uk(self.train_bag[i], k)
                temp3 = self.f_xi_sigmak(self.train_bag[i], k)
                vector = np.append(vector, temp1)
                vector = np.append(vector, temp2)
                vector = np.append(vector, temp3)  # 此时的vector为最开始的fv向量
                temp_matrix.append(vector)
            temp_vector = np.sum(temp_matrix, axis=0)  # 每个包转化成向量
            temp_vector = np.multiply(np.sqrt(np.abs(temp_vector)), np.sign(temp_vector))  # 带符号的开方
            temp_vector = temp_vector / np.sqrt(np.sum(np.power(temp_vector, 2)))  # 除以二范数
            # temp_vector = np.append(temp_vector, self.train_label[i])
            temp_trs_vector_matrix.append(temp_vector)
        temp_trs_vector_matrix = np.array(temp_trs_vector_matrix)
        self.svm.fit(temp_trs_vector_matrix, self.train_label)

    def f_xi_wk(self, bag, k):
        temp_sum = 0
        for j in range(len(bag)):
            x_ij = bag[j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) - self.weights[k]
        return 1 / np.sqrt(self.weights[k]) * temp_sum

    def gamma(self, k, x_gamma):
        return self.gmm.predict_proba(x_gamma.reshape(1, -1))[0, k]  # reshape(1, -1)在单向量外套一对中括号

    def f_xi_uk(self, bag, k):
        temp_sum = 0
        for j in range(len(bag)):
            x_ij = bag[j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) * ((x_ij - self.means[k]) / self.covars[k])
        return 1 / np.sqrt(self.weights[k]) * temp_sum

    def f_xi_sigmak(self, bag, k):
        temp_sum = 0
        for j in range(len(bag)):
            x_ij = bag[j]
            temp_sum += self.gamma(k=k, x_gamma=x_ij) * 1/np.sqrt(2) * (((x_ij - self.means[k])**2 / self.covars[k]**2) - 1)
        return 1 / np.sqrt(self.weights[k]) * temp_sum

    def pred(self, test_bags):
        temp_trs_vector_matrix = []
        for i in range(len(test_bags)):
            temp_matrix = []
            for k in range(self.k):
                vector = []
                temp1 = self.f_xi_wk(test_bags[i], k)
                temp2 = self.f_xi_uk(test_bags[i], k)
                temp3 = self.f_xi_sigmak(test_bags[i], k)
                vector = np.append(vector, temp1)
                vector = np.append(vector, temp2)
                vector = np.append(vector, temp3)  # 此时的vector为最开始的fv向量
                temp_matrix.append(vector)
            temp_vector = np.sum(temp_matrix, axis=0)  # 每个包转化成向量
            temp_vector = np.multiply(np.sqrt(np.abs(temp_vector)), np.sign(temp_vector))  # 带符号的开方
            temp_vector = temp_vector / np.sqrt(np.sum(np.power(temp_vector, 2)))  # 除以二范数
            # temp_vector = np.append(temp_vector, self.train_label[i])
            temp_trs_vector_matrix.append(temp_vector)
        temp_trs_vector_matrix = np.array(temp_trs_vector_matrix)
        return self.svm.predict(temp_trs_vector_matrix)


if __name__ == '__main__':
    def get_index(num_bags, para_k=10):
        np.random.RandomState(seed=66)
        temp_rand_idx = np.random.permutation(num_bags)

        temp_fold = int(np.ceil(num_bags / para_k))
        ret_tr_idx = {}
        ret_te_idx = {}
        for i in range(para_k):
            temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
            temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
            ret_tr_idx[i] = temp_tr_idx
            ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
        return ret_tr_idx, ret_te_idx
    train_idx_list, test_idx_list = get_index(92)
    train_idx = train_idx_list[0]
    test_idx = test_idx_list[0]

    data = loadmat('../MILframe/Benchmark/musk1+.mat')['data']
    bag = []
    label = []
    for i in range(len(data)):
        bag.append(data[i][0])
        label.append(data[i][1][0][0])
    bag = np.array(bag, dtype=object)
    label = np.array(label)


    train_bag = bag[train_idx]
    train_label = label[train_idx]
    test_bag = bag[test_idx]  # test_bag为三维数据，每个元素为一个包
    mifv = miFV(train_bag, train_label, 2)
    mifv.train()
    print(mifv.pred(test_bag))
    for bag in test_bag:
        print(mifv.pred([bag])[0], end=' ')