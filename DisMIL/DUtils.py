import numpy as np
from scipy.io import loadmat


def load_data(path):
    data = loadmat(path)['data']
    bags = []
    labels = []
    for i in range(len(data)):
        bags.append(data[i][0][:, :-1])
        labels.append(data[i][1][0, 0])
    labels = np.array(labels)
    ins = np.vstack(bags)
    # 记录每个包包含的实例，形式为: bag_idx [head_ins_idx, tail_ins_idx]
    bag_ins_idx = []
    counter = 0
    for bag in bags:
        temp = [counter, counter + bag.shape[0]]
        counter += bag.shape[0]
        bag_ins_idx.append(temp)
    bag_ins_idx = np.array(bag_ins_idx)
    return bags, labels, ins, bag_ins_idx


def get_index(num_bags=92, para_k=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
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


if __name__ == '__main__':
    load_data('../Data/Benchmark/musk1+.mat')