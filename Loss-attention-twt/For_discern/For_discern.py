import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import torch
from weight_loss_for_dis import CrossEntropyLoss as CE
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from model_for_dis import LossAttention
from MILFrame.MIL import MIL
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl

warnings.filterwarnings('ignore')


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
    return eucl([positive_mean], [negative_mean])[0][0] / fenmu  # if fenmu > 1e-3 else 1e-3


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


class MyDateset(Dataset):
    def __init__(self, path):
        self.data = loadmat(path)['data']

    def __getitem__(self, idx):
        return self.data[idx][0], 1 if self.data[idx][1].tolist()[0][0] == 1 else 0

    def __len__(self):
        return len(self.data)


def run(trainDataset, testDataset, ins_len, n_class, epochs, lr, i_th_run, i_th_cv):
    train_loader = DataLoader(trainDataset, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataset, shuffle=False, batch_size=1)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    weight_criterion = CE(aggregate='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LossAttention(ins_len, n_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    acc_list = []
    f1_list = []
    dis_list = []
    for epoch in range(epochs):
        # 开始训练
        model.train()
        train_loss = 0.0
        for data, bag_label in train_loader:
            bag = data[0][:, :-1]
            if bag.shape[0] == 1:
                continue
            ins_label = data[0][:, -1]
            optimizer.zero_grad()
            bag = bag.to(device)
            y, y_c, alpha, _ = model(bag)
            bag_label = bag_label.to(device)
            loss_1 = criterion(y, bag_label)
            ins_label = ins_label.to(device)
            loss_2 = weight_criterion(y_c, ins_label, weights=alpha)
            loss = loss_1 + 0.1 * loss_2  # 0.1为论文规定
            train_loss += loss.item()
            # backward pass
            loss.backward()
            # step
            optimizer.step()
        train_loss /= len(train_loader)
        # print('%2d -th CV, %2d -th Run, %3d -th epoch: Train: loss: %.3f' %
        #       (i_th_cv + 1, i_th_run + 1, epoch + 1, train_loss), end=' # ')

        # 测试
        true_label = []
        predict_labels = []
        out_list = []
        with torch.no_grad():
            model.eval()
            for data, bag_label in test_loader:
                bag = data[0][:, :-1]
                bag = bag.to(device)
                y, _, _, out = model(bag)
                out_list.append(np.squeeze(out.cpu().numpy()))
                _, predicted = torch.max(y.data, 1)
                predict_labels.append(predicted.cpu().data.numpy())
                true_label.append(bag_label)
        predict_labels = np.squeeze(predict_labels)
        true_label = np.squeeze(true_label)
        accuracy = accuracy_score(true_label, predict_labels)
        f1 = f1_score(true_label, predict_labels)
        acc_list.append(accuracy)
        f1_list.append(f1)
        out_list = np.array(out_list)
        discer = 0
        if np.sum(true_label) != 0 and np.sum(true_label) != len(true_label) and len(true_label) > 2:
            discer = compute_discer(out_list, true_label)
        dis_list.append(discer)
        if accuracy == 1:
            break
        # print('Test: acc: %.1f, f1: %.1f, Dsicer: %.3f' % (accuracy * 100, f1 * 100, discer))
    return np.max(acc_list), np.max(f1_list), np.max(dis_list)


def one_cv(path, k, epochs, lr, i_th_cv):
    """
    :param k: k-cv
    :return: mean acc and f1 of one time k-cv
    """
    AllDataset = MyDateset(path)
    num_bags = len(AllDataset)
    mil = MIL(path)
    ins_len = len(mil.ins[0])
    n_class = mil.num_classes
    train_idx_list, test_idx_list = get_index(num_bags, k)
    acc_list, f1_list, dis_list = [], [], []
    for i in tqdm(range(k)):
        train_idx = train_idx_list[i]
        test_idx = test_idx_list[i]
        trainDataset = Subset(AllDataset, train_idx)
        testDataset = Subset(AllDataset, test_idx)
        acc, f1, dis = run(trainDataset, testDataset, ins_len, n_class, epochs, lr, i, i_th_cv)
        acc_list.append(acc)
        f1_list.append(f1)
        dis_list.append(dis)
    return np.mean(acc_list), np.mean(f1_list), np.mean(dis_list)


def n_cv(path, times, k, epochs, lr):
    acc_list, f1_list, dis_list = [], [], []
    for i in range(times):
        acc, f1, dis = one_cv(path, k, epochs, lr, i)
        # print('%.1f' % (acc * 100), end=' ')
        # print('%.1f' % (f1 * 100))
        # print()
        acc_list.append(acc)
        f1_list.append(f1)
        dis_list.append(dis)
    return np.mean(acc_list), np.std(acc_list, ddof=1), np.mean(f1_list), \
           np.std(f1_list, ddof=1), float(np.mean(dis_list)), float(np.std(dis_list, ddof=1))


if __name__ == '__main__':
    path = '../../Data/Text(sparse)/normalized/alt_atheism+.mat'
    times = 5  # 5次CV
    k = 10  # 10CV
    epochs = 100
    lr = 0.0001  # 论文规定为0.0001
    acc, acc_std, f1, f1_std, dis, dis_std = n_cv(path, times, k, epochs, lr)
    print('Dataset:', path.split('/')[-1].split('.')[0])
    print('Acc: $%.1f_{\\pm%.1f}$' % (acc * 100, float(acc_std * 100)), end=', ')
    print('F1: $%.1f_{\\pm%.1f}$' % (f1 * 100, float(f1_std * 100)))
    print('Discernibility: m: $%.2f_{\\pm%.2f}$'
          % (dis, dis_std))
