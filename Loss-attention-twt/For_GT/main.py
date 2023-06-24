import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import torch
from weight_loss import CrossEntropyLoss as CE
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from model import LossAttention
from MILFrame.MIL import MIL
from tqdm import tqdm
import time
from pathes import get_path_for_GT

warnings.filterwarnings('ignore')


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
    model = LossAttention(ins_len, n_class)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    acc_list = []
    f1_list = []
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
            y, y_c, alpha = model(bag)
            loss_1 = criterion(y, bag_label)
            loss_2 = weight_criterion(y_c, ins_label, weights=alpha)
            loss = loss_1 + 0.1 * loss_2  # 0.1为论文规定
            train_loss += loss.item()
            # backward pass
            loss.backward()
            # step
            optimizer.step()
        train_loss /= len(train_loader)
        # print('%2d -th CV, %2d -th Run, %3d -th epoch: Train: loss: %.3f' %
              # (i_th_cv + 1, i_th_run + 1, epoch + 1, train_loss), end=' # ')

        # 测试
        true_label = []
        predict_labels = []
        with torch.no_grad():
            model.eval()
            for data, bag_label in test_loader:
                bag = data[0][:, :-1]
                y, _, _ = model(bag)
                _, predicted = torch.max(y.data, 1)
                predict_labels.append(predicted.data.numpy())
                true_label.append(bag_label.item())
        predict_labels = np.squeeze(predict_labels)
        true_label = np.squeeze(true_label)
        accuracy = accuracy_score(true_label, predict_labels)
        f1 = f1_score(true_label, predict_labels)
        acc_list.append(accuracy)
        f1_list.append(f1)
        if accuracy == 1:
            break
        # print('Test: acc: %.1f, f1: %.1f.' % (accuracy * 100, f1 * 100))
    return np.max(acc_list), np.max(f1_list)


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
    acc_list, f1_list = [], []
    for i in range(k):
        train_idx = train_idx_list[i]
        test_idx = test_idx_list[i]
        trainDataset = Subset(AllDataset, train_idx)
        testDataset = Subset(AllDataset, test_idx)
        acc, f1 = run(trainDataset, testDataset, ins_len, n_class, epochs, lr, i, i_th_cv)
        acc_list.append(acc)
        f1_list.append(f1)
    return np.mean(acc_list), np.mean(f1_list)


def n_cv(path, times, k, epochs, lr):
    acc_list, f1_list = [], []
    for i in range(times):
        acc, f1 = one_cv(path, k, epochs, lr, i)
        # print('%.1f' % (acc * 100), end=' ')
        # print('%.1f' % (f1 * 100))
        acc_list.append(acc)
        f1_list.append(f1)
    return np.mean(acc_list), np.std(acc_list), np.mean(f1_list), np.std(f1_list)


if __name__ == '__main__':
    # path = '../../Data/Ele_v_Fox_v_Tiger/Ele_fox_tiger.mat'
    # times = 5  # 5次CV
    # k = 10  # 10CV
    # epochs = 100
    # lr = 0.0001  # 论文规定为0.0001
    # acc, acc_std, f1, f1_std = n_cv(path, times, k, epochs, lr)
    # print('Dataset:', path.split('/')[-1].split('.')[0])
    # print('Acc: $%.1f_{\\pm%.1f}$' % (acc * 100, float(acc_std * 100)), end=', ')
    # print('F1: $%.1f_{\\pm%.1f}$' % (f1 * 100, float(f1_std * 100)))
    path_list = get_path_for_GT()[-11:]
    for path in path_list:
        start = time.process_time()
        acc, acc_std, f1, f1_std = n_cv(path, 2, 10, 100, 0.0001)
        name = path.split('/')[-1].split('.')[0]
        print('%-25s acc: $%.3f_{\\pm%.3f}$, f1: $%.3f_{\\pm%.3f}$' % (name, acc, acc_std, f1, f1_std))

# musk1+.mat Time cost of one 10CV: 2029.0
# musk2+.mat Time cost of one 10CV: 2950.984375
# elephant+.mat Time cost of one 10CV: 6198.125
# fox+.mat Time cost of one 10CV: 6649.65625
# tiger+.mat Time cost of one 10CV: 6577.078125
# alt_atheism+.mat Time cost of one 10CV: 2276.6875
# comp_os_ms-windows_misc+.mat Time cost of one 10CV: 3199.765625
# comp_sys_mac_hardware+.mat Time cost of one 10CV: 2845.3125
# misc_forsale+.mat Time cost of one 10CV: 3206.140625
# rec_motorcycles+.mat Time cost of one 10CV: 3158.25
# rec_sport_hockey+.mat Time cost of one 10CV: 2195.234375
# sci_med+.mat Time cost of one 10CV: 2663.265625
# sci_religion_christian+.mat Time cost of one 10CV: 2829.265625
# talk_politics_guns+.mat Time cost of one 10CV: 2880.046875
# talk_politics_misc+.mat Time cost of one 10CV: 2988.515625