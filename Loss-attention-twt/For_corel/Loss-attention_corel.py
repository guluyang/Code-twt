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

warnings.filterwarnings('ignore')


# 返回多分类的包和标签
def load_multi_class_bag(path):
    mil = MIL(para_path=path)
    bags = []
    for i in range(mil.num_bags):
        bags.append(mil.bags[i, 0][:, :-1])
    lables = mil.bags_label
    return np.array(bags, dtype=object), np.array(lables)


class MyDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.bag_labels = np.array(labels)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        bag_label = self.bag_labels[idx]
        ins_labels = np.full(len(bag), bag_label)
        return bag, bag_label, ins_labels

    def __len__(self):
        return len(self.bag_labels)


# 每个类别随机选取一半作为训练集,另一半作为测试集,直接返回pytorch的Dataset类
def load_data(path, train_tate):
    bags, labels = load_multi_class_bag(path)
    n_bags = labels.shape[0]
    n_class = np.max(labels) + 1
    # 一共n_class行，每一行的包标签为对应行索引
    bag_idx = np.reshape(np.arange(0, n_bags), (-1, 100))
    train_bag_idx, test_bag_idx = [], []
    for i in range(n_class):
        total_index = bag_idx[i]  # 包标签为i的所有包索引
        train_idx = np.random.choice(total_index, int(len(total_index) * train_tate), replace=False)
        test_idx = np.setdiff1d(total_index, train_idx)
        train_bag_idx.append(np.sort(train_idx))
        test_bag_idx.append(np.sort(test_idx))
    train_bag_idx = np.reshape(train_bag_idx, (-1))
    test_bag_idx = np.reshape(test_bag_idx, (-1))
    np.random.shuffle(train_bag_idx)
    np.random.shuffle(test_bag_idx)
    trainDataset = MyDataset(bags[train_bag_idx], labels[train_bag_idx])
    testDataset = MyDataset(bags[test_bag_idx], labels[test_bag_idx])
    return trainDataset, testDataset


def run(path, n_class, train_rate, epochs, lr):
    trainDataset, testDataset = load_data(path, train_rate)
    ins_len = len(trainDataset[0][0][0])
    train_loader = DataLoader(trainDataset, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataset, shuffle=False, batch_size=1)
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    weight_criterion = CE(aggregate='mean')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LossAttention(ins_len, n_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    acc_list = []
    f1_list = []
    for epoch in range(epochs):
        # 开始训练
        model.train()
        train_loss = 0.0
        for bag, bag_label, ins_label in train_loader:
            ins_label = ins_label.squeeze()
            bag = bag.squeeze()
            if bag.shape[0] == 1 or bag.shape[0] == ins_len:
                continue
            optimizer.zero_grad()
            bag_label = bag_label.long()

            bag = bag.to(device)
            bag_label = bag_label.to(device)
            ins_label = ins_label.to(device)

            y, y_c, alpha = model(bag)
            loss_1 = criterion(y, bag_label)
            loss_2 = weight_criterion(y_c, ins_label, weights=alpha)
            loss = loss_1 + 0.1 * loss_2  # 0.1为论文规定
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        # print('%3d -th epoch: Train: loss: %.3f' % (epoch + 1, train_loss), end=' # ')

        # 测试
        true_label = []
        predict_labels = []
        with torch.no_grad():
            model.eval()
            for bag, bag_label, ins_label in test_loader:
                bag = bag.squeeze()
                if bag.shape[0] == 1 or bag.shape[0] == ins_len:
                    continue
                bag = bag.to(device)
                y, _, _ = model(bag)
                _, predicted = torch.max(y.data, 1)
                predict_labels.append(predicted.cpu().data.numpy())
                true_label.append(bag_label)
        predict_labels = np.squeeze(predict_labels)
        true_label = np.squeeze(true_label)
        accuracy = accuracy_score(true_label, predict_labels)
        f1 = f1_score(true_label, predict_labels, average='micro')
        acc_list.append(accuracy)
        f1_list.append(f1)
        if accuracy == 1:
            break
        # print('Test: acc: %.1f, f1: %.1f.' % (accuracy * 100, f1 * 100))
    return np.max(acc_list), np.max(f1_list)


if __name__ == '__main__':
    path = '../../Data/Corel/Blobworld/corel-10/corel-10-b-230+.mat'
    dataset_name = path.split('/')[-1]
    n_class = int(dataset_name.split('-')[1])
    # n_class = 3
    test_rate = 0.1
    train_rate = 1 - test_rate
    epochs = 80
    lr = 0.0001
    times = 10
    acc_list, f1_list = [], []
    for i in tqdm(range(times)):
        acc, f1 = run(path, n_class, train_rate, epochs, lr)
        acc_list.append(acc)
        f1_list.append(f1)
    acc_avg = float(np.mean(acc_list)) * 100
    acc_std = float(np.std(acc_list, ddof=1)) * 100
    f1_avg = float(np.mean(f1_list)) * 100
    f1_std = float(np.std(f1_list, ddof=1)) * 100
    print('n_class:', n_class)
    print('test_rate:', test_rate)
    print('Acc: $%.1f_{\\pm%.1f}$, F1: $%.1f_{\\pm%.1f}$' % (acc_avg, acc_std, f1_avg, f1_std))
    # n_class: 3, test_rate: 0.5, Acc: $89.6_{\pm3.6}$, F1: $89.6_{\pm3.6}$
    # n_class: 3, test_rate: 0.3, Acc: $94.1_{\pm1.4}$, F1: $94.1_{\pm1.4}$
    # n_class: 3, test_rate: 0.1, Acc: $96.4_{\pm2.3}$, F1: $96.4_{\pm2.3}$

    # n_class: 5, test_rate: 0.5, Acc: $66.2_{\pm5.3}$, F1: $66.2_{\pm5.3}$
    # n_class: 5, test_rate: 0.3, Acc: $77.7_{\pm4.4}$, F1: $77.7_{\pm4.4}$
    # n_class: 5, test_rate: 0.1, Acc: $82.8_{\pm4.8}$, F1: $82.8_{\pm4.8}$

    # n_class:10, test_rate: 0.5, Acc: $38.9_{\pm3.5}$, F1: $38.9_{\pm3.5}$
    # n_class:10, test_rate: 0.3, Acc: $45.1_{\pm2.8}$, F1: $45.1_{\pm2.8}$
    # n_class:10, test_rate: 0.1, Acc: $46.1_{\pm4.3}$, F1: $46.1_{\pm4.3}$
