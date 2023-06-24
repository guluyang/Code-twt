from torch.utils.data import Dataset, DataLoader, Subset
from utils import load_data, get_index
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import time
from utils import compute_discer
from tqdm import tqdm


# 返回包与其标签的DataSet类
class MyDataSet(Dataset):
    def __init__(self, path='../MILFrame/data/benchmark/musk1+.mat'):
        self.bags, self.labels = load_data(path=path)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# 定义注意力层聚合包内信息的层, 输入为[batchsize, ins_len]
class AttentionInBag(nn.Module):
    def __init__(self, agg_in_len, agg_out_len):
        super(AttentionInBag, self).__init__()
        self.ascend_dim = nn.Linear(in_features=agg_in_len, out_features=agg_out_len)
        self.compute_e = nn.Linear(in_features=agg_out_len, out_features=1)

    def forward(self, bag):
        bag = bag.float()
        bag = nn.LeakyReLU(0.2)(self.ascend_dim(bag))
        e_list = self.compute_e(bag)
        e_list = torch.reshape(e_list, (1, bag.shape[0]))
        alpha_list = F.softmax(e_list, dim=1)
        x = torch.mm(alpha_list, bag)
        return x


# 定义自注意力层聚合实例标签和包标签的信息, 输入为[batchsize, ins_len]
class SelfAttention(nn.Module):
    def __init__(self, self_att_in_len, self_att_out_len):
        super(SelfAttention, self).__init__()
        self.ascend_dim = nn.Linear(in_features=self_att_in_len, out_features=self_att_out_len)
        self.compute_e = nn.Linear(in_features=2 * self_att_out_len, out_features=1)

    def forward(self, bag):
        bag = nn.LeakyReLU(0.2)(self.ascend_dim(bag))
        center_ins = torch.reshape(bag[0], (1, bag.shape[1]))
        center_ins_matrix = center_ins.repeat(bag.shape[0], 1)  # 将center_ins在第0个维度上复制三次，第1个维度上只复制一次
        self_neighbors = torch.cat((center_ins_matrix, bag), dim=1)
        self_neighbors = self_neighbors.float()
        # print('self_neighbors:', self_neighbors)
        e_list = self.compute_e(self_neighbors)
        e_list = torch.reshape(e_list, (1, e_list.shape[0]))  # e_list reshape为1*3
        alpha_list = F.softmax(e_list, dim=1)
        aggrgated_ins = torch.matmul(alpha_list, bag)  # 聚合后的单向量
        return aggrgated_ins


# 定义整个网络
class Net(nn.Module):
    def __init__(self, agg_in_len, agg_out_len, self_in_len, self_out_len, n_class):
        super(Net, self).__init__()
        self.attention = AttentionInBag(agg_in_len=agg_in_len, agg_out_len=agg_out_len)
        self.bag_info = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=agg_out_len, out_features=int(agg_out_len / 10)),
            nn.LeakyReLU(negative_slope=0.2),
            # 处理到self-attention层的输入长度
            nn.Linear(in_features=int(agg_out_len / 10), out_features=self_in_len)
        )
        self.ins_info = nn.Sequential(
            nn.Linear(in_features=agg_in_len, out_features=int(agg_in_len / 2)),
            nn.LeakyReLU(negative_slope=0.2),
            # 处理到self-attention层的输入长度
            nn.Linear(in_features=int(agg_in_len / 2), out_features=self_in_len)
        )
        self.agg_bag_ins = SelfAttention(self_att_in_len=self_in_len, self_att_out_len=self_out_len)
        self.agg_bag_ins_linear = nn.Linear(in_features=self_out_len, out_features=n_class)

    def forward(self, bag):
        bag = bag.float()
        h = self.attention(bag)
        bag_info = self.bag_info(h)
        ins_info = self.ins_info(bag)
        total_info = torch.cat((bag_info, ins_info), dim=0)
        x = nn.LeakyReLU(0.2)(self.agg_bag_ins(total_info))
        y = self.agg_bag_ins_linear(x)
        return y, h, x


def run_test(cv_idx, run_test_idx, trainDataSet, testDataSet,
             epochs, lr, agg_out_len, self_in_len, self_out_len, n_class):
    train_loader = DataLoader(trainDataSet, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataSet, shuffle=False, batch_size=1)
    ins_len = len(trainDataSet[0][0][0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(agg_in_len=ins_len, agg_out_len=agg_out_len,
                self_in_len=self_in_len, self_out_len=self_out_len, n_class=n_class).to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # 与tensorflow不同，此处交叉熵损失先会将输入softmax
    # 并且真实标签只能为单数字，不能为one-hot
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练一轮，测试一次
    acc_list = []
    f1_list = []
    h_discer_list, b_discer_list = [], []
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            target = target.long()
            target = target.to(device)
            inputs = inputs.squeeze(0)
            inputs = inputs.to(device)
            outputs, h, b = model(inputs)

            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print('loss of %3d -th opoch in %2d -th Run Test of %d -th CV: %.3f :'
        #       % (epoch + 1, run_test_idx + 1, cv_idx + 1, running_loss / len(trainDataSet)), end=' # ')
        # testing phase
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []
            h_list = []
            b_list = []
            for data in test_loader:
                inputs, labels = data
                y_true.append(labels.cpu().item())
                inputs = inputs.squeeze(0)
                inputs = inputs.to(device)
                outputs, h, b = model(inputs)
                h_list.append(np.squeeze(h.cpu().numpy()))
                b_list.append(np.squeeze(b.cpu().numpy()))
                _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引
                y_pred.append(pred.cpu().numpy())
                total += labels.size(0)
                correct += (pred == labels.cpu().item()).sum().detach().cpu().item()
                acc = correct / total
        h_list = np.array(h_list)
        b_list = np.array(b_list)
        h_discer, b_discer = 0, 0
        if np.sum(y_true) != 0 and np.sum(y_true) != len(y_true) and len(y_true) > 2:
            h_discer = compute_discer(h_list, y_true)
            b_discer = compute_discer(b_list, y_true)
        h_discer_list.append(h_discer)
        b_discer_list.append(b_discer)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # print('Test: acc: %.2f | f1: %.2f, Dsicer: h: %.3f, b: %.3f' % (acc, f1, h_discer, b_discer))
        acc_list.append(acc)
        f1_list.append(f1)
        if acc == 1:
            break
    return np.max(acc_list), np.max(f1_list), np.max(h_discer_list), np.max(b_discer_list)


def one_cv(path, cv_idx, para_k, epochs, lr, agg_out_len):
    AllDataSet = MyDataSet(path=path)
    n_class = np.max(AllDataSet[:][1]) + 1
    train_idx_list, test_idx_list = get_index(len(AllDataSet), para_k=para_k)
    acc_list, f1_list, h_list, b_list = [], [], [], []
    for i in tqdm(range(para_k)):
        trainDataSet = Subset(AllDataSet, train_idx_list[i])
        testDataSet = Subset(AllDataSet, test_idx_list[i])
        acc, f1, h, b = run_test(cv_idx=cv_idx, run_test_idx=i, trainDataSet=trainDataSet, testDataSet=testDataSet,
                                 epochs=epochs, lr=lr,
                                 # 自注意力层的输入维度为5倍标签数，输出维度为2倍标签数
                                 agg_out_len=agg_out_len, self_in_len=5 * n_class, self_out_len=2 * n_class,
                                 n_class=n_class)
        acc_list.append(acc)
        f1_list.append(f1)
        h_list.append(h)
        b_list.append(b)
    print('-' * 50 + 'One CV Done' + '|' + 'acc: ' + str(np.mean(acc_list)) + ' f1: ' + str(np.mean(f1_list)))
    return float(np.mean(acc_list)), float(np.mean(f1_list)), float(np.mean(h_list)), float(np.mean(b_list))


def n_cv(path, num_cv, para_k, epochs, lr, agg_out_len):
    acc_list, f1_list, h_list, b_list = [], [], [], []
    for i in range(num_cv):
        acc, f1, h, b = one_cv(path=path, cv_idx=i, para_k=para_k, epochs=epochs, lr=lr,
                               agg_out_len=agg_out_len)
        acc_list.append(acc)
        f1_list.append(f1)
        h_list.append(h)
        b_list.append(b)
    print('*' * 10 + path.split('/')[-1].split('.')[0] + '*' * 10)
    print('lr: ', lr)
    print('epochs: ', epochs)
    print('agg_out_len: ', agg_out_len)
    return float(np.mean(acc_list)), float(np.std(acc_list)), float(np.mean(f1_list)), float(np.std(f1_list)), \
           float(np.mean(h_list)), float(np.std(h_list)), float(np.mean(b_list)), float(np.std(b_list))


if __name__ == '__main__':
    # input_ = torch.randn(3, 166)
    # net = Net(ins_len=166, n_class=2)
    # out = net(input_)
    # print(out)
    # print(out.shape)
    start = time.process_time()

    acc, acc_std, f1, f1_std, h, h_std, b, b_std = n_cv(path='../../Data/Text(sparse)/normalized/rec_motorcycles+.mat',
                                                        num_cv=5, para_k=10,
                                                        epochs=100, lr=0.0005, agg_out_len=512)
    print('Mean Result of 5 CV : acc: $%.2f_{\\pm%.2f}$ | f1: $%.2f_{\\pm%.2f}$'
          % (acc * 100, acc_std * 100, f1 * 100, f1_std * 100))
    print('Discernibility: h: acc: $%.2f_{\\pm%.2f}$ | b: $%.2f_{\\pm%.2f}$'
          % (h, h_std, b, b_std))
    print('Time Cost: ', (time.process_time() - start))
    # data = MyDataSet(path='../../Data/Benchmark/musk1+.mat')
    # print(data[0][1])
    # musk1: Discernibility: h: acc: $0.72_{\pm0.05}$ | b: $1.08_{\pm0.03}$
    # musk2: Discernibility: h: acc: $1.41_{\pm1.08}$ | b: $3.57_{\pm1.85}$
    # ele: Discernibility:   h: acc: $0.87_{\pm0.02}$ | b: $1.16_{\pm0.06}$
    # fox: Discernibility: h: acc: $0.28_{\pm0.01}$ | b: $0.39_{\pm0.01}$
    # tiger: Discernibility: h: acc: $0.73_{\pm0.02}$ | b: $0.91_{\pm0.03}$
    # News.aa: Discernibility: h: acc: $120.38_{\pm12.83}$ | b: $171.87_{\pm15.98}$
    # News.co: Discernibility: h: acc: $0.70_{\pm0.15}$ | b: $0.84_{\pm0.25}$
    # News.csm: Discernibility: h: acc: $0.73_{\pm0.03}$ | b: $0.89_{\pm0.04}$
    # News.mf: Discernibility: h: acc: $0.67_{\pm0.08}$ | b: $0.66_{\pm0.09}$
    # News.rm: Discernibility: h: acc: $0.96_{\pm0.10}$ | b: $1.12_{\pm0.19}$
