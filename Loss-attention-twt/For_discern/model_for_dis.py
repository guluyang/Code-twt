import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat


class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag == 1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            # print(out.shape)
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            alpha01 = features.size(0) * alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0), 1)

        return context, out_c, torch.squeeze(alpha)


class LossAttention(nn.Module):
    def __init__(self, ins_len, n_class):
        super(LossAttention, self).__init__()
        self.linear_1 = nn.Linear(ins_len, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.drop = nn.Dropout()
        self.linear = nn.Linear(64, n_class)
        self.att_layer = AttentionLayer(ins_len)

    def forward(self, bag, flag=1):
        bag = bag.float()
        bag_1 = self.drop(F.relu(self.linear_1(bag)))
        bag_2 = self.drop(F.relu(self.linear_2(bag_1)))
        bag_3 = self.drop(F.relu(self.linear_3(bag_2)))
        out, out_c, alpha = self.att_layer(bag_3, self.linear.weight, self.linear.bias, flag)
        out = out.mean(0, keepdim=True)

        y = self.linear(out)
        return y, out_c, alpha, out


if __name__ == '__main__':
    path = '../../Data/Benchmark/musk1+.mat'
    data = loadmat(path)['data']
    bag = data[0][0][:, :-1]
    print('n_ins:', len(bag))
    ins_len = len(bag[0])
    bag = torch.tensor(bag)
    n_class = 2
    lossatt = LossAttention(ins_len=ins_len, n_class=n_class)
    y, out_c, alpha, out = lossatt(bag)
    print(y)
    print(out_c)
    print(alpha)
    print(out)
