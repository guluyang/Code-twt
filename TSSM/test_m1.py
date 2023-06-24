from MILframe import MIL
import numpy as np
from DP_2 import DP

def dis_between_ins(ins1, ins2):
    # return np.sqrt((np.sum((ins1 - ins2) ** 2)))
    return np.sqrt(np.sum(np.power((ins1 - ins2), 2)))


mil = MIL.MIL('../MILframe/data/benchmark/musk1.mat')
bag_1 = mil.bags[0, 0][:, :-1]
matrix = np.zeros((len(bag_1), len(bag_1)))
for i in range(len(bag_1)):
    for j in range(len(bag_1)):
        matrix[i, j] = dis_between_ins(bag_1[i], bag_1[j])

dp = DP()
dp.train(matrix=matrix, r=0.2)
vec = np.matmul(dp.lambda_list, bag_1)
vec = vec / len(bag_1)
vec = vec.tolist()
vec.append(1)
print(vec)
# print(dp.get_outlier_idx())
# print(dp.get_incluster_idx())


