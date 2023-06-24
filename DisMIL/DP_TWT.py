import numpy as np
import math
import time

class DP:
    def __init__(self):
        self.dis_matrix = np.array([])
        self.num_bags = 0
        self.d_c, self.n_c = 0, 0  # n_c为聚类中心的比例
        self.local_density_list = []
        self.dis_to_master_list = []
        self.lambda_list = []
        self.center_list = []

    def train(self, matrix, r=0.2, u=4, kernel='gaussian'):
        self.dis_matrix = np.array(matrix)
        self.num_bags = len(self.dis_matrix)
        self.d_c = r * np.max(matrix)
        self.n_c = u
        self.local_density_list = self.get_local_density_list(kernel)
        # print(self.local_density_list)
        self.dis_to_master_list = self.get_dis_to_master_list()
        # print(self.dis_to_master_list)
        self.lambda_list = self.get_lambda_list()
        # print(self.lambda_list.tolist())
        self.center_list = self.get_center_list()
        #print(self.center_list)

    def get_center_list(self):
        #print("computing center list........")
        temp_center_list = []
        temp_lambda_list = self.lambda_list.tolist()
        for i in range(0, self.n_c):
            temp_center_list.append(temp_lambda_list.index(max(temp_lambda_list)))
            temp_lambda_list[temp_lambda_list.index(max(temp_lambda_list))] = 0
        return np.array(temp_center_list)

    def get_lambda_list(self):
        #print("computing lambda list........")
        lambda_list = np.multiply(self.local_density_list, self.dis_to_master_list)
        return lambda_list


    def get_dis_to_master_list(self):
        temp_dis_to_master_list = []
        #print("getting dis to master list........")
        for i in range(0, self.num_bags):
            density_list = []  # 记录所有局部密度比第i个包更大的包的序号
            i_density = self.local_density_list[i]
            for j in range(0, self.num_bags):
                if self.local_density_list[j] > i_density:
                    density_list.append(j)
            dis_list = []  # 记录所有局部密度大于第i个包的包与i包的距离
            for k in density_list:
                dis_list.append(self.dis_matrix[i, k])
            # print(dis_list)
            if dis_list:
                dis_list.sort()
                temp_dis_to_master_list.append(dis_list[0])  # 返回i包与其master的距离
            else:
                temp_dis_to_master_list.append(np.max(self.dis_matrix))  # 表示此包为局部密度最大的包
        return np.array(temp_dis_to_master_list)

    def get_local_density_list(self, kernel):
        if kernel == 'gaussian':
            # temp_local_density_list = []
            # for i in range(0, self.num_bags):
            #     temp_local_density_list.append(self.gaussian_kernel(i))
            temp_local_density_list = np.sum(1 / (np.exp(np.power(self.dis_matrix / self.d_c, 2))), axis=1)
            return temp_local_density_list
        elif kernel == 'cutoff':
            temp_local_density_list = []
            for i in range(0, self.num_bags):
                temp_local_density_list.append(self.cutoff_kernel(i))
            return temp_local_density_list

    # 截断核中的F(x)
    def F(self, x):
        if x < 0:
            return 1
        else:
            return 0

    # 截断核,计算包的局部密度
    def cutoff_kernel(self, i):
        p_i = 0
        for j in range(0, self.num_bags):
            p_i = p_i + self.F(self.dis_matrix[i, j] - self.d_c)
        return p_i


if __name__ == '__main__':
    # dp = DP()

    # # smdp = SMDP()
    # # smdp.train()
    # np.random.seed(10)
    # temp_dis = np.random.rand(5, 5)
    # temp_dis = np.triu(temp_dis)
    # temp_dis = np.transpose(temp_dis) + temp_dis
    # # print(temp_dis)
    # dp.train(temp_dis, kernel='cutoff')
    # print(dp.lambda_list)
    # print(dp.center_list)
    start = time.process_time()
    data = np.loadtxt('../distance/benchmark/musk1/musk1+_ave_hausdorff_euclidean.txt')

    dp = DP()
    dp.train(matrix=data, kernel='gaussian', r=0.2, u=4)

    print('centers:', dp.center_list)

    end = time.process_time()
    print('time cost:', end - start)
