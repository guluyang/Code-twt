import pandas as pd
import numpy as np
import re
import os


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# for name in os.listdir('results/RDMIL-C/'):
#     path = 'results/RDMIL-C/' + name
#     df = pd.read_csv(path, header=None)
#     # print(df)
#     # print(df[2].apply(lambda x: float(re.split(r'[$\\]', x)[1])))
#     print(name.split('.')[0], end=': ')
#     print(sorted(df[2].values, key=lambda x: float(re.split(r'[$\\]', x)[1]))[-1], end=' | ')
#     print(sorted(df[3].values, key=lambda x: float(re.split(r'[$\\]', x)[1]))[-1].split('| ')[-1])

for name in os.listdir('results/RDMIL-F/'):
    path = 'results/RDMIL-F/' + name
    df = pd.read_csv(path, header=None)
    # print(df)
    # print(df[2].apply(lambda x: float(re.split(r'[$\\]', x)[1])))
    print(name.split('.')[0], end=': ')
    print(sorted(df[1].values, key=lambda x: float(re.split(r'[$\\]', x)[1]))[-1], end=' | ')
    print(sorted(df[2].values, key=lambda x: float(re.split(r'[$\\]', x)[1]))[-1].split('| ')[-1])