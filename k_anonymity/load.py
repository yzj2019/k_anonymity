#!python3
# coding=UTF-8
'''加载数据，可扩展支持的文件类型'''
import pandas as pd
import numpy as np


def loaddata(path, attributes):
    '''从路径为path的.data文件中，按照列名attributes读入数据集成dataframe型，并去除空数据和'?'所在的行，返回所得dataframe'''
    # pandas读出dataframe型，分隔符为', '
    print('=>loading from {0}'.format(path))
    data_raw = pd.read_csv(path, sep=', ', engine='python', names=attributes)
    print('=>complete loading')

    # 清洗，去除'?'：将'?'替换为np.nan，用pandas.dataframe下处理缺失值的方法统一删除所在行
    print('=>start cleaning')
    print('===>before:')
    print(data_raw)
    data = data_raw.replace('?', np.nan)
    data = data.dropna()
    print('===>after:')
    print(data)  # 由清洗后的行数==30162可确定成功
    print('=>is cleaned\n')
    return data
