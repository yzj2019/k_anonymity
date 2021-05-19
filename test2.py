#!python3
# coding=UTF-8
# 打算最后再做成一个class发布出去
from sys import argv
from k_anonymity.load import loaddata
import os
import k_anonymity
from k_anonymity.MyPriorityQueue import MyPriorityQueue
import numpy as np
import pandas as pd
import random

script, path, k = argv[0], argv[1], argv[2]
# 注：path可以是相对路径如./data/adult.data，但是要确保树形结构的文件与data文件在同一文件夹下
# 后面也可以考虑将attributes与QI都输入


def multiply(x,y):
    return x*y

def bigger(a,b):
    return a>b


if __name__ == "__main__":
    # 测试读数据
    attributes = ['age', 'work_class', 'final_weight', 'education',
                  'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                  'native_datary', 'class']
    data = loaddata(path, attributes)

    # 测试路径分割
    #namesplit = os.path.splitext(path)
    #firstname = namesplit[len(namesplit) - 2]


    QI = ['age', 'education_num']
    S = ['occupation']
    print('test begin:')

    data = data[QI+S]

    # 取range
    print('测试取range：')
    print(data[QI].max() - data[QI].min())

    # dataframe分桶
    print('测试dataframe分桶：')
    print(data)
    mid = data.median()
    print(mid['age'])
    data_new = pd.qcut(data[QI[0]], q=2, labels=False)
    print(data_new == 1)
    a = random.randint(0,len(QI)-1)
    qi = QI[a]
    print(qi)
    data1 = data[data_new == 0]
    data2 = data[data_new == 1]
    print('data1 is:\n{0}'.format(data1))
    print('data2 is:\n{0}'.format(data2))
    data_new = pd.qcut(data2[QI[1]], q=2, labels=False)
    # print(data1.count())


    # dataframe多次分桶
    print('测试dataframe多次分桶后，index的正确性：')
    data_new = pd.qcut(data2[QI[1]], q=2, labels=False)
    data2_1 = data2[data_new == 0]
    data2_2 = data2[data_new == 1]
    print('data2_1 is:\n{0}'.format(data2_1))
    print('data2_2 is:\n{0}'.format(data2_2))
    data[QI[0]].loc[data2_1.index] = 'aaaaa'        # loc是按照索引来匹配，iloc是按照行号来匹配
    print(data)


    # dataframe排序

    
    # 测试优先队列
    print('测试实现的优先队列：')
    a = MyPriorityQueue(bigger)
    a.test_queue()

    print('test end.\n\n')


    # 测试Mondrian
    print('mondrian begin:')
    QI1 = ['age', 'gender']
    m = k_anonymity.mondrian(path, int(k), attributes, QI1, S)
    m.search()
    print('mondrian end.\n\n')



