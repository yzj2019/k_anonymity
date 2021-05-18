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

script, path, k, ms = argv[0], argv[1], argv[2], argv[3]
# 注：path可以是相对路径如./data/adult.data，但是要确保树形结构的文件与data文件在同一文件夹下
# 后面也可以考虑将attributes与QI都输入


def multiply(x,y):
    return x*y


if __name__ == "__main__":
    # 测试读数据
    attributes = ['age', 'work_class', 'final_weight', 'education',
                  'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                  'native_country', 'class']
    data = loaddata(path, attributes)

    # 测试路径分割
    #namesplit = os.path.splitext(path)
    #firstname = namesplit[len(namesplit) - 2]


    QI = ['age', 'gender', 'race', 'marital_status']
    S = ['occupation']
    print('test begin:')

    # 两list求差
    print('两list求差:')
    sub = list(set(attributes).difference(set(QI)))
    print(sub)

    # dataframe分组聚集
    print('dataframe分组聚集:')
    count = data.groupby(QI, as_index=False).count()
    count = count[(QI+sub)[0:len(QI)+1]]
    print(count)

    # dataframe列重命名
    print('dataframe列重命名:')
    rename = {sub[0]:'num'}
    count.rename(columns = rename, inplace=True)
    print(count)

    # dataframe按行应用函数
    print('dataframe按行应用函数:')
    for qi in QI:
        count[qi] = count.apply(lambda row: (row[qi]*row['num']), axis=1)
    print(count['num'].sum())

    # dataframe泛化重命名
    print('dataframe泛化重命名:')
    rename = {'Male':'*', 'Female':'*'}             # 泛化字典
    count['gender'] = count['gender'].map(rename)   # 泛化
    print(count)
    vals = np.array(count['age'].drop_duplicates()) # 数值型所有的取值
    print(vals)
    rename = {vals[0]:'17~20'}
    print(count['age'].map(rename))
    
    # 测试最大优先队列
    # a = MyPriorityQueue()
    # a.test_queue()

    print('test end.\n\n')



    # samarati

    print('samarati begin:')
    s = k_anonymity.samarati(path,int(k),int(ms),attributes,QI,S)

    # 测试samarati加载数据
    # print('测试samarati加载数据')
    # s.show_data()

    # 测试创建树形结构
    s.build_tree()
    # s.show_tree()

    # 测试k匿名判断，debug用
    # print('测试k匿名判断:')
    # s.is_kanonymity([1,0,1,2], False)
    # print(s.is_kanonymity([3,1,0,0], False))
    # print(s.is_kanonymity([4,1,1,0], False))

    # 打印泛化字典，debug用
    # print(s.tree['age'].h2GeneralizationDict)
    # print(s.tree['gender'].h2GeneralizationDict)
    # print(s.tree['race'].h2GeneralizationDict)
    # print(s.tree['marital_status'].h2GeneralizationDict)

    # 测试samarati搜索
    s.search()
    print('samarati end.\n\n')

