#!python3
# coding=UTF-8
'''k匿名的mondrian算法实现'''

'''处理标签型的思路：将标签型一一映射成为整数，获得数值型；最后发布的时候，文件中存放整数与标签的对应关系'''

from k_anonymity.load import loaddata
from k_anonymity.MyPriorityQueue import MyPriorityQueue
import random           # 产生随机数
import os               # path split
import numpy as np      # 数据处理
import pandas as pd     # 数据处理
import math             # 做floor、ceiling
import copy             # 用于深拷贝
import time             # 用于测量运行时间


def bigger(a,b):
    return a>b


class mondrian:
    '''
    k匿名的Mondrian算法实现\\
    初始化参数为数据集路径path(str)、匿名参数k(int)、数据集列名attributes(list)、需处理的QI(list)、待发布的隐私属性S(list)
    '''

    def __init__(self, path, k, attributes, QI, S):
        '''初始化：数据文件路径、k、attributes、QI、S'''
        self.data = loaddata(
            path, attributes)                   # 将有效数据从路径为path的.data文件中，按照列名attributes加载出来，成dataframe型
        self.data = self.data[QI+S]
        self.path = path
        self.k = k
        self.QI = QI
        self.S = S
        self.attributes = attributes
        # 初始化最大优先队列为只加入总data，key为
        self.Queue = MyPriorityQueue(bigger)
        self.Queue.enqueue(self.data.count()[S[0]], self.data)


    def search(self):
        '''对读入的数据搜解，即将最大优先队列的最大key变得不大于k；优先队列中的每个都相当于一个QI cluster'''
        start = time.time()                                         # 开始时间
        print('=>begin searching: k={0}'.format(self.k))            # 这里中间切分的过程太长，这里就不展示了
        print(self.data)
        while True:
            # 取数目最大的QI cluster
            data = self.Queue.dequeue()
            data_new = data[1]
            a = random.randint(0,len(self.QI)-1)
            qi = self.QI[a]
            # 按随机qi切半（不均匀，是按照中位数切的）
            data_new = pd.qcut(data_new[qi], q=2, labels=False, duplicates='drop')
            #print('before cut:\n{0}'.format(data_new))
            data1 = data[1][data_new == 0]
            num1 = data1.count()[self.S[0]]
            data2 = data[1][data_new == 1]
            num2 = data2.count()[self.S[0]]
            #print('after cut:')
            if(num1>=self.k and num2>=self.k):
                # 都满足，则加入优先队列中，继续循环
                self.Queue.enqueue(num1, data1)
                self.Queue.enqueue(num2, data2)
                #print('{0},{1},'.format(num1, num2))
                #print(data1)
                #print(data2)
                continue
            elif data[0] >= 2*self.k:
                # 某一不满足，但是数目能满足，则使用第二种中位数划分
                data_new = data[1].sort_values(by=self.QI)
                cut_num = math.floor(data[0]/2)
                data1 = data_new[0:cut_num]
                #print('data1:\n{0}'.format(data1))
                data2 = data_new[cut_num:data[0]]
                if(cut_num>=self.k and data[0]-cut_num>=self.k):
                    self.Queue.enqueue(cut_num, data1)
                    self.Queue.enqueue(data[0]-cut_num, data2)
                #print('{0},{1},{2},{3}'.format(data[1].count()[self.S[0]], data[0], cut_num, data[0]-cut_num))
            else:
                # 最大的不可再切，则完成
                self.Queue.enqueue(data[0], data[1])
                break

        # 打印测量的时间
        end = time.time()   # 结束时间
        print('=>end searching, total search time is:{0}(s)\n'.format(end-start))

        # 都切完了，做数据发布
        self.pub_data()


    def pub_data(self):
        '''将数据按照Queue里的QI cluster做泛化，并发布成.data文件，同时计算并打印LossMetric，返回生成的文件名'''
        start = time.time()                     # 开始时间
        LossMetric = 0
        max_QI = self.data[self.QI].max()       # 总data中的max
        min_QI = self.data[self.QI].min()       # 总data中的min
        range_QI = max_QI - min_QI              # 总data中的range
        print('=>begin publishing:')

        while self.Queue.is_empty() == False:
            # 对Queue中的每个QI cluster
            data = self.Queue.dequeue()
            max_qi = data[1][self.QI].max()     # 本QI cluster中的max
            min_qi = data[1][self.QI].min()     # 本QI cluster中的mmin
            range_qi = max_qi - min_qi          # 本QI cluster中的range
            for qi in self.QI:
                # 对每个QI属性
                LossMetric = LossMetric + data[0] * range_qi[qi] / range_QI[qi]     # 算LM
                self.data[qi].loc[data[1].index] = '[' + str(min_qi[qi]) + ',' + str(max_qi[qi]) + ']'   # 做泛化
        LossMetric = LossMetric / self.data.count()[self.S[0]]

        print('===>after generalization:')
        print(self.data)
        # 发布
        # 对数据文件路径做分割，得到去除文件后缀的firstname
        namesplit = os.path.splitext(self.path)
        firstname = namesplit[len(namesplit) - 2]
        out_path = firstname + '_mondrian.data'
        self.data.to_csv(out_path, index=False, header=False)
        print('===>publish file path:{0}'.format(out_path))

        end = time.time()   # 结束时间
        print('=>end publishing, total lossmetric is:{0}, total publishing time is:{1}'.format(LossMetric, end-start))
        return out_path