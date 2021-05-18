#!python3
# coding=UTF-8
'''k匿名的samarati算法实现'''

from k_anonymity.load import loaddata
from k_anonymity.MyTree import MyTree
import os               # path split
import numpy as np      # 数据处理
import pandas as pd     # 数据处理
import math             # 做floor、ceiling
import copy             # 用于深拷贝
import time             # 用于测量运行时间


class samarati:
    '''
    k匿名的samarati算法实现\\
    初始化参数为：数据集路径path(str)、匿名参数k(int)、最大suppression的个数ms(int)、数据集列名attributes(list)、需处理的QI(list)、待发布的隐私属性S(list)
    '''

    def __init__(self, path, k, ms, attributes, QI, S):
        '''初始化：数据文件路径、k、ms、attributes、QI、S'''
        self.data = loaddata(
            path, attributes)                   # 将有效数据从路径为path的.data文件中，按照列名attributes加载出来，成dataframe型
        self.data = self.data[QI+S]
        self.path = path
        self.k = k
        self.QI = QI
        self.S = S
        self.attributes = attributes
        self.ms = ms
        self.tree = {}                          # QI->对应的泛化树形结构
        # 按照QI各个具体取值分组聚集计行数，方便后面统计一个cluster中的记录数判断k匿名；其中as_index=False表示不按照组标签作为索引，方便二次处理
        self.count = self.data.groupby(QI, as_index=False).count()
        # 去掉冗余属性
        self.count = self.count[(QI+S)[0:len(QI)+1]]
        rename = {S[0]: 'num'}
        # dcount列重命名，仅包含QI+num
        self.count.rename(columns=rename, inplace=True)


    def change_k(self, k_new):
        '''将参数k更改为k_new'''
        self.k = k_new

    def change_ms(self, ms_new):
        '''将参数ms更改为ms_new'''
        self.ms = ms_new

    def show_data(self):
        '''测试用，输出data'''
        print(self.data)


    def build_tree(self):
        '''根据各QI的树形结构文件创建树形结构
            注意，建树完成后，数值型的h是包含了泛化结构的最底层“不泛化”'''
        # 对数据文件路径做分割，得到去除文件后缀的firstname
        namesplit = os.path.splitext(self.path)
        firstname = namesplit[len(namesplit) - 2]
        for qi in self.QI:
            # qi对应的泛化树形结构文件路径
            filename = firstname+'_'+qi+'.txt'
            # 建树
            self.tree[qi] = MyTree(qi)
            # 读文件，并进行字串处理，得到父子对应关系插入树中
            f = open(filename, mode='r')
            d = f.read()
            d = d.splitlines()
            # print('{0}:'.format(filename))
            for di in d:
                r = di.split(',')
                self.tree[qi].add_relation(r[1], r[0])  # 父子对应关系
            # 在树中创建高度->结点的对应关系
            self.tree[qi].build_h2node()
            # 在树中创建结点到叶节点的对应关系
            self.tree[qi].build_node2leaf()
            isdigit = self.tree[qi].build_isdigit()
            if isdigit == False:
                # 非数值型，创建h到泛化字典的对应关系
                self.tree[qi].build_h2GeneralizationDict()
                self.tree[qi].build_node2LM()
            else:
                # 数值型
                self.tree[qi].h = self.tree[qi].h + 1   # 最底层多加一层，因为数值型未把“未泛化”加入其中
            # self.tree[qi].print_tree()


    def show_tree(self):
        '''打印各个QI对应的泛化树，测试用'''
        for qi in self.QI:
            self.tree[qi].print_tree()


    def is_kanonymity(self, dom, do_evaluate):
        '''判别是否为k匿名\\
            其中dom为list型或ndarray数组，与self.QI对应，为收敛到第几层了\\
            而do_evaluate为布尔值，若为True时评估可用性损失并返回，否则返回的可用性损失为0'''
        # 用对self.count的泛化来模拟对数据集的泛化，简化了操作
        QI_num = len(self.QI)
        # 不能count_new = self.count！！！
        # 根据已有对象直接赋值产生的新对象，往往是原对象的映射，新对象改变原对象也会改变！
        # 也不能count_new = pd.DataFrame(self.count)，这种是浅拷贝，对多层嵌套结构，子对象仍为索引
        count_new = copy.deepcopy(self.count)
        i = 0
        while i < QI_num:
            # 逐个QI[i]，创建rename的dict，然后rename做泛化
            qi = self.QI[i]                             # 本轮循环处理的QI
            if self.tree[qi].isdigit == True:
                # 数值型(对应测试集中即为age)，单独创建泛化用的重命名字典rename、算LM用的node2LM，并传入
                #print('{0} h2GeneralizationDict len:{1}'.format(qi,len(self.tree[qi].h2GeneralizationDict)))
                if len(self.tree[qi].h2GeneralizationDict) == 0:
                    # 创建h2GeneralizationDict并传入
                    # 数值型qi，在dataframe中出现的所有值的实例vals
                    vals = np.array(self.count[qi].drop_duplicates())
                    h2GeneralizationDict = {}
                    h = 0
                    while h < self.tree[qi].h:
                        # 逐层（非最底层）创建对应的rename
                        rename = {}
                        # 本层对应的泛化区间range，默认数值型QI树形结构的配置文件中，每层仅1结点
                        r = self.tree[qi].h2node[h][0]
                        # print(r)
                        if r == '*':
                            # 默认数值型QI树形结构的配置文件中，只会有这一个非数值
                            for val in vals:
                                rename[val] = '*'
                            # LM一定为1
                            self.tree[qi].node2LM[r]=1
                        else:
                            # 对vals做区间泛化
                            # 待解决str(float('3'))=='3.0'的问题，为了适配浮点数，先不考虑解决，反正整数区间也能看；
                            # 可以考虑使用numpy的dtype来对不同类型分别处理
                            r = float(r)
                            for val in vals:
                                left = r * math.floor(val/r)    # val所在的泛化区间左端点
                                right = left + r                # 右端点，左闭右开
                                rename[val] = '[' + str(left) + ',' + str(right) + ')'
                                #print('{0} is in {1}'.format(val,rename[val]))
                                # LM为区间宽度r，除以所有区间的宽度和，即区间最右端点减去区间最左端点
                                self.tree[qi].node2LM[rename[val]] = r/(r * (math.floor(vals.max()/r)+1) - r * math.floor(vals.min()/r))
                        h2GeneralizationDict[h] = rename
                        h = h + 1
                    # 最底层泛化字典为“原始值->原始值”，LM为0
                    rename = {}
                    for val in vals:
                        rename[val] = val
                        self.tree[qi].node2LM[val] = 0
                    h2GeneralizationDict[h] = rename
                    # 生成完h2GeneralizationDict后传入数值型结构树
                    self.tree[qi].h2GeneralizationDict = h2GeneralizationDict
            # 取出qi对应的h2GeneralizationDict后，做替换泛化
            rename = self.tree[qi].h2GeneralizationDict[dom[i]]
            count_new[qi] = count_new[qi].map(rename)
            i = i + 1
        # 泛化完成，计数并判断k匿名（并做数据可用性评估）
        # 泛化后再分组聚集，此时将泛化前的count求和即为泛化后QIcluster的统计值
        count_new = count_new.groupby(self.QI, as_index=False).sum()
        # 统计不满足k匿名(即记录数<k)的QIcluster中的记录总数
        num = count_new[count_new['num'] < self.k]['num'].sum()
        # print('num={0}, dom={1}'.format(num,dom))
        value_utility = -1
        if do_evaluate:
            # 做评估
            value_utility = self.evaluate(count_new)
        if num <= self.ms:
            # 不发布的记录数满足要求
            return [True, value_utility]
        else:
            return [False, value_utility]


    def pub_data(self, dom):
        '''将数据按照dom做泛化，并发布成.data文件，返回生成的文件名'''
        QI_num = len(self.QI)
        data_new = copy.deepcopy(self.data)
        count_new = copy.deepcopy(self.count)
        i = 0
        print('=>begin publishing:')

        # 泛化
        while i < QI_num:
            qi = self.QI[i]                                         # 本轮循环处理的QI
            rename = self.tree[qi].h2GeneralizationDict[dom[i]]     # 此时必能取到，因为做过一次is_kanonymity了
            data_new[qi] = data_new[qi].map(rename)
            count_new[qi] = count_new[qi].map(rename)
            i = i + 1
        

        # 加索引，方便使用泛化后的QI查询记录(加速)
        data_new = data_new.set_index(self.QI)
        print('===>after generalization:')
        print(data_new)

        # Suppression
        # 获取待删除的QIcluster具体值，成为二维list(supression_QIcluster)
        count_new = count_new.groupby(self.QI, as_index=False).sum()
        # evaluate before suppression but after aggression(因为下面的聚集函数中，需要用num判断是否需要suppression)
        value_utility = self.evaluate(count_new)
        supression_QIcluster = count_new.loc[count_new['num'] < self.k][self.QI]
        supression_QIcluster = np.array(supression_QIcluster)
        supression_QIcluster = supression_QIcluster.tolist()
        # 通过supression_QIcluster来index_loc待删除的数据
        if len(supression_QIcluster) > 0:
            data_new = data_new.loc[data_new.index.difference(supression_QIcluster)]
        print('===>after supression:')
        print(data_new)

        # 发布
        # 对数据文件路径做分割，得到去除文件后缀的firstname
        namesplit = os.path.splitext(self.path)
        firstname = namesplit[len(namesplit) - 2]
        out_path = firstname + '_samarati.data'
        data_new.to_csv(out_path, index=True, header=False)
        print('===>publish file path:{0}'.format(out_path))

        print('=>end publishing, total lossmetric is:{0}'.format(value_utility))
        return out_path


    def multiply(self,x,y):
        '''自定义乘函数，用作计算Loss Metric的时候，按行处理\\
        y是QI cluster tuple number，x是属性对应的LM值'''
        if y < self.k:
            # 这个QIcluster是要被suppression的，故LM应用1而不是x
            return y
        else:
            return x*y


    def evaluate(self,count):
        '''LossMetric评价泛化后数据可用性损失：count为泛化后的QI cluster所有种类的num，构成的dataframe'''
        count_new = copy.deepcopy(count)
        for qi in self.QI:
            # 对每个qi，将属性值映射成为LM值后，乘上QI cluster tuple number
            count_new[qi] = count_new[qi].map(self.tree[qi].node2LM)
            count_new[qi] = count_new.apply(lambda row: self.multiply(row[qi], row['num']), axis=1)
        # 按列求和聚集
        count_new = count_new.sum()
        print(count_new)
        LossMetric = count_new[self.QI].sum() / count_new['num']
        return LossMetric


    def permutation(self,h,i,dom):
        '''DFS，对剩余可用高度总和h、当前QI[i]、这一层的dom(np.array型)，递归判断“本层”构成的dom组合能否满足k匿名'''
        qi = self.QI[i]             # 当前处理的qi
        QI_num = len(self.QI)       # QI总个数
        j = min(h, self.tree[qi].h) # 迭代用，从大的开始迭代
        j_end = 0
        k = i+1
        while k<QI_num:
            j_end = j_end + self.tree[self.QI[k]].h
            k = k+1
        j_end = max(0, h-j_end)
        while(j>=j_end):
            # 通过对dom[i]的修改，对可能的情况做遍历
            if(i != QI_num-1):
                # 不是最后一个qi，则对下一个qi递归做判断
                dom[i] = j
                h_new = h - j
                i_new = i + 1
                if(self.permutation(h_new,i_new,dom)):
                    # 找到了，则不继续遍历，返回已经找到的
                    return True
            else:
                # 是最后一个qi
                if(h <= self.tree[qi].h):
                    # 符合求和等于给定值的要求
                    dom[i] = h
                    if(self.is_kanonymity(dom,False)[0]):
                        return True
                    else:
                        return False
                else:
                    return False
            j = j - 1
        return False


    def find_opt(self, h, i, dom):
        '''待完成，搜索并返回第h层中LM最优的泛化结果dom'''
        qi = self.QI[i]             # 当前处理的qi
        QI_num = len(self.QI)       # QI总个数
        j = min(h, self.tree[qi].h) # 迭代用，从大的开始迭代
        j_end = 0
        k = i+1
        while k<QI_num:
            j_end = j_end + self.tree[self.QI[k]].h
            k = k+1
        j_end = max(0, h-j_end)
        while(j>=j_end):
            # 通过对dom[i]的修改，对可能的情况做遍历
            if(i != QI_num-1):
                # 不是最后一个qi，则对下一个qi递归做判断
                dom[i] = j
                h_new = h - j
                i_new = i + 1
                if(self.permutation(h_new,i_new,dom)):
                    # 找到了，则不继续遍历，返回已经找到的
                    return True
            else:
                # 是最后一个qi
                if(h <= self.tree[qi].h):
                    # 符合求和等于给定值的要求
                    dom[i] = h
                    if(self.is_kanonymity(dom,False)[0]):
                        return True
                    else:
                        return False
                else:
                    return False
            j = j - 1
        return False

    def search(self):
        '''将数据做快速k匿名处理并发布成.data文件，返回生成的文件名'''
        # 快速求出一个可行解
        # 维护list，表示各QI已经泛化到第几层了
        # 快速搜解：整体二分
        start = time.time()                     # 开始时间
        QI_num = len(self.QI)
        dom_right = np.zeros(QI_num,dtype=int)  # 泛化层数上界数组
        dom_left = np.zeros(QI_num,dtype=int)   # 泛化层数下界数组
        left = 0    # 泛化层数和下界
        right = 0   # 泛化层数和上界
        # initial
        i = 0
        while i < QI_num:
            qi = self.QI[i]
            dom_right[i] = self.tree[qi].h
            right = right + dom_right[i]
            i = i + 1
        print('=>begin searching: k={0}, ms={1}'.format(self.k, self.ms))
        # 按照层数分别二分搜索，找满足k匿名的最深
        while (dom_right==dom_left).all() == False:
            # 迭代直到区间端点相等
            dom = np.add(dom_left,dom_right)
            dom = np.divide(dom,2)
            dom_judge = np.ceil(dom).astype(int)
            if self.is_kanonymity(dom_judge,False)[0]:
                # 满足k匿名，向下找
                dom_left = dom_judge
            else:
                # 不满足k匿名，向上找
                dom_right = np.floor(dom).astype(int)
            print('===>left:{0}, right:{1}'.format(dom_left, dom_right))
        # 按照层数的和，二分搜索，TreeSearch架构
        left = np.array(dom_left).sum()
        dom = np.zeros(QI_num,dtype=int)            # 单个dom，迭代用；由于python是索引传递，所以在函数中修改dom内的值，会反映到函数外
        while(left!=right):
            mid = math.ceil((left+right)/2)
            print('===>left:{0}, right:{1}, mid:{2}'.format(left, right, mid))
            if(self.permutation(mid,0,dom)):
                # 该层有满足k匿名的，则向下找
                left = mid
            else:
                # 该层没有满足k匿名的，则向上找
                right = math.floor((left+right)/2)
        # 搜索完成，发布数据
        self.find_opt(left, 0, dom)
        print('=>end searching, result: dom={0}\n'.format(dom))
        self.pub_data(dom)
        # 打印测量的时间
        end = time.time()   # 结束时间
        print('=>total search time is:{0}(s)\n'.format(end-start))


    def search_opt(self):
        '''将数据做最优(?)k匿名处理并发布成.data文件，返回生成的文件名'''
        # 搜索一个可用性尽可能高的解，考虑A*
        # 利用pub_data生成的可行解，在剩下的搜索中用该可行解做剪枝（使用LM衡量数据可用性）
        pass
