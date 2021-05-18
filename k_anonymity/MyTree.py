#!python3
# coding=UTF-8


class MyTree:
    '''泛化树形结构的python3.6.9实现\\
    构建树的步骤：init -> add_relation*n -> build_h2node -> build_node2leaf -> build_isdigit -> (not digit)build_h2GeneralizationDict -> build_node2LM\\
    最终只剩下树高h、isdigit、h2node、高度到泛化字典的映射h2GeneralizationDict、node2LM这五个属性'''

    def __init__(self, name):
        self.name = name
        self.son2father = {}            # 子->父，一对一，建树用
        self.father2son = {}            # 父->子，一对多，建树用
        self.node = []                  # 结点列表，存储字符串，建树用
        self.h = 0                      # 树高(算边而不是算点，与PPT中不同，是为了方便使用列表映射h2node)
        self.h2node = []                # 高度->结点list
        self.node2leaf = {}             # 树节点映射到以他为根的子树的叶节点
        self.isdigit = False            # 树对应的QI是不是数值型
        self.h2GeneralizationDict = {}  # h映射到泛化重命名dict
        self.node2LM = {}               # 结点名映射到对应结构的LossMetric

    def add_relation(self, father, son):
        '''添加父子关系，father和son均为字符串'''
        self.son2father[son] = father

        if father not in self.father2son:
            self.father2son[father] = [son]
        else:
            self.father2son[father].append(son)

        if son not in self.node:
            self.node.append(son)
        if father not in self.node:
            self.node.append(father)

    def build_h2node(self):
        '''构建h2node映射，使得能通过h直接查询node集'''
        # 求树根
        name = self.node[0]
        while(name in self.son2father):
            name = self.son2father[name]

        # 求树高
        h = 0
        while(name in self.father2son):
            name = self.father2son[name][0]
            h = h + 1
        self.h = h

        # 构建h2node空列表
        i = 0
        while i <= h:
            self.h2node.append([])
            i = i + 1

        # 构建h2node，相当于对每个结点求树高
        for name in self.node:
            i = 0
            t = name
            while(t in self.son2father):
                t = self.son2father[t]
                i = i + 1
            self.h2node[i].append(name)

    def build_isdigit(self):
        '''判断是否为数值型并返回'''
        self.isdigit = self.h2node[self.h-1][0].isdigit()
        return self.isdigit

    def print_tree(self):
        '''打印树，测试用'''
        i = 0
        print('{0} tree, h=={1}:'.format(self.name, self.h))
        while i < self.h:
            print(self.h2node[i])
            i = i + 1
        if self.isdigit == False:
            # 区分是否为数值型，因为是在建树后打印，此时数值型的h为包含最底层“不泛化”的
            print(self.h2node[self.h])
        print('type is digit = {}'.format(self.isdigit))

    def build_node2leaf(self):
        '''构建node2leaf，使得能够直接查询以结点为根的子树的叶节点'''
        for name in self.node:
            # 为树中所有结点建空列表
            self.node2leaf[name] = []
        for name in self.node:
            if name not in self.father2son:
                # 为整棵树的叶节点
                self.node2leaf[name].append(name)  # 自身映射到自身
                p = name
                while (p in self.son2father):
                    self.node2leaf[self.son2father[p]].append(name)
                    p = self.son2father[p]


    def nodes_in_h(self, h):
        '''用h查询对应高度的nodelist'''
        return self.h2node[h]


    def build_h2GeneralizationDict(self):
        '''为非数值型的，创建树高h到对应的泛化字典的映射'''
        h = 0
        while h <= self.h:
            rename = {}
            for name in self.h2node[h]:
                for leaf in self.node2leaf[name]:
                    rename[leaf] = name     # 泛化时，leaf泛化到name
            self.h2GeneralizationDict[h] = rename
            h = h+1

    def build_node2LM(self):
        '''预先创建好node向Loss metric的映射，方便算LM评估'''
        if(self.isdigit):
            # 数值型在算h2GeneralizationDict的时候构建node2LM
            pass
        else:
            # 非数值型
            h = 0
            num_all = len(self.h2node[self.h])
            while(h <= self.h):
                for node in self.h2node[h]:
                    # 该结点对应的LM=（该结点为根的子树叶节点数目-1）/（总的叶节点数目-1）
                    self.node2LM[node] = (len(self.node2leaf[node])-1)/(num_all - 1)
                h = h + 1
        
        # 删除冗余
        del self.father2son
        del self.son2father
        del self.node
        del self.node2leaf
