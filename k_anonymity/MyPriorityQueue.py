#!python3
# coding=UTF-8


class MyPriorityQueue:
    '''使用二叉堆的最小优先队列的python3.6.9实现，方便做A* search和Mondrian\\
        优先级函数f(a,b)返回"(a优先级比b高)?"'''

    def __init__(self, f):
        '''初始化最小优先队列为空，二元优先级函数为f(a,b)'''
        self.queue = []
        self.function = f

    def father(self, i):
        '''得到queue[i-1]在二叉堆中的父节点queue[j-1]的位置j'''
        j = int(i/2)
        return j

    def left(self, i):
        '''得到queue[i-1]在二叉堆中的左孩子queue[j-1]的位置j'''
        j = 2*i
        return j

    def right(self, i):
        '''得到queue[i-1]在二叉堆中的右孩子queue[j-1]的位置j'''
        j = 2*i + 1
        return j

    def enqueue(self, key, value):
        '''入队，[key,value]'''
        self.queue.append([key, value])
        self.maintain(0)     # 自底向上维护

    def dequeue(self):
        '''出队，res=[key,value]'''
        length = len(self.queue)
        # 交换头尾，将最小值移至末尾
        t = self.queue[0]
        self.queue[0] = self.queue[length-1]
        self.queue[length-1] = t
        # 最小值出队
        self.queue.pop()
        # 自顶向下维护最小优先队列性质
        self.maintain(1)
        # 返回出队的最小值
        return t

    def maintain(self, tag):
        '''维护最小优先队列性质，tag为1时下沉，为0时上浮'''
        length = len(self.queue)
        if tag == 1:
            # 自顶向下下沉
            p = 1   # 迭代用
            while(self.right(p) <= length or self.left(p) <= length):
                # p还没到底
                q = self.left(p)    # q为待与p判断的
                if self.right(p) <= length:
                    # 有右孩子，则q取优先级高的那个，注意queue取元素时一定-1！
                    if self.function(self.queue[self.right(p)-1][0], self.queue[q-1][0]):
                        q = self.right(p)
                if self.function(self.queue[q-1][0], self.queue[p-1][0]):
                    # q优先级比p高，则交换队列中p、q位置的值，并以q为新p继续迭代
                    t = self.queue[q-1]
                    self.queue[q-1] = self.queue[p-1]
                    self.queue[p-1] = t
                    p = q
                else:
                    # p优先级高，则已经到对应位置了，迭代结束
                    break
        else:
            # 自底向上上浮
            p = length
            while p > 1:
                # p还没到顶
                q = self.father(p)
                if self.function(self.queue[p-1][0], self.queue[q-1][0]):
                    # p优先级高，则交换队列中p、q位置的值，并以q为新p继续迭代
                    t = self.queue[q-1]
                    self.queue[q-1] = self.queue[p-1]
                    self.queue[p-1] = t
                    p = q
                else:
                    # q优先级高，则已经到对应位置了，迭代结束
                    break


    def is_empty(self):
        '''判断并返回优先队列是否为空'''
        return len(self.queue) == 0
    

    def test_queue(self):
        '''测试优先队列'''
        self.enqueue(4, 'd')
        print(self.queue)
        self.enqueue(3, 'c')
        print(self.queue)
        self.enqueue(1, 'a')
        print(self.queue)
        self.enqueue(2, 'b')
        print(self.queue)
        self.enqueue(7, 'g')
        print(self.queue)
        self.enqueue(5, 'e')
        print(self.queue)
        self.enqueue(6, 'f')
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
        self.dequeue()
        print(self.queue)
