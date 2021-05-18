'''
k匿名两种算法的python3.6.9实现\\
依赖：pandas、numpy\\
实现了samarati算法、mondrian算法，并提供了加载.data数据的方法
'''
from k_anonymity.samarati import samarati
from k_anonymity.mondrian import mondrian
from k_anonymity.load import loaddata
