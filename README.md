# k匿名两种算法的python实现

## 环境

conda环境配置如下：

```bash
conda create -n k_anonymity python=3.6.9 -y
conda activate k_anonymity
conda install -c conda-forge numpy pandas pyyaml -y
```

## 简介

本仓库实现了k匿名的samarati算法和mondrian算法，做成了即拆即用、适用性强的python模块；下面是两种算法的简介和Quick Startup。

### k匿名

预定义待发布的隐私属性S、Quasi-identifiers属性QI，通过对QI属性在数据集中出现的实例进行一定程度的泛化(Generalization)，达成如下效果，称为使数据集满足k匿名：

按照QI属性的每一种实例组合，对数据集进行分组，每个QI cluster中所包含的元组tuples的个数不小于k。

### samarati算法

1. 技术：

   - Generalization：泛化；参照一定的树形结构，将QI属性的实例做多对一映射，相当于对QI cluster做了聚集。
   - Suppression：不发布，删除；即进行一定泛化操作后，若仍有不满足k匿名的QI cluster，则不发布这些QI cluster所含的所有tuples（即这些数据不会出现在最后发布的数据集中，即删除这些数据）。

2. 准备：

   - 给定数据集、数据集属性名attributes、QI、S、匿名数k、最大不发布tuple数ms；

   - 给定各个QI cluster泛化需要参照的树形结构的子->父关系，使得能通过这样的结构对数据集中的QI属性做泛化。

3. 基本过程：

   维护一个len(QI)维的向量dom，用于表示当前各QI泛化到树形结构对应的哪一层了（这里为了程序的可用性，使用了full domain型泛化方式）；对向量dom形成的向量空间做快速二分搜索，搜索得一个泛化后满足k匿名和ms的最大不发布个数，并且dom各元素求和最大的向量dom（泛化到树形结构越高层，越容易满足k匿名）。

### mondrian算法

1. 技术：

   - Generalization：泛化；
   - Priority Queue：优先队列（最大），队列元素[key,value]按照key有优先级，队头一定是优先级最高的，用于代替递归。

2. 准备：

   - 给定数据集、数据集属性名attributes、QI、S、匿名数k；
   - 初始化最大优先队列为单一元素，key是数据集中tuples数 ，value是整个数据集，代表“将整个数据集泛化成一个QI cluster”。

3. 基本过程：循环：

   每次循环，对最大的QI cluster单独选择一个属性（可以选择范围最大的属性或者随机选），找到属性的中位数，对Partition划分；

   重复上述过程，直到不能划分（当前最大的QI cluster的大小在[k,2k)区间内）为止。

## Use it in your code

代码实现了一个基于python3的模块k_anonymity，使用时仅需要引入该模块，定义所需使用的算法对象，在配置好的conda环境下执行算法即可。

1. 准备：

   - 待处理的数据集（目前仅支持.data等csv类型）；
   - 若欲使用samarati算法，需要在数据集同一文件夹下，放置树形结构.txt文件；树形结构.txt文件中，是按照“子,父”的模式描述树形结构的；
   - 需要将调用模块的python脚本跟模块文件夹k_anonymity置于同一父文件夹下，或需要将k_anonymity模块的路径加入环境变量；
   - 最后发布的数据集也会放到与数据集相同的父文件夹下。

2. 使用：

   使用方法见如下示例代码：

   [samarati](./test1.py)：

   ```python
   from sys import argv	# 脚本参数
   import k_anonymity		# 引入模块
   # Begin:预定义path、attributes、QI、S等参数
   script, path, k, ms = argv[0], argv[1], argv[2], argv[3]
   # 注：path可以是相对路径如./data/adult.data，但是要确保树形结构的文件与data文件在同一文件夹下
   attributes = ['age', 'work_class', 'final_weight', 'education',
                 'education_num', 'marital_status', 'occupation', 'relationship',
                 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'class']
   QI = ['age', 'gender', 'race', 'marital_status']
   S = ['occupation']
   # End:预定义
   # Begin:samarati
   s = k_anonymity.samarati(path,int(k),int(ms),attributes,QI,S)		# 定义算法对象
   s.build_tree()														# 读入并创建树形结构
   s.search()															# 做samarati搜索，并发布数据
   # Ended:samarati
   ```

   [mondrian](./test2.py)：

   ```python
   from sys import argv	# 脚本参数
   import k_anonymity		# 引入模块
   # Begin:预定义path、attributes、QI、S等参数
   script, path, k = argv[0], argv[1], argv[2]
   attributes = ['age', 'work_class', 'final_weight', 'education',
                 'education_num', 'marital_status', 'occupation', 'relationship',
                 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'class']
   QI = ['age', 'gender', 'race', 'marital_status']
   S = ['occupation']
   # End:预定义
   # Begin:mondrian
   m = k_anonymity.mondrian(path, int(k), attributes, QI, S)			# 定义算法对象
   m.search()															# 做mondrian划分，并发布数据
   # End:mondrian
   ```

## My experiment

由于实验需要，我实现了两个示例程序，`test1.py`和`test2.py`，可在命令行执行以下代码获取提示，不再赘述

```bash
(k_anonymity) yuzijian@linke6:~/k_anonymity$ python test1.py -h
usage: test1.py [-h] [-ms MS] path k

you should add those parameter

positional arguments:
  path        The path of the raw data file (*.data)
  k           The value of k for k_anonymity

optional arguments:
  -h, --help  show this help message and exit
  -ms MS      The value of max suppression
```