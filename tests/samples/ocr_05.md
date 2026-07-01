# 纯 Python 快速判断方法：

## 方法 1：itertools （极快，推荐）

用 `itertools` 配对，底层 C 实现，比手动循环快很多

import itertools
def same_diff(a, b):

先算第一个差值

​	d = b[0] - a[0]

一键判断所有位置

​	return all(x2 - x1 == d for x1, x2 in itertools.zip_longest(a, b))



![1782898211528](C:\Users\14155\AppData\Roaming\Typora\typora-user-images\1782898211528.png)