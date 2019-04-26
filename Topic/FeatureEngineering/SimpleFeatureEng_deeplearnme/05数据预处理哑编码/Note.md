### 哑编码概念

先来讲解下哑编码的概念吧:

**当你的变量不是定量特征的时候是无法拿去进行训练模型的，哑编码主要是针对定性的特征进行处理然后得到可以用来训练的特征**

关于定性和定量还是在这里也说明下，举个例子就可以看懂了

#### 定性：

博主很胖

博主很瘦

#### 定量:

博主有 80kg

博主有 60kg（ps：好难啊）

一般定性都会有相关的描述词，定量的描述都是可以用数字来量化处理

现在假设给你的一个病人的病情描述，一般病情的描述包含以下几个方面，将病情严重程度划分：

非常严重，严重，一般严重，轻微

现在有个病人过来了，要为他构造一个病情的特征，假设他的病情是严重情况，我们可以给他的哑编码是

0 1 0 0

病情总共有四种情况因此使用四位来表示，第二位表示严重，这位病人是严重的病情因此将其置为 1，其余为 0

以上就是哑编码的原理，看完这个解释应该可以理解了，如果还是看不懂，我逃/(ㄒoㄒ)/~~

sklearn 代码剖析

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[1, 1, 4], [2, 2, 1], [1, 3, 2], [2, 1, 3]])
print enc.n_values_
print enc.feature_indices_
print enc.transform([[1, 1, 4], [2, 2, 1], [1, 3, 2], [2, 1, 3]])
print enc.transform([[1, 2, 2]]).toarray()
```

```
输出的结果

#输出 3，4，5 分别表示当前该维特征有三种、4 种和 5 种情况，案例只有 2，3，4，该问题可以看下一段解释
[3 4 5]
#表示每个特征的哑编码的索引范围
[ 0  3  7 12]
#返回哑编码的缩减版说明,举个例子说明一下，（0，8）表示索引 0 和 8   1.0 表示在这两个索引的数据都是 1
  (0, 8)	1.0
  (0, 2)	1.0
  (0, 0)	1.0
  (1, 5)	1.0
  (1, 3)	1.0
  (1, 1)	1.0
  (2, 6)	1.0
  (2, 4)	1.0
  (2, 0)	1.0
  (3, 7)	1.0
  (3, 2)	1.0
  (3, 1)	1.0
[[ 1.  0.  0.  1.  0.  0.  1.  0.  0.]]
```

哑编码的大小说明

```python
def _fit_transform(self, X):
    """Assumes X contains only categorical features."""
  X = check_array(X, dtype=np.int)
  if np.any(X < 0):
    raise ValueError("X needs to contain only non-negative integers.")
  n_samples, n_features = X.shape
'''
这一步就是比较关键啦，我们传入的参数是 auto,此时他会寻找特征列的最大值然后对其加 1 处理

然后将数据返回给 n_values，这就是我们刚才看到的数据 3，4，5，虽然我们第一维特征数字只有 1 和 2，但是取最大值加 1 就变为 3 了

这也就解释了出现这种情况的原因，看到这里就顿悟了，遇到问题还是可以看看源码，这样可以理解的前提是代码还是简单点
'''
    if (isinstance(self.n_values, six.string_types) and self.n_values == 'auto'):
      n_values = np.max(X, axis=0) + 1
    elif isinstance(self.n_values, numbers.Integral):
      if (np.max(X, axis=0) >= self.n_values).any():
        raise ValueError("Feature out of bounds for n_values=%d" % self.n_values)
        n_values = np.empty(n_features, dtype=np.int)
        n_values.fill(self.n_values)
    else:
      try:
        n_values = np.asarray(self.n_values, dtype=int)
      except (ValueError, TypeError):
        raise TypeError("Wrong type for parameter `n_values`. Expected"
            " 'auto', int or array of ints, got %r" % type(X))
        if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
          raise ValueError("Shape mismatch: if n_values is an array,"" it has to be of shape (n_features,).")
    self.n_values_ = n_values

```

### spark 代码剖析

```
<em class="property">class </em><tt class="descclassname">pyspark.ml.feature.</tt><tt class="descname">OneHotEncoder</tt><big>(</big><em>self</em>, <em>includeFirst=True</em>, <em>inputCol=None</em>, <em>outputCol=None</em><big>)</big>
#不得不说 spark ml 受 sklearn 启发，基本上二者的 api 定义基本一致，不过这样也好，基本上记住其中一个，另外一个基本上套着用
```

```Python
>>> from pyspark.ml.feature import StringIndexer
>>> from pyspark.ml.feature import OneHotEncoder
>>> from pyspark.sql import SQLContext    
>>> sq=SQLContext(sc)

>>> data=sq.createDataFrame([(1, 1, 4),(2, 2, 1),(2, 1, 3),(3,4,5)], ["a","b","c"])  
>>> strmodel=StringIndexer(inputCol='a',outputCol='features')        
>>> model=strmodel.fit(data)
>>> tdd=model.transform(data)
>>> tdd.show()
+---+---+---+--------+
|  a|  b|  c|features|
+---+---+---+--------+
|  1|  1|  4|     2.0|
|  2|  2|  1|     0.0|
|  2|  1|  3|     0.0|
|  3|  4|  5|     1.0|
+---+---+---+--------+
>>> encoder = OneHotEncoder(inputCol="features", outputCol="feature")
>>> aa=encoder.transform(tdd).show()
+---+---+---+--------+-------------+
|  a|  b|  c|features|      feature|
+---+---+---+--------+-------------+
|  1|  1|  4|     2.0|    (2,[],[])|
|  2|  2|  1|     0.0|(2,[0],[1.0])|
|  2|  1|  3|     0.0|(2,[0],[1.0])|
|  3|  4|  5|     1.0|(2,[1],[1.0])|
+---+---+---+--------+-------------+
#此时发现为什么 a 列有三类结果只显示两类，因为 spark 默认忽略了最后一位，现在我们
#自定义参数让其显示出来

>>> params = {encoder.dropLast: False, encoder.outputCol: "test"}
>>> encoder.transform(tdd, params).show()
+---+---+---+--------+-------------+
|  a|  b|  c|features|         test|
+---+---+---+--------+-------------+
|  1|  1|  4|     2.0|(3,[2],[1.0])|
|  2|  2|  1|     0.0|(3,[0],[1.0])|
|  2|  1|  3|     0.0|(3,[0],[1.0])|
|  3|  4|  5|     1.0|(3,[1],[1.0])|
+---+---+---+--------+-------------+
```

备注：sklearn 的哑编码与 spark 不一样，sklearn 一次性编码所有特征列，spark 不会
