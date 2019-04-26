最常见的就是使用最大最下值来进行处理，处理的公式如下

y = (x − min) / (max − min)
上述公式中 min 代表数据最小值，max 代表数据最大值

```Python
from sklearn.preprocessing import  MinMaxScaler
tmp=MinMaxScaler().fit_transform(irisdata.data)
print tmp[0:5]
```

部分结果如下：

```
[[ 0.22222222 0.625 0.06779661 0.04166667]
[ 0.16666667 0.41666667 0.06779661 0.04166667]
[ 0.11111111 0.5 0.05084746 0.04166667]
[ 0.08333333 0.45833333 0.08474576 0.04166667]
[ 0.19444444 0.66666667 0.06779661 0.04166667]]
```

### spark 中的区间缩放法

```python
>>>from pyspark.mllib.linalg import Vectors
>>>from pyspark.sql import SQLContext
>>>sqlContext=SQLContext(sc)
>>>df = sqlContext.createDataFrame([(Vectors.dense([0.0]),), (Vectors.dense([2.0]),)], ["a"])
>>> mmScaler = MinMaxScaler(inputCol="a", outputCol="scaled")
>>> model = mmScaler.fit(df)
>>> model.transform(df).show()
'''
+-----+------+
|    a|scaled|
+-----+------+
|[0.0]| [0.0]|
|[2.0]| [1.0]|
'''
```
