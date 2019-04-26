> https://www.deeplearn.me/1376.html

## 特征工程（1）-数据预处理标准化

机器学习中特征工程的构造分析，以前在这方便还是没有去全面的了解，最近有一段磨刀的时间，还是从基础学习开始，理论结合代码推进

**通过特征提取，我们能得到未经处理的特征**，这时的特征可能有以下问题：

- 量纲不一致 : 不属于同一量纲，即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。比如身高和年龄

- 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。二值化可以解决这一问题。

- 定性特征不能直接使用：某些机器学习算法和模型只能接受定量特征的输入，**那么需要将定性特征转换为定量特征**。最简单的方式是为每一种定性值指定一个定量值，但是这种方式过于灵活，增加了调参的工作。**通常使用哑编码的方式将定性特征转换为定量特征**：假设有`N`种定性值，则将这一个特征扩展为`N`种特征，当原始特征值为第`i`种定性值时，第`i`个扩展特征赋值为`1`，其他扩展特征赋值为`0`。哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用哑编码后的特征可达到非线性的效果。比如当前属性有 5 种情况，然后当前样本 x 拥有当前属性第三种情况，可以构造特征向量(0,0,1,0,0)，这就是哑编码的过程.

- 存在缺失值：缺失值需要补充。常见的有均值还有众数，中值来补充

- 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的转换，都能达到非线性的效果。
我们使用 sklearn 中的`preproccessing`和`spark.ml.feature`库来进行数据预处理，可以覆盖以上问题的解决方案。

### 无量纲处理

标准化处理

标准化处理会用到数据的均值和标准差，标准化的结果反映了数据围绕均值上下波动的情况，

- 数据小于 0 则表示当前数据是低于平均值水平
- 数据的绝对值反映的偏离平均值的程度，数值绝对值越大则表示偏离越远
下面看下计算公式

x = (x − 均值)/标准差
sklearn 中的处理如下

```Python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

irisdata=load_iris()
print irisdata.data
stdmethod=StandardScaler()
stddata=stdmethod.fit_transform(irisdata.data)
```
标准化后部分数据如下

```
[[ -9.00681170e-01   1.03205722e+00  -1.34127240e+00  -1.31297673e+00]
 [ -1.14301691e+00  -1.24957601e-01  -1.34127240e+00  -1.31297673e+00]
 [ -1.38535265e+00   3.37848329e-01  -1.39813811e+00  -1.31297673e+00]
 [ -1.50652052e+00   1.06445364e-01  -1.28440670e+00  -1.31297673e+00]
 [ -1.02184904e+00   1.26346019e+00  -1.34127240e+00  -1.31297673e+00]
 [ -5.37177559e-01   1.95766909e+00  -1.17067529e+00  -1.05003079e+00]
 [ -1.50652052e+00   8.00654259e-01  -1.34127240e+00  -1.18150376e+00]
 [ -1.02184904e+00   8.00654259e-01  -1.28440670e+00  -1.31297673e+00]
 [ -1.74885626e+00  -3.56360566e-01  -1.34127240e+00  -1.31297673e+00]
 [ -1.14301691e+00   1.06445364e-01  -1.28440670e+00  -1.44444970e+00]
 ```

下面解析 StandardScaler 方法的内部函数

```Python
sklearn.preprocessing.StandardScaler(<copy=True,with_mean=True,with_std=True)
#参数 copy 当为 True 的时候是要返回数据的备份，一般都是默认 true
# with_mean true 时是在缩放之前中心数据
# with_std 在数据缩放至单位方差或者单位标准差
#方法函数
fit(X[, y])#计算数据的均值和标准方差
fit_transform(X[, y])#先计算均值和方差在转换数据，就是 fit 和 transform 两部操作合二为 1
get_params([deep])#获取当前估计器的参数
inverse_transform(X[, copy])#将当前的数据返回至之前的状态
partial_fit(X[, y])#在线计算数据的均值和方差然后用于后面的缩放处理
set_params(**params)#自定义参数用于估计器
transform(X[, y, copy])#缩放数据
```

spark 版本

```Python
class pyspark.ml.feature.StandardScaler(self, withMean=False,
 withStd=True, inputCol=None, outputCol=None)
#参数
#这里参数的含义可以参考上面部分的描述，区别在于 inputCol。。
#inputCol 是输入的 dataframe 数据中的需要处理的列，那么 outputCol 就是输出的列了
```

```Python
from pyspark.sql import SQLContext
from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import StandardScaler
conf=SparkConf().setAppName('decsion').setMaster('local')
sc=SparkContext(conf=conf)
sqlContext=SQLContext(sc)
dataFrame = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
scaledData.show()
```
