# 机器学习: scikit-learn 中的设置以及预估对象

## 数据集

Scikit-learn可以从一个或者多个数据集中学习信息，这些数据集合可表示为2维阵列，也可认为是一个列表。列表的第一个维度代表**样本**，第二个维度代表**特征**（每一行代表一个样本，每一列代表一种特征）。

样例: iris 数据集（鸢尾花卉数据集）

```Python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> data = iris.data
>>> data.shape
(150, 4)
```

这个数据集包含150个样本，每个样本包含4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，详细数据可以通过`iris.DESCR`查看。

如果原始数据不是`(n_samples, n_features)`的形状时，使用之前需要进行预处理以供scikit-learn使用。

数据预处理样例:digits数据集(手写数字数据集)

digits数据集包含1797个手写数字的图像，每个图像为8*8像素

```Python
>>> digits = datasets.load_digits()
>>> digits.images.shape
(1797, 8, 8)
>>> import matplotlib.pyplot as plt
>>> plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
<matplotlib.image.AxesImage object at ...>
```

![](./1.png)

为了在scikit中使用这一数据集，需要将每一张8×8的图像转换成长度为64的特征向量

```Python
>>> data = digits.images.reshape((digits.images.shape[0], -1))
```

## 预估对象

**拟合数据**: scikit-learn实现最重要的一个API是`estimator`。`estimators`是基于数据进行学习的任何对象，它可以是一个分类器，回归或者是一个聚类算法，或者是从原始数据中提取/过滤有用特征的变换器。
所有的拟合模型对象拥有一个名为`fit`的方法，参数是一个数据集（通常是一个2维列表）:

**拟合模型对象构造参数**: 在创建一个拟合模型时，可以设置相关参数，在创建之后也可以修改对应的参数:

```Python
>>> estimator = Estimator(param1=1, param2=2)
>>> estimator.param1
```

**拟合参数**: 当拟合模型完成对数据的拟合之后，可以从拟合模型中获取拟合的参数结果，所有拟合完成的参数均以下划线`_`作为结尾:

```Python
>>> estimator.estimated_param_
```
