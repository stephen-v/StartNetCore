<a id="markdown-1-tensorflow梯度下降与损失函数" name="1-tensorflow梯度下降与损失函数"></a>
# 1. Tensorflow梯度下降与损失函数

>在本节中将通过一个预测房屋价格的实例来讲解梯度下降和损失函数的原理，以及在tensorflow中如何实现
<!-- TOC -->

- [1. Tensorflow梯度下降与损失函数](#1-tensorflow梯度下降与损失函数)
    - [1.1. 准备工作](#11-准备工作)
    - [1.2. 归一化数据](#12-归一化数据)
    - [1.3. 用随机的值填充a,b并计算误差，误差采用上文所使用SSE(和方差)](#13-用随机的值填充ab并计算误差误差采用上文所使用sse和方差)
    - [1.4. 计算误差梯度](#14-计算误差梯度)
    - [1.5. 调整参数直到SSE参数最小](#15-调整参数直到sse参数最小)
    - [1.6. 概念](#16-概念)
        - [1.6.1 简单线性回归](#161-简单线性回归)
        - [1.6.2 梯度下降](#162-梯度下降)
            - [梯度](#梯度)
            - [步长](#步长)

<!-- /TOC -->
<a id="markdown-11-准备工作" name="11-准备工作"></a>
## 1.1. 准备工作

从网上得到的数据可以看到房屋价格与房屋尺寸的一个对比关系，如下图：

![2017-09-01-13-36-10](http://qiniu.xdpie.com/2017-09-01-13-36-10.png?imageView2/2/w/700&_=5603928)


我们假设x轴（房屋尺寸）而Y轴（房屋价格）依据上表数据绘制折线图

![2017-09-01-13-37-04](http://qiniu.xdpie.com/2017-09-01-13-37-04.png?imageView2/2/w/700&_=5603928)

现在我们使用简单的线性模型来预测，

* 红线表述我们的预测曲线 ： $$y_p=ax+b$$
* 蓝线表述房屋价格与尺寸的实际关系
* 预测与实际的不同用黄线表示
![2017-09-01-13-39-58](http://qiniu.xdpie.com/2017-09-01-13-39-58.png?imageView2/2/w/700&_=5603928)

接下来需要通过数据来找到a,b的最佳值从而使预测与实际的误差最小。此次我们采用SSE(和方差)来判别误差。该统计参数计算的是拟合数据和原始数据对应点的误差的平方和，计算公式如下

$$\frac{1}{2}\sum_{k=1}^{n} \ {(\bold{y} -\bold{y_p})^2}$$

在拿到原始的数据后，为方便运算，我们将数据进行归一化处理，归一化计算公式如下

$$\frac{x-x_{min}}{x_{max}-x_{min}}$$

<a id="markdown-12-归一化数据" name="12-归一化数据"></a>
## 1.2. 归一化数据

我们将原始的数据进行归一化处理，归一化处理后的结果如图：

![2017-09-01-14-04-28](http://qiniu.xdpie.com/2017-09-01-14-04-28.png?imageView2/2/w/700&_=5603928)

```Python

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_out = []
    for item in arr:
        out = np.divide(np.subtract(item, arr_min), np.subtract(arr_max, arr_min))
        arr_out = np.append(arr_out, np.array(out))
    return arr_out

```

<a id="markdown-13-用随机的值填充ab并计算误差误差采用上文所使用sse和方差" name="13-用随机的值填充ab并计算误差误差采用上文所使用sse和方差"></a>
## 1.3. 用随机的值填充a,b并计算误差，误差采用上文所使用SSE(和方差)

![2017-09-01-14-07-11](http://qiniu.xdpie.com/2017-09-01-14-07-11.png?imageView2/2/w/700&_=5603928)

```Python

def model(x, b, a):
    # linear regression is just b*x + a, so this model line is pretty simple
    return tf.multiply(x, b) + a

loss = tf.multiply(tf.square(Y - y_model), 0.5)

```

<a id="markdown-14-计算误差梯度" name="14-计算误差梯度"></a>
## 1.4. 计算误差梯度

对sse分别求a,b的偏微分
$$\frac{\partial sse}{\partial a}$$ 
$$\frac{\partial sse}{\partial b}$$

![2017-09-01-14-22-22](http://qiniu.xdpie.com/2017-09-01-14-22-22.png?imageView2/2/w/700&_=5603928)

<a id="markdown-15-调整参数直到sse参数最小" name="15-调整参数直到sse参数最小"></a>
## 1.5. 调整参数直到SSE参数最小


![2017-09-01-14-24-10](http://qiniu.xdpie.com/2017-09-01-14-24-10.png?imageView2/2/w/700&_=5603928)

新 a = a – r * ∂SSE/∂a = 0.45-0.01*3.300 = 0.42

新 b = b – r * ∂SSE/∂b= 0.75-0.01*1.545 = 0.73

（r是学习率，表示调整的步长）

```Python

# construct an optimizer to minimize cost and fit line to mydata
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

```
然后再重复上一步骤计算，直到所设定的次数完成

![2017-09-01-14-26-44](http://qiniu.xdpie.com/2017-09-01-14-26-44.png?imageView2/2/w/700&_=5603928)

```Python
for i in range(500):
    for (x, y) in zip(trX, trY):
        output = sess.run(train_op, feed_dict={X: x, Y: y})

```

通过刚才几步的组合，程序便能计算出最合适的a,b的值，完成代码清单如下：

```Python
import tensorflow as tf
import numpy as np

sess = tf.Session()


# 线性模型 y=bx+a
def model(x, b, a):
    return tf.multiply(x, b) + a


# 归一化函数
def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_out = []
    for item in arr:
        out = np.divide(np.subtract(item, arr_min), np.subtract(arr_max, arr_min))
        arr_out = np.append(arr_out, np.array(out))
    return arr_out

# 原始数据
trX_i = [1100., 1400., 1425., 1550., 1600., 1700., 1700., 1875., 2350., 2450.]
trY_i = [199000., 245000., 319000., 240000., 312000., 279000., 310000., 308000., 405000., 324000.]

trX = normalize(trX_i)
trY = normalize(trY_i)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 设一个权重变量b，和一个偏差变量a
b = tf.Variable(0.0, name="weights")
# create a variable for biases
a = tf.Variable(0.0, name="biases")
y_model = model(X, b, a)

# 损失函数
loss = tf.multiply(tf.square(Y - y_model), 0.5)

# 梯度下降
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# 训练数据
for i in range(500):
    for (x, y) in zip(trX, trY):
        output = sess.run(train_op, feed_dict={X: x, Y: y})

print('b:' + str(sess.run(b)) + ' || a:' + str(sess.run(a)))


---result

b:0.682465 || a:0.1512

```

<a id="markdown-16-概念" name="16-概念"></a>
## 1.6. 概念
<a id="markdown-161-简单线性回归" name="161-简单线性回归"></a>
### 1.6.1 简单线性回归
在房价预测例子中，我们发现房价数据呈一种比较明显的线性关系，那么自然我们可能会选择简单线性回归对数据进行拟合，首先从线性模型着手：
$$y_p=ax+b$$
从上面的二元一次方程看出，我们的输入**x**是已知向量，只要我们求出a，b的值，就能通过上述公式进行房价预测了，这就是简单线性回归的思想。
<a id="markdown-162-梯度下降" name="162-梯度下降"></a>
### 1.6.2 梯度下降
<a id="markdown-梯度" name="梯度"></a>
#### 梯度
如上一节中讲的我们需要找出SSE最小化时的a，b的值，采用的这种方法就叫做梯度下降。梯度下降不仅仅局限于最小化这个函数，也可能根据实际情况需要最大化某个函数，这种情况叫做梯度上升。单纯从数学上讲，对一个函数来说，梯度表示某个向量的偏导数，同时还代表了该向量的方向，在这个方向上，函数增加得最快，在相反的方向上，函数减小得最快。
利用梯度这一性质，我们采用梯度下降算法去最小化我们的损失函数，我们在梯度的反方向跨域一小步，再从一个新起点开始重复这个过程，直到我们找到损失函数的最小值，最后确定我们的a, b值。
我们需要最小化的函数为（又称为损失函数）：
$$sse=\frac{1}{2}\sum_{k=1}^{n} \ {(\bold{y} -\bold{y_p})^2}=\frac{1}{2}\sum_{k=1}^{n} \ {(y_k-ax_k-b)^2}$$
对a，b分别求偏导，并令偏导等于0：
$$\frac{\partial sse}{\partial a}=- \sum_{k=1}^n \ x_k(y_k-ax_k-b) \ =0$$ 
$$\frac{\partial sse}{\partial b}=- \sum_{k=1}^n \ (y_k-ax_k-b) =0$$
最后，输入已知的**x**和**y**值（均为向量），解两个一次方程就计算出a,b的确切值。

<a id="markdown-步长" name="步长"></a>
#### 步长
为了求SSE的最小值，我们需要向梯度相反的方法移动，每移动一步，梯度逐渐降低，但是移动多少才合适呢，这需要我们谨慎的选择步长。目前，主流的选择方法有：
• 使用固定步长
• 随时间增长逐步减小步长
• 在每一步中通过最小化目标函数的值来选择合适的步长
在上一例子中，我们选择固定步长r=0.01，其实，最后一种方法很好，但它的计算代价很大。我们可以尝试一系列步长，并选出使目标函数值最小的那个步长来求其近似值。
```stepSizes=[10, 1, 0.1, 0.01, 0.001]```
### 1.6.3 损失函数
损失函数是用来评价模型的预测值与真实值的不一致程度，它是一个非负实值函数。通常使用L(Y,f(x))来表示，损失函数越小，模型的性能就越好。
在预测房价的例子中，我们使用了和方差来计算误差，并把该函数称为损失函数，即计算实际值和预测值的误差平方和。为什么要选择这一函数来计算误差，而不采用绝对误差（$\sum_{k=1}^{n} \ {|\bold{y} -\bold{y_p}|}$），或误差的三次方，四次方来定义误差函数是因为：
1. 相对于绝对误差，误差平方和计算更加方便。
2. 这里的损失函数使用的是“最小二乘法”的思想，假定我们的误差满足均值为0的高斯分布，这样符合一般的统计规律，然后根据最大似然函数估计进行推导，就得出了求导结果，平方和最小公式：
$$sse=\frac{1}{2}\sum_{k=1}^{n} \ {(\bold{y} -\bold{y_p})^2}$$

除上面提到的损失函数外，还有其他的一些常见的损失函数：
##### 0-1 Loss
如果预测值与标值不等，则记为1；如果相等，则标记为0
##### Log对数损失函数
在逻辑回归中损失函数的推导是假设样本服从伯努利分布（0-1分布），然后求满足该分布的似然函数，最后推导出顺势函数的公式为：$L(Y,P(Y|X)) = -logP(Y|X)$$
##### 指数损失函数
出现在Adaboost算法中
$$L(y,f(x))=\frac{1}{n}\sum_{i=1}^{n}\ {exp[-y_if(x_i)]}$$
##### Hinge损失函数
在线性支持向量机中，Hinge的损失函数标准形式为：
$$L(y)=\frac{1}{n}\sum_{i=1}^{n}\ {l(wx_i+by_i)}$$
##### 绝对值损失函数
L(y，f(x))=|Y-f(x)|

### 1.6.4 特征归一化
对于多属性的样本，我们在做分类或预测的时候，应该把每个属性看作同等重要，不能让某个属性的计算结果严重影响模型的预测结果。例如，以下有一个样本数据：
| 玩游戏所耗时间百分比    | 描述每年获得的飞行常客里程数 | 每周消费的冰淇淋公升数 |
| ----------------------|------------|------------|
| 0.8             |400       |0.5|
| 12              |134000    |0.9|
| 0               |20000     |1.1|
| 67              |32000     |0.1|
如果我们采用KNN算法做分类预测，在计算欧式距离的时候，比如计算样本3和样本4之间的距离，很明显我们发现每年获得的飞行常客里程数由于本身数值很大，其计算结果的影响将远远大于其他两个特征值的影响，对于三个等权重的特征之一，我们不能让它严重的影响计算结果，所以，我们通常会采用特征归一化的方法把值处理为0到1或者-1到1之间。
$$\sqrt{(0-67)^2+(20000-32000)^2+(1.1-0.1)^2}$$
即上面提到的公式：
$$\frac{x-x_{min}}{x_{max}-x_{min}}$$
其中$x_{min}$和$x_{max}$是特征向量**x**的最小值和最大值，这样通过对每个特征向量进行归一化处理，所有特征值的计算都统一了，而计算得到的结果就更加准确。
在之前预测房价的例子中，我们对仅有的一个特征向量，即房屋大小做了归一化处理，即便是只有一个特征向量，我们仍然对其做归一化处理，其目的与上面的样本数据一样，假设该房屋预测中又增加了房间数量或房屋年龄等特征，我们都可以采用同一类方法进行处理，以减少各特征值对计算结果的影响。

---
参考链接

【1】：http://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html




