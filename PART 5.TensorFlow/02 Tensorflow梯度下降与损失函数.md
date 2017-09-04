<a id="markdown-1-tensorflow梯度下降与损失函数" name="1-tensorflow梯度下降与损失函数"></a>
# 1. Tensorflow梯度下降与损失函数

>在本节中将通过一个预测房屋价格的实例来讲解梯度下降和损失函数的原理，以及在tensorflow中如何实现
<!-- TOC -->

- [1. Tensorflow梯度下降与损失函数](#1-tensorflow梯度下降与损失函数)
    - [1.1. 准备工作](#11-准备工作)
    - [1.2. 归一化数据](#12-归一化数据)
    - [1.3. 用随机的值填充a,b并计算误差，误差采用上文所使用SSE(和方差)](#13-用随机的值填充ab并计算误差，误差采用上文所使用sse和方差)
    - [1.4. 计算误差梯度](#14-计算误差梯度)
    - [1.5. 调整参数直到SSE参数最小](#15-调整参数直到sse参数最小)
    - [1.6. 概念](#16-概念)

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

$$\frac{1}{2}\sum_{k=1}^{n} \ {(y-y_p)^2}$$

在拿到原始的数据后，为方便运算，我们将数据进行归一化处理，归一化计算公式如下

$$\frac{x-x_{min}}{x_{min}-x_{min}}$$

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

# create a shared for weight s
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

# train
for i in range(500):
    for (x, y) in zip(trX, trY):
        output = sess.run(train_op, feed_dict={X: x, Y: y})

print('b:' + str(sess.run(b)) + ' || a:' + str(sess.run(a)))


---result

b:0.682465 || a:0.1512

```

<a id="markdown-16-概念" name="16-概念"></a>
## 1.6. 概念

**梯度下降：** 

**损失函数：**

**特征归一化：**

---
参考链接

【1】：http://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html




