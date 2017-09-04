<a id="markdown-tensorflow开发环境配置及其基本概念" name="tensorflow开发环境配置及其基本概念"></a>
# Tensorflow开发环境配置及其基本概念


<!-- TOC -->

- [Tensorflow开发环境配置及其基本概念](#tensorflow开发环境配置及其基本概念)
    - [1.1. 安装Tensorflow开发环境](#11-安装tensorflow开发环境)
        - [1.1.1. 安装pycharm](#111-安装pycharm)
        - [1.1.2. 安装pythe3.6](#112-安装pythe36)
        - [1.1.3. 安装Tensorflow](#113-安装tensorflow)
    - [1.2. Tensorflow基本概念](#12-tensorflow基本概念)
        - [1.2.1. 声明Tensor](#121-声明tensor)
        - [1.2.2. 变量和占位符](#122-变量和占位符)
            - [1.2.2.1. 变量](#1221-变量)
            - [1.2.2.2. 占位符](#1222-占位符)
        - [1.2.3. 计算图（The Computational Graph）](#123-计算图（the-computational-graph）)
        - [1.2.4. 矩阵操作](#124-矩阵操作)
        - [1.2.5. 声明运算符](#125-声明运算符)

<!-- /TOC -->

<a id="markdown-11-安装tensorflow开发环境" name="11-安装tensorflow开发环境"></a>
## 1.1. 安装Tensorflow开发环境
<a id="markdown-111-安装pycharm" name="111-安装pycharm"></a>
### 1.1.1. 安装pycharm
Pycharm目前是机器学习中最普遍，最收欢迎的工具之一，它强大，具有亲和力，正如它的名字一样魅力无穷。Pycharm官网上有专业版和社区版，社区版是免费的，仅做数据科学方面的研究的话社区版也足够开发使用了，Windows系统的下载地址为：https://www.jetbrains.com/pycharm/download/#section=windows， 下载完成后就可以安装了，安装无需做特别的设置，默认安装就可以了。

<a id="markdown-112-安装pythe36" name="112-安装pythe36"></a>
### 1.1.2. 安装pythe3.6
tensorflow需要运行在python3.4及以上版本，在这个问题上我就出错过一次。之前我电脑上的python版本为2.7，一开始我没有注意到这种情况，我就直接在pycharm中打开File > Default Setting > Project Interpreter，查找tensorflow然后点击安装，结果报错了（如图1-1，1-2所示），错误的提示信息也让人摸不着头脑，查阅了一些资料猛的才发现是我的python版本出了问题，于是毫不犹豫的去下载python3.6（目前已更新到3.6.2版本了），下载地址为官网：https://www.python.org/getit/， 注意python3.6版本已经不支持Win32位的系统了，只有Win64位的安装包，下载如图1-3所示的红色框中的安装包。

![2017-08-29-09-31-41](http://qiniu.xdpie.com/2017-08-29-09-31-41.png?imageView2/2/w/700&_=5603928)
图1-1

![2017-08-29-09-33-07](http://qiniu.xdpie.com/2017-08-29-09-33-07.png?imageView2/2/w/700&_=5603928)
图1-2

![2017-08-29-09-33-49](http://qiniu.xdpie.com/2017-08-29-09-33-49.png?imageView2/2/w/700&_=5603928)
图1-3

下载完成后开始安装，在正式安装之前一定要记得勾选“Add Python 3.6 to PATH”，如图1-4所示，意思是把python的安装路径加入到系统的环境变量中。接着可根据自己需要选择默认安装或自定义安装，不管怎样都要记住安装路径，方便后续相关设置。

![2017-08-29-09-38-03](http://qiniu.xdpie.com/2017-08-29-09-38-03.png?imageView2/2/w/700&_=5603928)
图1-4

![2017-08-29-09-46-30](http://qiniu.xdpie.com/2017-08-29-09-46-30.png?imageView2/2/w/700&_=5603928)
图1-5

图1-5所示的安装成功后，我们再来验证一下是否真正安装成功。打开cmd，输入py，回车，可以看到出现了python3.6版本的相关信息（图1-6所示），证明安装成功。接着查看一下python的安装路径是否已经加入到了系统的环境变量中，打开控制面板 > 所有控制面板项 > 系统 > 高级系统设置 > 高级 > 环境变量，可以看到python的安装路径已经加入到了系统环境变量中，图1-7所示。如果没有路径信息，可能是安装python3.6之前忘记勾选“Add Python 3.6 to PATH”这一步，这个时候就只能自己手动添加了，把你之前记住的安装路径在新建环境变量里面填写清楚就可以了。

![2017-08-29-09-49-18](http://qiniu.xdpie.com/2017-08-29-09-49-18.png?imageView2/2/w/700&_=5603928)
图1-6

![2017-08-29-09-54-24](http://qiniu.xdpie.com/2017-08-29-09-54-24.png?imageView2/2/w/700&_=5603928)
图1-7

<a id="markdown-113-安装tensorflow" name="113-安装tensorflow"></a>
### 1.1.3. 安装Tensorflow
Pycharm，以及Python3.6都安装完毕，接着打开Pycharm，在File > Default Setting > Project Interpreter中点击设置图片按钮，选择Create VirtualEnv，如图1-8所示，表示新建一个虚拟环境。建立虚拟环境的目的是为了方便以后便捷、快速的安装各种库或插件，以及以后的程序执行等都在该虚拟环境下运行。点击“Create VirtualEnv”后跳出新建虚拟环境对话框，图1-9所示，在“Name”处为虚拟环境命名，“Location”是指虚拟环境的安装路径，注意不要与python3.6的安装目录相同，“Base interpreter”处选择python3.6的安装目录下的python.exe文件，设置完成后，Pycharm会为你新建好一个3.6版本的虚拟环境，如图1-10所示。

![2017-08-29-10-03-30](http://qiniu.xdpie.com/2017-08-29-10-03-30.png?imageView2/2/w/700&_=5603928)
图1-8

![2017-08-29-10-09-04](http://qiniu.xdpie.com/2017-08-29-10-09-04.png?imageView2/2/w/700&_=5603928)
图1-9

![2017-08-29-10-13-30](http://qiniu.xdpie.com/2017-08-29-10-13-30.png?imageView2/2/w/700&_=5603928)
图1-10

初始建立好的虚拟环境已经有pip工具了，接着还是按照最初的步骤安装Tensorflow。点击图1-10上绿色的“+”号键，表示新安装一个库或插件，然后在出现的搜索框中搜索tensorflow，找到后点击“Install Package”就好了，图1-11所示，而不需要你亲自码代码 ``` pip install tensorflow```，真的是方面快捷。

![2017-08-29-10-21-53](http://qiniu.xdpie.com/2017-08-29-10-21-53.png?imageView2/2/w/700&_=5603928)
图1-11

提示安装完成后，我们最后来验证是否真正安装成功，File > New Project，新建一个名字为tensorflow的项目，图1-12，注意“Interpreter”处已经有了刚刚建立的虚拟环境，在该虚拟环境下新建一个项目，并开展相关的数据挖掘工作就是我们以后将要做的事情了。接着在该项目下新建一个test.py的文件，输入``` import tensorflow```，没有报错的话证明Tensorflow安装成功。

![2017-08-29-10-35-59](http://qiniu.xdpie.com/2017-08-29-10-35-59.png?imageView2/2/w/700&_=5603928)
图1-12

至此，我们的工具，环境都已经安装，配置好了，下面的章节我们将了解Tensorflow的概念和用法，开始我们的数据科学之旅。

<a id="markdown-12-tensorflow基本概念" name="12-tensorflow基本概念"></a>
## 1.2. Tensorflow基本概念

<a id="markdown-121-声明tensor" name="121-声明tensor"></a>
### 1.2.1. 声明Tensor
在Tensorflow中，tensor是数据的核心单元，也可以称作向量或多维向量，一个tensor组成了一个任意维数的数组的初始值模型，而一个tensor的秩（rank）是就是它的维数，这里有一些例子。

```python
3  #秩为0的tensor，是一个标量，shape[]
[1., 2., 3.]  #秩为1的tensor，是一个向量，shape[3]
[1., 2., 3.], [4., 5., 6.]]  #秩为2的tensor,是一个矩阵，shape[2,3]
[[[1., 2., 3.]], [[7., 8., 9.]]]  #秩为3的tensor,shape[2, 1, 3]
```

如何声明一个tensor？请参见如下代码。

```python
import tensorflow as tf

#声明固定tensor
zero_tsr = tf.zeros([2, 3])  #声明一个2行3列矩阵，矩阵元素全为0
filled_tsr = tf.fill([4, 3], 42)  #声明一个4行3列的矩阵，矩阵元素全部为42
constant_tsr = tf.constant([1, 2, 3])  #声明一个常量，行向量[1 2 3]

#声明连续tensor
linear_tsr = tf.range(start=6, limit=15, delta=3)  #[6, 9, 12]

#声明随机tensor
randnorm_tsr = tf.random_normal([2, 3], mean=0.0, stddev=1.0)  #[[ 0.68031377  1.2489816  -1.50686061], [-1.37892687 -1.04466891 -1.21666121]]
```
注意，```tf.constant()```函数可以用来把一个值传播给一个数组，比如通过这样的声明```tf.constant(42, [4, 3])```来模拟```tf.fill([4, 3], 42)```的行为。

<a id="markdown-122-变量和占位符" name="122-变量和占位符"></a>
### 1.2.2. 变量和占位符
<a id="markdown-1221-变量" name="1221-变量"></a>
#### 1.2.2.1. 变量
变量是算法的参数，Tensorflow追踪这些变量并在算法中优化他们。```Variable()```函数用来声明一个变量，并把一个tensor作为输入，同时输出一个变量。使用该函数仅仅是声明了变量，我们还需要初始化变量，以下代码是关于如果声明和初始化一个变量的例子。
```python

my_var = tf.Variable(tf.zeros([2,3]))   
init = tf.global_variables_initializer()
print(sess.run(init))
print(sess.run(my_var))

---result

[[ 0.  0.  0.]
 [ 0.  0.  0.]]

```
这里请注意一个问题，当你调用```tf.constant()```函数的时候，声明的常量就已经被初始化了，它们的值永远不变。而相反，当你调用```tf.Variable()```函数的时候，变量并没有被初始化，为了初始化所有的变量，你必须要运行一次如下代码：
```python

init = tf.global_variables_initializer()
sess.run(init)

#init是初始化所有的全局变量，在没有调用sess.run()之前，变量都没有被初始化。

```
<a id="markdown-1222-占位符" name="1222-占位符"></a>
#### 1.2.2.2. 占位符

占位符是一个对象，你可以对它赋予不同类型和维度的数据，它依赖于计算图的结果，比如一个计算的期望输出。占位符犹如它的名字一样仅仅为要注入到计算图中的数据占据一个位置。声明占位符用```tf.placeholder()```函数，以下是一个例子：
```python
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2,2)
sess.run(y, feed_dict={x: x_vals})

---result

[[ 0.11068643  0.57449234]
 [ 0.26858184  0.83333433]]
```

<a id="markdown-123-计算图the-computational-graph" name="123-计算图the-computational-graph"></a>
### 1.2.3. 计算图（The Computational Graph）
Tensorflow是一个通过计算图的形式来表述计算的编程系统，计算图也叫数据流图，可以把计算图看做是一种有向图，Tensorflow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。Tensorflow在创建Tensor的同时，并没有把任何值都加入到计算图中。

看如下代码清单：

```python
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

---result
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

```
可以看到输出的并不是我们所期望的3.0 和 4.0 数字，而是一个`node`,通过`sesson`可计算出`node`的实际值。

```python
sess = tf.Session()
print(sess.run([node1, node2]))

---result
[3.0, 4.0]
```

我们还可以通过操作符构建各种复杂的`Tesnor Nodes`

```python
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

---result
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
```

![2017-08-28-16-57-51](http://qiniu.xdpie.com/2017-08-28-16-57-51.png?imageView2/2/w/700&_=5603928)

上面的计算我们都是建立在静态的数据上，Tensorflow还提供了`palceholder`用于后期输入值的一个占位符

```python

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

---result

7.5
[ 3.  7.]

```
还可以再上面的计算途中加入更多操作，见下代码清单：

```python

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

---result
22.5
```

![2017-08-28-17-05-03](http://qiniu.xdpie.com/2017-08-28-17-05-03.png?imageView2/2/w/700&_=5603928)



在机器学习中，我们希望看到一个可随意输入的模型，就像上面代码清单一样，为让模型可以训练，我们需要能够修改图获得新的输出。`Variables`允许添加可训练参数，如果使用如下代码清单：

```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

---result

[ 0.          0.30000001  0.60000002  0.90000004]
```

接下来我们将计算图变得更为复杂，代码清单如下：

```python

sess = tf.Session()

my_array = np.array([[1., 3., 5., 7., 9.], [-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))


---result 

[[ 102.]
 [  66.]
 [  58.]]
[[ 114.]
 [  78.]
 [  70.]]


```

接下来我们来看下代码如何运行，placeholder 在运行时送入(feed in)数据,从计算图中可以看到，要执行Add操作，需要首先执行prod1，然后执行prod2,最后才执行Add操作。

![2017-08-29-15-01-44](http://qiniu.xdpie.com/2017-08-29-15-01-44.png?imageView2/2/w/700&_=5603928)

<a id="markdown-124-矩阵操作" name="124-矩阵操作"></a>
### 1.2.4. 矩阵操作
许多的算法依赖于矩阵的操作，Tensorflow给我们提供了非常方便，快捷的矩阵运算。

```python
# 生成对角线矩阵
identity_matrix = tf.diag([1.0, 1.0, 1.0])
# 随机填充2行3列矩阵
A = tf.truncated_normal([2, 3])
# 填充5 到一个2行3列矩阵
B = tf.fill([2, 3], 5.0)
# 填充三行两列的随机数矩阵
C = tf.random_uniform([3, 2])
# 将numpy的矩阵转换
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
sess = tf.Session()
print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(D))

---result

[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]

[[ 0.08475778 -0.81952369 -0.40169609]
 [-0.6406377  -0.67895085 -1.13811123]]

[[ 5.  5.  5.]
 [ 5.  5.  5.]]

[[ 0.30655277  0.81441486]
 [ 0.68046188  0.64171898]
 [ 0.76518583  0.10888731]]
 
[[ 1.  2.  3.]
 [-3. -7. -1.]
 [ 0.  5. -2.]]
```
<a id="markdown-125-声明运算符" name="125-声明运算符"></a>
### 1.2.5. 声明运算符
Tensor具有很多标准的运算符，如```add()```，```sub()```，```mul()```，```div()```等，除了这些标准的运算符外，Tensorflow给我们提供了更多的运算符。以下是一个基础的数学函数列表，所有的这些函数都是按元素操作。

| 运算符      | 描述                             | 
| -----------|-------------                     |
| abs()      |计算一个输入tensor的绝对值          |
| ceil()     |一个输入tensor的顶函数              |
| cos()      |Cosine函数          C               |
| exp()      |底数为e的指数函数，指数为输入的tensor |
| floor()    |一个输入tensor的底函数               |
| inv()      |一个输入tensor的倒数函数，(1/x)      |
| log()      |一个输入tensor的自然对数             |
| maximum()  |取两个tensor中的最大的一个           |
| minimum()  |取两个tensor中的最小的一个           |
| neg()      |对一个tensor取负值                  |
| pow()      |第一个tensor是第二个tensor的幂       |
| round()    |舍入最接近的整数                     |
| rsqrt()    |计算一个tensor的平方根后求倒          |
| sign()     |根据tensor的符号返回-1，0，1中的某个值 |
| sin()      |Sine函数                             |
| sqrt()     |计算一个输入tensor的平方根            |
| square()   |计算一个输入的tensor的平方            |

下面还有一些值得我们了解的函数，这些函数在机器学习中比较常用，Tensorflow已经包装好了。

| 运算符                 | 描述       | 
| ----------------------|------------|
| digamma()             |计算lgamma函数的导数                      |
| erf()                 |计算tensor的高斯误差                      |
| erfc()                |计算tensor的高斯互补误差                  |
| igamma()              |计算gamma(a, x)/gamma(a),gamma(a,x)=\intergral_from_0_to_x t^(a-1) *exp^(-t)dt             |
| igammac()             |计算gamma(a,x)/gamma(a),gamma(a,x)=\intergral_from_x_to_inf t^(a-1) *exp^(-t)dt             |
| lbeta()               |计算自然对数的beta函数的绝对值             |
| lgamma()              |计算自然对数的gamma函数的绝对值            |
| squared_difference()  |计算两个tensor的误差平方                  |

