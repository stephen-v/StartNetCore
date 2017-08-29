# 1. Tensorflow开发环境配置及其基本概念


## 1.1. 安装Tensorflow开发环境
### 1.1.1 安装pycharm
Pycharm目前是机器学习中最普遍，最收欢迎的工具之一，它强大，具有亲和力，正如它的名字一样魅力无穷。Pycharm官网上有专业版和社区版，社区版是免费的，仅做数据科学方面的研究的话社区版也足够开发使用了，Windows系统的下载地址为：https://www.jetbrains.com/pycharm/download/#section=windows， 下载完成后就可以安装了，安装无需做特别的设置，默认安装就可以了。

###1.1.2 安装pytheon3.6
tensorflow需要运行在python3.4及以上版本，在这个问题上我就出错过一次。之前我电脑上的python版本为2.7，一开始我没有注意到这种情况，我就直接在pycharm中打开File > Default Setting > Project Interpreter，查找tensorflow然后点击安装，结果报错了（如图1-1，1-2所示），错误的提示信息也让人摸不着头脑，查阅了一些资料猛的才发现是我的python版本出了问题，于是毫不犹豫的去下载python3.6（目前已更新到3.6.2版本了），下载地址为官网：https://www.python.org/getit/， 注意python3.6版本已经不支持Win32位的系统了，只有Win64位的安装包，下载如图1-3所示的红色框中的安装包。

![2017-08-29-09-31-41](http://qiniu.xdpie.com/2017-08-29-09-31-41.png)
图1-1

![2017-08-29-09-33-07](http://qiniu.xdpie.com/2017-08-29-09-33-07.png)
图1-2

![2017-08-29-09-33-49](http://qiniu.xdpie.com/2017-08-29-09-33-49.png)
图1-3

下载完成后开始安装，在正式安装之前一定要记得勾选“Add Python 3.6 to PATH”，如图1-4所示，意思是把python的安装路径加入到系统的环境变量中。接着可根据自己需要选择默认安装或自定义安装，不管怎样都要记住安装路径，方便后续相关设置。

![2017-08-29-09-38-03](http://qiniu.xdpie.com/2017-08-29-09-38-03.png)
图1-4

![2017-08-29-09-46-30](http://qiniu.xdpie.com/2017-08-29-09-46-30.png)
图1-5

图1-5所示的安装成功后，我们再来验证一下是否真正安装成功。打开cmd，输入py，回车，可以看到出现了python3.6版本的相关信息（图1-6所示），证明安装成功。接着查看一下python的安装路径是否已经加入到了系统的环境变量中，打开控制面板 > 所有控制面板项 > 系统 > 高级系统设置 > 高级 > 环境变量，可以看到python的安装路径已经加入到了系统环境变量中，图1-7所示。如果没有路径信息，可能是安装python3.6之前忘记勾选“Add Python 3.6 to PATH”这一步，这个时候就只能自己手动添加了，把你之前记住的安装路径在新建环境变量里面填写清楚就可以了。

![2017-08-29-09-49-18](http://qiniu.xdpie.com/2017-08-29-09-49-18.png)
图1-6

![2017-08-29-09-54-24](http://qiniu.xdpie.com/2017-08-29-09-54-24.png)
图1-7

###1.1.3 安装Tensorflow
Pycharm，以及Python3.6都安装完毕，接着打开Pycharm，在File > Default Setting > Project Interpreter中点击设置图片按钮，选择Create VirtualEnv，如图1-8所示，表示新建一个虚拟环境。建立虚拟环境的目的是为了方便以后便捷、快速的安装各种库或插件，以及以后的程序执行等都在该虚拟环境下运行。点击“Create VirtualEnv”后跳出新建虚拟环境对话框，图1-9所示，在“Name”处为虚拟环境命名，“Location”是指虚拟环境的安装路径，注意不要与python3.6的安装目录相同，“Base interpreter”处选择python3.6的安装目录下的python.exe文件，设置完成后，Pycharm会为你新建好一个3.6版本的虚拟环境，如图1-10所示。

![2017-08-29-10-03-30](http://qiniu.xdpie.com/2017-08-29-10-03-30.png)
图1-8

![2017-08-29-10-09-04](http://qiniu.xdpie.com/2017-08-29-10-09-04.png)
图1-9

![2017-08-29-10-13-30](http://qiniu.xdpie.com/2017-08-29-10-13-30.png)
图1-10

初始建立好的虚拟环境已经有pip工具了，接着还是按照最初的步骤安装Tensorflow。点击图1-10上绿色的“+”号键，表示新安装一个库或插件，然后在出现的搜索框中搜索tensorflow，找到后点击“Install Package”就好了，图1-11所示，而不需要你亲自码代码 ``` pip install tensorflow```，真的是方面快捷。

![2017-08-29-10-21-53](http://qiniu.xdpie.com/2017-08-29-10-21-53.png)
图1-11

提示安装完成后，我们最后来验证是否真正安装成功，File > New Project，新建一个名字为tensorflow的项目，图1-12，注意“Interpreter”处已经有了刚刚建立的虚拟环境，在该虚拟环境下新建一个项目，并开展相关的数据挖掘工作就是我们以后将要做的事情了。接着在该项目下新建一个test.py的文件，输入``` import tensorflow```，没有报错的话证明Tensorflow安装成功。

![2017-08-29-10-35-59](http://qiniu.xdpie.com/2017-08-29-10-35-59.png)
图1-12

至此，我们的工具，环境都已经安装，配置好了，下面的章节我们将了解Tensorflow的概念和用法，开始我们的数据科学之旅。

## 1.2. Tensorflow基本概念

### 计算图（The Computational Graph）
Tensorflow是一个通过计算图的形式来表述计算的编程系统，计算图也叫数据流图，可以把计算图看做是一种有向图，Tensorflow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。

看如下代码清单：

```Python
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

---result
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

```
可以看到输出的并不是我们所期望的3.0 和 4.0 数字，而是一个`node`,通过`sesson`可计算出`node`的实际值。

```Python
sess = tf.Session()
print(sess.run([node1, node2]))

---result
[3.0, 4.0]
```

我们还可以通过操作符构建各种复杂的`Tesnor Nodes`

```Python
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

---result
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
```

![2017-08-28-16-57-51](http://qiniu.xdpie.com/2017-08-28-16-57-51.png)

上面的计算我们都是建立在静态的数据上，Tensorflow还提供了`palceholder`用于后期输入值的一个占位符

```Python

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

```Python

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

---result
22.5
```

![2017-08-28-17-05-03](http://qiniu.xdpie.com/2017-08-28-17-05-03.png)

在机器学习中，我们希望看到一个可随意输入的模型，就像上面代码清单一样，为让模型可以训练，我们需要能够修改图获得新的输出。`Variables`允许添加可训练参数，如果使用如下代码清单：

```Python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

```

许多的算法依赖于矩阵的操作，Tensorflow给我们提供了非常方便，快捷的矩阵运算。

```Python
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

