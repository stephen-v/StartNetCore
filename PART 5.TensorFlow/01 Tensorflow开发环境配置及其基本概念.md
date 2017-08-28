# 1. Tensorflow开发环境配置及其基本概念


## 1.1. 安装Tensorflow开发环境

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

我们还可以通过操作符构建各位复杂的`Tesnor Nodes`

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

在机器学习中，我们希望看到一个可随意输入的模型，就像上面代码清单一样，为让模型可以训练，我们需要能够修改图获得新的输出。`Variables`允许添加添加可训练参数，如果使用见如下代码清单：

```Python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```

许多的算法依赖于矩阵的操作，在Tensorflow中给我们提供非常方便，快捷的矩阵运算

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

