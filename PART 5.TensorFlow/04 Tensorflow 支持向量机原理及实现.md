
<!-- TOC -->

- [1.1. SVM介绍](#11-svm介绍)
- [1.2. 工作原理](#12-工作原理)
    - [1.2.1. 几何间隔和函数间隔](#121-几何间隔和函数间隔)
    - [1.2.2. 最大化间隔](#122-最大化间隔)
                - [1.2.2.0.0.1. $L(\bold{x}^*)$对$\bold{x}^*$求导为0](#122001-l\boldx^对\boldx^求导为0)
                - [1.2.2.0.0.2. $\alpha_{_i} g_{_i}(\bold{x}^*)=0$，对于所有的$i=1,.....,n$](#122002-\alpha__i-g__i\boldx^0对于所有的i1n)
- [1.3. 软间隔](#13-软间隔)
- [1.4. SMO算法](#14-smo算法)
- [1.5. 核函数](#15-核函数)
- [1.6. 实例](#16-实例)

<!-- /TOC -->

<a id="markdown-11-svm介绍" name="11-svm介绍"></a>
## 1.1. SVM介绍
SVM（Support Vector Machines）——支持向量机是在所有知名的数据挖掘算法中最健壮，最准确的方法之一，它属于二分类算法，可以支持线性和非线性的分类。发展到今天，SVM已经可以支持多分类了，但在这一章里，我们着重讲支持向量机在二分类问题中的工作原理。
假设在一个二维线性可分的数据集中，图一A所示，我们要找到一个超平面把两组数据分开，这时，我们认为线性回归的直线或逻辑回归的直线也能够做这个分类，这条直线可以是图一B中的直线，也可以是图一C中的直线，或者图一D中的直线，但哪条直线才最好呢，也就是说哪条直线能够达到最好的泛化能力呢？那就是一个能使两类之间的空间大小最大的一个超平面。
这个超平面在二维平面上看到的就是一条直线，在三维空间中就是一个平面...，因此，我们把这个划分数据的决策边界统称为超平面。离这个超平面最近的点就叫做支持向量，点到超平面的距离叫间隔。支持向量机就是要使超平面和支持向量之间的间隔尽可能的大，这样超平面才可以将两类样本准确的分开，而保证间隔尽可能的大就是保证我们的分类器误差尽可能的小，尽可能的健壮。
![2017-09-07-13-58-24](http://qiniu.xdpie.com/2017-09-07-13-58-24.png)
图一


<a id="markdown-12-工作原理" name="12-工作原理"></a>
## 1.2. 工作原理
<a id="markdown-121-几何间隔和函数间隔" name="121-几何间隔和函数间隔"></a>
### 1.2.1. 几何间隔和函数间隔
在最大化支持向量到超平面距离前，我们首先要定义我们的超平面$h({\bold{x}})$（称为超平面的判别函数，也称给定$\bold{w}$和$\bold{b}$的泛函间隔），其中$\bold{w}$为权重向量，$\bold{b}$为偏移向量：
$$h({\bold{x}})=\bold{w}^T\bold{x}+\bold{b}$$
样本$\bold{x}$到最优超平面的**几何间隔**为：
$$r=\frac{h(\bold{x})}{||\bold{w}||}=\frac{\bold{w}^T\bold{x}+\bold{b}}{||\bold{w}||}$$
$||\bold{w}||$是向量$\bold{w}$的内积，是个常数，即$||\bold{w}||=\sqrt{{w_{_0}}^2+{w_{_1}}^2+...+{w_{_n}}^2}$，而$h({\bold{x}})$就是下面要介绍的函数间隔。

**函数间隔**：
$$\widehat{r}=h({\bold{x}})$$
函数间隔$h(\bold{x})$它是一个并不标准的间隔度量，是人为定义的，它不适合用来做最大化的间隔值，因为，一旦超平面固定以后，如果我们人为的放大或缩小$\bold{w}$和$\bold{b}$值，那这个超平面也会无限的放大或缩小，这将对分类造成严重影响。而几何间隔是函数间隔除以$||\bold{w}||$，当$\bold{w}$的值无限放大或缩小时，$||\bold{w}||$也会放大或缩小，而整个$r$保持不变，它只随着超平面的变动而变动，不受两个参数的影响。因而，我们用几何间隔来做最大化间隔度量。
<a id="markdown-122-最大化间隔" name="122-最大化间隔"></a>
### 1.2.2. 最大化间隔
在支持向量机中，我们把几何间隔$\bold{r}$作为最大化间隔进行分析，并且采用-1和1作为类别标签，什么采用-1和+1，而不是0和1呢？这是由于-1和+1仅仅相差一个符号，方便数学上的处理。我们可以通过一个统一公式来表示间隔或者数据点到分隔超平面的距离，同时不必担心数据到底是属于-1还是+1类。
我们一步一步的进行分析，首先如下图，在这个$\mathbb{R}^2$空间中，假设我们已经确定了一个超平面，这个超平面的函数关系式应该是$h({\bold{x}})=\bold{w}^T\bold{x}+\bold{b}=0$，这个式子表示我们图中的那条虚线，很明显，这个式子意思是说点x在超平面上，但我们要想使所有的点都尽可能的远离这个超平面，我们只要保证离这个超平面最近的点远离这个超平面，也就是说这些叫支持向量的点$x^*$需要尽可能的远离它。

![2017-09-08-10-26-04](http://qiniu.xdpie.com/2017-09-08-10-26-04.png)

我们把其中一个支持向量$x^*$到最优超平面的距离定义为：
$$r^*={\frac{h(\bold{x}^*)}{||\bold{w}||}}= \left\{\begin{matrix}
{\frac{1}{||\bold{w}||}} & if:y^*=h(\bold{x}^*)=+1 \\ 
& \\ 
{-\frac{1}{||\bold{w}||}} & if:y^*=h(\bold{x}^*)=-1
\end{matrix}\right.$$

这是我们通过把函数间隔$h(\bold{x})$固定为1而得来的。我们可以把这个式子想象成还存在两个平面，这两个平面分别是$\bold{w}^T\bold{x}_{_s}+\bold{b}=1$和$\bold{w}^T\bold{x}_{_s}+\bold{b}=-1$，对应上图中的两根实线。这些支持向量$\bold{x}_{_s}$就在这两个平面上，这两个平面离最优超平面的距离越大，我们的间隔也就越大。对于其他的点$\bold{x}_{_i}$如果满足$\bold{w}^T\bold{x}_{_i}+\bold{b}>1$，则被分为1类，如果满足满足$\bold{w}^T\bold{x}_{_i}+\bold{b}<-1$，则被分为-1类。即有约束条件：
$$\left\{\begin{matrix}
{\bold{w}^T\bold{x_{_i}}+\bold{b}}\geqslant 1 & y_{_i}=+1 \\ 
& \\ 
{\bold{w}^T\bold{x{_i}}+\bold{b}} \leqslant -1& y_{_i}=-1
\end{matrix}\right.$$

支持向量到超平面的距离知道后，那么分离的间隔$\rho$很明显就为：
$$\rho=2r^*=\frac{2}{||\bold{w}||} $$
这下我们就要通过找到最优的$\bold{w}$和$\bold{b}$来最大化$\rho$了，感觉又像回到了逻辑回归或线性回归的例子。但是这里，我们最大化$\rho$值需要有条件限制，即：
$$\begin{cases}
 & \max \limits_{\bold{w},\bold{b}} {\frac{2}{||\bold{w}||}} \\ 
 &  \\ 
 & \bold{y_{_i}}(\bold{w}^T\bold{x_{_i}}+\bold{b}) \geqslant 1, \ (i=1,..,n)
\end{cases}$$
$\bold{y_{_i}}(\bold{w}^T\bold{x_{_i}}+\bold{b})$的意思是通过判断$\bold{y_{_i}}$和$\bold{w}^T\bold{x_{_i}}+\bold{b}$是否同号来确定分类结果。
接着，为了计算方便，我们把上式最大化$\rho$换成：
$$\begin{cases}
 & \min \limits_{\bold{w},\bold{b}} {\frac{1}{2}}||\bold{w}||^2 \\ 
 &  \\ 
 & \bold{y_{_i}}(\bold{w}^T\bold{x_{_i}}+\bold{b}) \geqslant 1, \ (i=1,..,n)
\end{cases}$$

这种式子通常我们用拉格朗日乘数法来求解，即：
$$L(\bold{x})=f(\bold{x})+\sum\alpha g(\bold{x})$$
$f(\bold{x})$是我们需要最小化的目标函数，$g(\bold{x})$是不等式约束条件，即前面的$\bold{y_{_i}}(\bold{w}^T\bold{x_{_i}}+\bold{b}) \geqslant 1$，$\alpha$是对应的约束系数，也叫拉格朗日乘子。为了使得拉格朗日函数得到最优化解，我们需要加入能使该函数有最优化解法的KKT条件，或者叫最优化条件、充要条件。即假设存在一点$\bold{x}^*$
<a id="markdown-122001-l\boldx^对\boldx^求导为0" name="122001-l\boldx^对\boldx^求导为0"></a>
###### 1.2.2.0.0.1. $L(\bold{x}^*)$对$\bold{x}^*$求导为0
<a id="markdown-122002-\alpha__i-g__i\boldx^0对于所有的i1n" name="122002-\alpha__i-g__i\boldx^0对于所有的i1n"></a>
###### 1.2.2.0.0.2. $\alpha_{_i} g_{_i}(\bold{x}^*)=0$，对于所有的$i=1,.....,n$
这样构建我们的拉格朗日函数为：
$$L(\bold{w},\bold{b},\alpha)=\frac{1}{2}\bold{w}^T\bold{w}-\sum_{i=1}^{n}\alpha_{_i}[y_{_i}(\bold{w}^T\bold{x_{_i}}+\bold{b})-1]$$
以上的KKT条件$\alpha_{_i}[y_{_i}(\bold{w}^T\bold{x_{_i}}+\bold{b})-1]=0$表示，只有距离最优超平面的支持向量$(x_i,y_i)$对应的$\alpha$非零，其他所有点集的$\alpha$等于零。综上所述，引入拉格朗日乘子以后，我们的目标变为：
$$\min_{\bold{w},\bold{b}}\max_{\alpha\geqslant 0 }L(\bold{w},\bold{b},\alpha)$$
即先求得$\alpha$的极大值，再求$\bold{w}$和$\bold{b}$的极小值。可以把该问题转为为等价的凸优化和对偶问题来求解，对于凸优化和对偶问题可以参考《凸优化》这本书，因为该理论可以整整用一本书来介绍了，笔者在这里也只能点到为止了。通过对偶，我们的目标可以又变成：
$$\max_{\alpha\geqslant 0}\min_{\bold{w},\bold{b}}L(\bold{w},\bold{b},\alpha)$$
即先求得$\bold{w}$和$\bold{b}$的极小值，在求$\alpha$的极大值。用$L(\bold{w},\bold{b},\alpha)$对$\bold{w}$和$\bold{b}$分别求偏导，并令其等于0：
$$\begin{cases}
 & \frac{\partial L(\bold{w},\bold{b},\alpha )}{\partial \bold{w}} =0\\ 
 & \ \\ 
 & \frac{\partial L(\bold{w},\bold{b},\alpha )}{\partial \bold{b}}=0
\end{cases}$$
得：
$$\begin{cases}
 & \bold{w}=\sum_{i=1}^{n}\alpha{_i}y_{_i}x_{_i} \\ 
 & \ \\ 
 & \sum_{i=1}^{n}\alpha{_i}y_{_i}=0
\end{cases}$$
把该式代入原来的的拉格朗日式子可得（推导过程省略）：
$$W(\alpha)=\sum_{i=1}^{n}\alpha{_i}-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha{_i}\alpha{_j}y_{_i}y_{_j}{x_{_i}}^Tx_{_j} $$
$$
\sum_{i=1}^{n}\alpha{_i}y_{_i}=0  \ , \alpha_{_i}\geqslant 0 (i=1,...,n)$$
该$W(\alpha)$函数消去了向量$\bold{w}$和向量$\bold{b}$，仅剩$\alpha$这个未知参数，只要我们能够最大化$W(\alpha)$，就能求出对应的$\alpha$，进而求得$\bold{w}$和$\bold{b}$。对于如何求解$\alpha$，SMO算法给出了完美的解决方案，下一节我们详细讲述。这里我们假设通过SMO算法确定了最优$\alpha^*$，则
$$\bold{w}^*=\sum_{i=1}^{n}\alpha{_i}^*y_{_i}x_{_i}$$
最后使用一个正的支持向量$\bold{x}_{_s}$，就可以计算出$\bold{b}$：
$$\bold{b}^*=1-{\bold{w}^*}^T \bold{x}_{_s}$$

<a id="markdown-13-软间隔" name="13-软间隔"></a>
## 1.3. 软间隔
在4.2节中我们推导了如何计算$\bold{w}$、$\bold{b}$和$\bold{\alpha}$，但别忘了以上所有的推导都是在线性可分的条件下进行的，但是现实世界的许多问题并不都是线性可分的，尤其存在许多复杂的非线性可分的情形。如果样本不能被完全线性分开，那么情况就是：间隔为负，原问题的可行域为空，对偶问题的目标函数无限，这讲导致相应的最优化问题不可解。
要解决这些不可分问题，一般有两种方法。第一种是放宽过于严格的间隔，构造软间隔。另一种是运用核函数把这些数据映射到另一个维度空间去解决非线性问题。在本节中，我们首先介绍软间隔优化。
假设两个类有几个数据点混在一起，这些点对最优超平面形成了噪声干扰，软间隔就是要扩展一下我们的目标函数和KKT条件，允许少量这样的噪声存在。具体地说，就要引入松驰变量$\xi_{_i}$来量化分类器的违规行为：
$$\begin{cases}
 & \min \limits_{\bold{w},\bold{b}} {\frac{1}{2}}||\bold{w}||^2 +C\sum_{i=1}^{n} {\xi_{_i}}
 &  \\
 &  \\
 & \bold{y_{_i}}(\bold{w}^T\bold{x_{_i}}+\bold{b}) \geqslant 1-\xi_{_i} & ,\xi_{_i}\geqslant0, & (i=1,..,n)
\end{cases}$$
参数C用来平衡机器的复杂度和不可分数据点的数量，它可被视为一个由用户依据经验或分析选定的“正则化”参数。松驰变量$\xi_{_i}$的一个直接的几何解释是一个错分实例到超平面的距离，这个距离度量的是错分实例相对于理想的可分模式的偏差程度。对上式的化解，可得：
$$W(\alpha)=\sum_{i=1}^{n}\alpha{_i}-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha{_i}\alpha{_j}y_{_i}y_{_j}{x_{_i}}^Tx_{_j} $$

$$\sum_{i=1}^{n}\alpha{_i}y_{_i}=0, 0\leqslant \alpha_{_i}\leqslant C (i=1,...,n)$$
可以看到，松驰变量$\xi_{_i}$没有出现在$W(\alpha)$中，线性可分与不可分的差异体现在约束$\alpha_{_i}\geqslant 0$被替换成了约束$0\leqslant \alpha_{_i}\leqslant C$。但是，这两种情况下求解$\bold{w}$和$\bold{b}$是非常相似的，对于支持向量的定义也都是一致的。
在不可分情况下，对应的KKT条件为：
$$\alpha_{_i}[y_{_i}(\bold{w}^T\bold{x_{_i}}+\bold{b})-1+\xi_{_i}]=0, (i=1,...,n)$$
<a id="markdown-14-smo算法" name="14-smo算法"></a>
## 1.4. SMO算法
1996年， John Platt发布了一个称为SMO的强大算法，用于训练SVM。 SMO表示序列最小优化（Sequential Minimal Optimization）。 Platt的SMO算法是将大优化问题分解为多个小优化问题来求解，这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。
SMO算法的目标是求出一系列$\alpha$，一旦求出了这些$\alpha$，就很容易计算出权重向量$\bold{w}$和$\bold{b}$，并得到分隔超平面。
SMO算法的工作原理是：每次循环中选择两个$\alpha$进行优化处理。一旦找到一对合适的$\alpha$，那么就增大其中一个同时减小另一个。这里所谓的“合适”就是指两个$\alpha$必须要符合一定的条件，条件之一就是这两个$\alpha$必须要在间隔边界之外，而其第二个条件则是这两个$\alpha$还没有进行过区间化处理或者不在边界上。
对SMO具体的分析如下，在4.3节中我们已经得出了

$$W(\alpha)=\sum_{i=1}^{n}\alpha{_i}-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha{_i}\alpha{_j}y_{_i}y_{_j}{x_{_i}}^Tx_{_j} $$

$$\sum_{i=1}^{n}\alpha{_i}y_{_i}=0, 0\leqslant \alpha_{_i}\leqslant C (i=1,...,n)$$

其中$(x_i,y_i)$已知，C可以预先设定，也是已知数，现在就是要最大化$W(\alpha)$，求得参数$\bold{\alpha}=[\alpha_{_1},\alpha_{_2},...,\alpha_{_n}]$。SMO算法是一次选择两个$\alpha$进行优化，那我们就选择$\alpha_{_1}$和$\alpha_{_2}$，然后把其他参数$[\alpha_{_3},\alpha_{_4},...,\alpha_{_n}]$固定，这样$\alpha_{_1}$、$\alpha_{_2}$表示为下面的式子，其中$\zeta$是实数值:
$$\alpha_{_1}y_{_1}+\alpha_{_2}y_{_2}=-\sum_{i=3}^{n}\alpha_{_i}y_{_i}=\zeta$$
然后用$\alpha_{_2}$来表示$\alpha_{_1}$：
$$\alpha_{_1}=(\zeta-\alpha_{_2}y_{_2})y_{_1}$$
把上式带入$W(\bold{\alpha})$中：
$$W(\alpha)=W(\alpha_{_1},\alpha_{_2},...,\alpha_{_n})=W((\zeta-\alpha_{_2}y_{_2})y_{_1},\alpha_{_2},...,\alpha_{_n})$$
省略一系列化解过程后，最后会化解成我们熟悉的一元二次方程，a，b，c均是实数值：
$$W(\alpha_{_2})=a\alpha_{_2}^2+b\alpha_{_2}+c$$
最后对$\alpha_{_2}$求导,解得$\alpha_{_2}$的具体值，我们暂时把这个实数值叫$\alpha_{_2}^*$。而这个$\alpha_{_2}^*$需要满足一个条件$L\leqslant \alpha_{_2}^*\leqslant H$，其中$L$和$H$是什么呢？如下图所示：

![2017-09-13-16-56-00](http://qiniu.xdpie.com/2017-09-13-16-56-00.png)
（图片来自网络）

根据之前的条件$0\leqslant \alpha_{_i}\leqslant C$和等式$\alpha_{_1}y_{_1}+\alpha_{_2}y_{_2}=\zeta$知$\alpha_{_1}$和$\alpha_{_2}$要在矩形区域内，并且在直线上。当$y_{_1}$和$y_{_2}$异号时：
$$\begin{cases}
  L=max(0,\alpha_{_2}-\alpha_{_1}) \\ 
 &\\ 
 H=min(C,C+\alpha_{_2}-\alpha_{_1}) 
\end{cases}$$
当$y_{_1}$和$y_{_2}$同号时：
$$\begin{cases}
 L=max(0,\alpha_{_2}+\alpha_{_1}-C) \\ 
 &\\ 
 H=min(C,\alpha_{_2}+\alpha_{_1}) 
\end{cases}$$
最后，满足条件的$\alpha_{_2}$应该由下面的式子得到，$\alpha_{_2}^{**}$才为最终的值：
$$\alpha_{_2}^{**} =\begin{cases}
H &,   \alpha_{_2}^*> H \\ 
\\
\alpha_{_2}^*& , L\leq \alpha_{_2}^*\leq H \\ 
\\
L & ,\alpha_{_2}^*<L 
\end{cases}$$
求得$\alpha_{_2}^{**}$后我们就可以求得$\alpha_{_1}^{**}$了。然后我们重复地按照最优化$(\alpha_{_1},\alpha_{_2})$的方式继续选择$(\alpha_{_3},\alpha_{_4})$，$(\alpha_{_5},\alpha_{_6})$....$(\alpha_{_{n-1}},\alpha_{_n})$进行优化求解，这样$\bold{\alpha}=[\alpha_{_1},\alpha_{_2},...,\alpha_{_n}]$求解出来后，整个线性划分问题就迎刃而解。
<a id="markdown-15-核函数" name="15-核函数"></a>
## 1.5. 核函数
对于以上几节讲的SVC算法，我们都在线性可分或存在一些噪声点的情况下进行的二分类，但是如果我们存在两组数据，它们的散点图如下图所示，你可以看出这完全是一个非线性不可分的问题，我们无法使用之前讲的SVC算法在这个二维空间中找到一个超平面把这些数据点准确的分开。

![2017-09-13-18-11-01](http://qiniu.xdpie.com/2017-09-13-18-11-01.png)

解决这个划分问题我们需要引入一个核函数，核函数能够恰当的计算给定数据的内积，将数据从输入空间的非线性转变到特征空间，特征空间具有更高甚至无限的维度，从而使得数据在该空间中被转换成线性可分的。如下图所示，我们把二维平面的一组数据，通过核函数映射到了一个三维空间中，这样，我们的超平面就面成了一个平面（在二维空间中是一条直线），这个平面就可以准确的把数据划分开了。

![2017-09-13-18-22-21](http://qiniu.xdpie.com/2017-09-13-18-22-21.png)

核函数有Sigmoid核、线性核、多项式核和高斯核等，其中高斯核和多项式核比较常用，两种核函数均可以把低维数据映射到高维数据。高斯核的公式如下，$\sigma$是达到率，即函数值跌落到0的速度参数：
$$K(\bold{x_1},\bold{x_2})=exp(\frac{-||\bold{x_1}-\bold{x_2}||^2}{2\sigma^2 })$$
多项式核函数的公式如下，$R$为实数，$d$为低维空间的维数：
$$K(\bold{x_1},\bold{x_2})=(\left \langle \bold{x_1}, \bold{x_2}\right \rangle +R)^d$$
应用于我们的上个例子，我们先定义，用$\phi : \bold{x}\to H$表示从输入空间$\bold{x}\subset \mathbb{R}^n$到特征空间H的一个非线性变换。假设在特征空间中的问题是线性可分的，那么对应的最优超平面为：
$$\bold{w}^{\phi T}\phi(\bold{x})+\bold{b}=0$$

通过拉格朗日函数我们推导出:
$$\bold{w}^{\phi *}=\sum_{i=1}^{n}\alpha{_i}^*y_{_i}\phi(\bold{x_{_i}})$$
带入上式得特征空间的最优超平面为：
$$\sum_{i=1}^{n}\alpha{_i}^*y_{_i}\phi^T(\bold{x_{_i}})\phi(\bold{x})+\bold{b}=0$$
这里的$\phi^T(\bold{x_{_i}})\phi(\bold{x})$表示内积，用核函数代替内积则为：
$$\sum_{i=1}^{n}\alpha{_i}^*y_{_i}K(\bold{x_{_i}},\bold{x})+\bold{b}=0$$
这说明，我们的核函数均是内积函数，通过在低维空间对输入向量求内积来映射到高维空间，从而解决在高维空间中数据线性可分的问题，至于具体的推导过程，这里就不再进行了，感兴趣的可以自己再推导一次，加深理解。
为什么核函数可以把低维数据映射成高维数据呢，我们以多项式核来解释一下。
假设有两个输入样本，它们均为二维行向量$\bold{x_1}=[x_1,x_2]$，$\bold{x_2}=[x_3,x_4]$，他们的内积为：
$$\left \langle \bold{x_1},\bold{x_2} \right \rangle=\bold{x_1}\bold{x_2}^T=\begin{bmatrix}
x_1 &x_2 
\end{bmatrix}\begin{bmatrix}
x_3\\x_4 
\end{bmatrix}=x_1x_3+x_2x_4$$
用多项式核函数进行映射，令$R=0$，$d=2$：
$$ K(\bold{x_1},\bold{x_2})=(\left \langle \bold{x_1},\bold{x_2} \right \rangle)^2 
=(x_1x_3+x_2x_4 )^2={x_1}^2{x_3}^2+2x_1x_2x_3x_4+{x_2}^2{x_4}^2=\phi(\bold{x_1}) \phi(\bold{x_2})$$
按照线性代数中的标准定义，$\phi(\bold{x_1})$和$\phi(\bold{x_2})$为映射后的三维行向量和三维列向量，即：

$$\phi(\bold{x_1})=\begin{bmatrix}
{x_1}^2 & \sqrt{2}{x_1}{x_2} & {x_2}^2
\end{bmatrix}$$
$$\phi(\bold{x_2})=\begin{bmatrix}
{x_3}^2\\ \\ \sqrt{2}{x_3}{x_4}\\
\\ {x_4}^2
\end{bmatrix}$$
它们的内积用向量的方式表示则更直观：
$$\phi(\bold{x_1})\phi(\bold{x_2})=\begin{bmatrix}
{x_1}^2 & \sqrt{2}{x_1}{x_2} & {x_2}^2
\end{bmatrix}\begin{bmatrix}
{x_3}^2\\ \\ \sqrt{2}{x_3}{x_4}\\
\\ {x_4}^2
\end{bmatrix}={x_1}^2{x_3}^2+2x_1x_2x_3x_4+{x_2}^2{x_4}^2$$
这样我们就把二维数据映射成了三维数据，对于高斯核的映射，会用到泰勒级数展开式，读者可以自行推导一下。
对于核函数我们就暂时介绍到这里。下一节我们开始运用tensorflow来进行实战，由于没有找到线性不可分的数据集，我们的例子中就没有用到核函数来建模，因为我们只找到了一个线性可分的数据集，所以下一节我们仅运用tensorflow来进行线性二分类的分类器构建。
<a id="markdown-16-实例" name="16-实例"></a>
## 1.6. 实例
我们在网上下载了一组鸢尾花数据集，这组数据集有100个样本点，我们用SVM来预测这些鸢尾花数据集中哪些是山鸢尾花，哪些是非山鸢尾花。

* 首先需要加载数据集,加载数据集需要用到sklearn、scipy、mkl库，sklearn直接在Pycharm中安装即可，而另外两个库需要在网上下载安装，下载地址：http://www.lfd.uci.edu/~gohlke/pythonlibs/
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# 加载数据
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
```

* 分离测试集与训练集

``` python

# 分离训练和测试集
train_indices = np.random.choice(len(x_vals),
                                 round(len(x_vals)*0.8),
                                 replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

```
* 定义模型和loss函数

```python

batch_size = 100

# 初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建变量
A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

```

* 开始训练数据

```python
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
```
* 绘制图像

```python
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
best_fit = []

x1_vals = [d[1] for d in x_vals]

for i in x1_vals:
    best_fit.append(slope*i+y_intercept)


# Separate I. setosa
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

```

训练后的结果
![2017-09-14-16-44-38](http://qiniu.xdpie.com/2017-09-14-16-44-38.png)