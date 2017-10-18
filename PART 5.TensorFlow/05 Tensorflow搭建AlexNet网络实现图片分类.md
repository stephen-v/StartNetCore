###**版权声明：本文为博主原创文章，未经博主允许不得转载。**
#1. 图片数据处理
我们都知道，一张图片是由一个个像素组成，每个像素的颜色我们常常用RGB、HSB、CYMK、RGBA等颜色值来表示，每个颜色值的取值范围不一样，但都代表了一个像素点数据信息。我们在对图片的数据处理过程中，常常用RGB数据来对图片进行处理，RGB表示红绿蓝三通道色，取值范围为0~255，所以一个像素点我们可以把它看作是一个三维数组，即：`array([[[0, 255, 255]]])`，三个数值分布表示R、G、B（红、绿、蓝）的颜色值。比如下图一张3*3大小的jpg格式的图片:

![2017-10-08-13-48-58](http://qiniu.xdpie.com/2017-10-08-13-48-58.png)

它的图片经过Tensorflow解码后，数据值输出为
```Python
image_path = 'images/image.jpg'
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_path))
image_reader = tf.WholeFileReader()
_,image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)  # 如果是png格式的图片，使用tf.image.decode_png()
sess.run(image)


--result

array([[[0, 0, 0], [255, 255, 255], [254, 0, 0]],
       [[0, 191, 0], [3, 108, 233], [0, 191, 0]],
       [[254, 0, 0], [255, 255, 255], [0, 0, 0]])
```
图片的数据处理不仅仅就是把RGB值转换成我们运算需要的值，还包括调整图片大小、图片截取、图片翻转、图片色彩调整，标注框、多线程处理图片等等，在这里我们就不一一介绍了，但是对于图片的处理是进行卷积网络的首要任务，我们需要了解，并学会对图片的相关操作。这里，我们只介绍RGB值的转换，为下一节的卷积提供数据支持。
#2. 卷积神经网络
卷积神经网络（CNN）的基本架构通常包括卷积层，池化层，全链层三大层次，其中不同的层中可能还会包括一些非线性变化（RELU函数）、数据归一化处理、dropoout等。我们常听说的LeNet-5、AlexNet、VGG、ResNet等都是卷积神经网络，而且都是由这些层组成，只是每个网络的层数不一样，所达到的分类效果也不一样。
##2.1 卷积层
卷积层是整个神经网络中最重要的一层，该层最核心的部分为过滤器，或者称为卷积核，卷积核有大小和深度两个属性，大小常用的有3X3、5X5，也有11X11的卷积核，而深度通俗一点理解就是卷积核的个数。卷积核的大小和深度均由人工指定，而权重参数则在初始化的时候由程序随机生成，并在后期训练过程中不断优化这些权重值，以达到最好的分类效果。卷积的过程就是用这些权重值不断的去乘这些图片的RGB值，以提取图片数据信息。下面的动图完美地诠释了卷积是怎么发生的：
![2017-10-08-15-05-31](http://www.omegaxy.com/Pictures/CNN/Convolution_schematic.gif)
上面黄色3X3大小不停移动的就是卷积核，绿色部分是5X5的输入矩阵，粉色部分是卷积后的结果，称作特征值。从上面动图看出，卷积不仅提取了图片信息，也可以达到降维效果。如果希望卷积后的特征值维度和原图片一致，需要设置`padding`值（全零填充）为`SAME`（如果为`VALID`表示不填充），其中`i`为输入图片，`k`为卷积核大小，`strides`为移动步长（移动步长>1也可以达到降维的效果）。
```Pythong 
tf.nn.conv2d(i, k,strides,padding='VALID')
```

在卷积层中，过滤器中的参数是共享的，即一个过滤器中的参数值在对所有图片数据进行卷积过程中保持不变，这样卷积层的参数个数就和图片大小无关，它只和过滤器的尺寸，深度，以及当前层节点的矩阵深度有关。比如，以手写图片为例，输入矩阵的维度是28X28X1，假设第一层卷积层使用的过滤器大小为5X5，深度为16，则该卷积层的参数个数为5X5X1X16+16=416个，而如果使用500个隐藏节点的全链层会有1.5百万个参数，相比之下，卷积层的参数个数远远小于全链层，这就是为什么卷积网络广泛用于图片识别上的原因。

对于卷积后的矩阵大小，有一个计算公式，如果使用了全0填充，则卷积后的矩阵大小为：
$$out_{length}=\left \lceil in_{length}/stride_{length} \right \rceil$$

$$out_{width}=\left \lceil in_{width}/stride_{width} \right \rceil$$
即输出矩阵的长等于输入矩阵长度除以长度方向上的步长，并向上取整数值；输出矩阵的宽度等于输入矩阵的宽度除以宽度方向上的步长，并向上取整数值。
如果不使用全0填充，则输出矩阵的大小为：
$$out_{length}=\left \lceil (in_{length}-filter_{length}+1)/stride_{length} \right \rceil$$

$$out_{width}=\left \lceil (in_{width}-filter_{width}+1)/stride_{width} \right \rceil$$
卷积计算完成后，往往会加入一个修正线性单元ReLU函数，也就是把数据非线性化。为什么要把数据进行非线性化呢，这是因为非线性代表了输入和输出的关系是一条曲线而不是直线，曲线能够刻画输入中更为复杂的变化。比如一个输入值大部分时间都很稳定，但有可能会在某个时间点出现极值，但是通过ReLU函数以后，数据变得平滑，这样以便对复杂的数据进行训练。
ReLU是分段线性的，当输入为非负时，输出将与输入相同；而当输入为负时，输出均为0。它的优点在于不受“梯度消失”的影响，且取值范围为[0,+∞]；其缺点在于当使用了较大的学习速率时，易受达到饱和的神经元的影响。

![2017-10-09-15-39-32](http://qiniu.xdpie.com/2017-10-09-15-39-32.png)

##2.2 池化层
卷积层后一般会加入池化层，池化层可以非常有效地缩小矩阵的尺寸，从而减少最后全链层中的参数，使用池化层既可以加快计算速度也有防止过拟合问题的作用。
池化层也存在一个过滤器，但是过滤器对于输入的数据的处理并不是像卷积核对输入数据进行节点的加权和，而只是简单的计算最大值或者平均值。过滤器的大小、是否全0填充、步长等也是由人工指定，而深度跟卷积核深度不一样，卷积层使用过滤器是横跨整个深度的，而池化层使用的过滤器只影响一个深度上的节点，在计算过程中，池化层过滤器不仅要在长和宽两个维度移动，还要在深度这个维度移动。使用最大值操作的池化层被称之为最大池化层，这种池化层使用得最多，使用平均值操作的池化层被称之为平均池化层，这种池化层的使用相对要少一点。
以下动图可以看到最大值池化层的计算过程：

![2017-10-09-15-58-41](http://www.omegaxy.com/Pictures/CNN/pooling.gif)

Tensorflow程序很容易就可以实现最大值池化层的操作：
```Python

pool = tf.nn.max_pool(i, ksize=[1,3,3,1], stride=[1,2,2,1], padding='SAME')

# i为输入矩阵
# ksize为过滤器尺寸，其中第一个和第四个值必须为1，表示过滤器不可以垮不同的输入样列和节点矩阵深度。中间的两个值为尺寸，常使用2*2或3*3。
# stride为步长，第一个值和第四个值与ksize一样
# padding为全0填充，‘SAME’表示使用全0填充，‘VALID’表示不使用全0填充
```
##2.3 全链层

在KNN或线性分类中有对数据进行归一化处理，而在神经网络中，也会做数据归一化的处理，原因和之前的一样，避免数据值大的节点对分类造成影响。归一化的目标在于将输入保持在一个可接受的范围内。例如，将输入归一化到[0.0，1.0]区间内。在卷积神经网络中，对数据归一化的处理我们有可能放在数据正式输入到全链层之前或之后，或其他地方，每个网络都可能不一样。

全链层的作用就是进行正确的图片分类，不同神经网络的全链层层数不同，但作用确是相同的。输入到全链层的神经元个数通过卷积层和池化层的处理后大大的减少了，比如以AlexNet为例，一张227*227大小，颜色通道数为3的图片经过处理后，输入到全链层的神经元个数有4096个,最后softmax的输出，则可以根据实际分类标签数来定。

在全链层中，会使用dropout以随机的去掉一些神经元，这样能够比较有效地防止神经网络的过拟合。相对于一般如线性模型使用正则的方法来防止模型过拟合，而在神经网络中Dropout通过修改神经网络本身结构来实现。对于某一层神经元，通过定义的概率来随机删除一些神经元，同时保持输入层与输出层神经元的个人不变，然后按照神经网络的学习方法进行参数更新，下一次迭代中，重新随机删除一些神经元，直至训练结束。
![2017-10-09-16-57-39](http://qiniu.xdpie.com/2017-10-09-16-57-39.png)

#3. AlexNet
AlexNet是2012年ILSVRC比赛的冠军，它的出现直接打破了沉寂多年的图片识别领域（在1998年出现LeNet-5网络一直占据图片识别的领头地位），给该领域带来了新的契机，并一步步发展至今，甚至打败了人类的识别精确度，可惜的是2017年的ILSVRC举办方宣布从2018年起将取消该比赛，因为目前的神经网络精确度已经达到跟高的程度了。但深度学习的步伐不会停止，人们将在其他方面进行深入的研究。

AlexNet是神经网络之父Hinton的学生Alex Krizhevsky开发完成，它总共有8层，其中有5个卷积层，3个全链层，附上最经典的AlexNet网络架构图，如下。Alex在他的论文中写到，他在处理图片的时候使用了两个GPU进行计算，因此，从图中看出，在卷积过程中他做了分组的处理，但是由于硬件资源问题，我们做的Alex网络是使用一个CPU进行计算的，但原理和他的一样，只是计算速度慢一点而已，对于大多数没有性能优良的GPU的人来说，用我们搭建好的网络，完全可以使用家用台式机进行训练。
![2017-10-09-17-25-04](http://qiniu.xdpie.com/2017-10-09-17-25-04.png)

Alex在论文中写到他使用的输入图片大小为224 X 224 X 3，但我们使用的图片尺寸为227 X 227 X 3，这个没有太大影响。AlexNet网络分为8层结构，前5层其实不完全是卷积层，有些层还加入了池化层，并对数据进行标准化处理。下面简要介绍一下每一层：

**第一层**

| 卷积核     |深度 |步长        |
|---------------|-------------|---|
| 11 * 11    |96|4 * 4          |

| 池化层过滤器     |步长        |
|---------------|-------------|---|
| 3 * 3    |2 * 2          |

第一层包含了卷积层、标准化操作和池化层，其中卷积层和池化层的参数在上表已给出。在Tensorflow中，搭建的部分代码程序为：
```Python
# 1st Layer: Conv (w ReLu) -> Lrn -> Pool
conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
```

**第二层**

| 卷积核     |深度 |步长        |
|---------------|-------------|---|
| 5 * 5    |256|1 * 1          |

| 池化层过滤器     |步长        |
|---------------|-------------|---|
| 3 * 3    |2 * 2          |
第二层实际也包含了卷积层、标准化操作和池化层，其中卷积层和池化层的参数在上表已给出。在Tensorflow中，搭建的部分代码程序为：
```Python
# 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
```

**第三层**
| 卷积核     |深度 |步长        |
|---------------|-------------|---|
| 3 * 3    |384|1 * 1          |
第三层仅有一个卷积层，卷积核的相关信息如上表，在Tensorflow中的部分代码为：
```Python
# 3rd Layer: Conv (w ReLu)
conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
```

**第四层**
| 卷积核     |深度 |步长        |
|---------------|-------------|---|
| 3 * 3    |384|1 * 1          |
第四层仅有一个卷积层，卷积核的相关信息如上表，该层与第三层很相似，只是把数据分成了2组进行处理，在Tensorflow中的部分代码为：
```Python
# 4th Layer: Conv (w ReLu) splitted into two groups
conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
```

**第五层**
| 卷积核     |深度 |步长        |
|---------------|-------------|---|
| 3 * 3    |256|1 * 1          |

| 池化层过滤器     |步长        |
|---------------|-------------|---|
| 3 * 3    |2 * 2          |
第五层是最后一层卷积层，包含一个卷积层和一个池化层，卷积核和池化层过滤器的相关信息如上表，该层仍然把数据分成了2组进行处理，在Tensorflow中的部分代码为：
```Python
# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
```

**第六层**

第六层是全链层，卷积层输出的数据一共有4096个神经元，在进入第六层全链层后，首先做了数据的平滑处理，并随机删除了一些神经元，在Tensorflow中的部分代码为：
```Python
# 6th Layer: Flatten -> FC (w ReLu) -> Dropout
flattened = tf.reshape(pool5, [-1, 6*6*256])
fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
dropout6 = dropout(fc6, self.KEEP_PROB)
```

**第七层**

第七层是全链层，也会做dropout处理，在Tensorflow中的部分代码为：
```Python
# 7th Layer: FC (w ReLu) -> Dropout
fc7 = fc(dropout6, 4096, 4096, name='fc7')
dropout7 = dropout(fc7, self.KEEP_PROB)
```

**第八层**

第八层是全链层，在最后softmax函数输出的分类标签是根据实际分类情况来定义的，可能有2种，可能10种，可能120种等等，在Tensorflow中的部分代码为：
```Python
# 8th Layer: FC and return unscaled activations
self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
```
#4. 用Tensorflow搭建完整的AlexNet
在搭建完整的AlexNet之前，我们可能需要做一些准备工作，以方便我们在后期做训练的时候观测网络的运行情况。首先就是配置Tensorboard，Tensorboard是一款可视化工具，可以用它来展现你的TensorFlow图像，绘制图像生成的定量指标图，观察loss函数的收敛情况，网络的精确度，以及附加数据等等，具体如何配置，网上也有很多讲解，这里就不详细讲述了；另外就是准备数据，imageNet官网上有很多图片数据可以供大家免费使用，官网地址：http://image-net.org/download-images 。网上还有很多免费使用的爬虫可以去爬取数据，总之，数据是训练的根本，在网络搭建好之前最好准备充分。准备好的数据放入当前训练项目的根目录下。

为了让各种需求的人能够复用AlexNet，我们在Python类中定义了AlexNet，并把接口暴露出来，需要使用的人根据自己的情况调用网络，并输入数据以及分类标签个数等信息就可以开始训练数据了。要使用搭建好的网络进行训练，不仅仅要利用网络，更是需要网络中的各项权重参数和偏置来达到更好的分类效果，目前，我们使用的是别人已经训练好的参数，所有的参数数据存放在bvlc_alexnet.npy这个文件中，下载地址为：http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ ，下载后放入当前训练项目的根目录下即可。如果你有充分的时间和优越的硬件资源，你也可以自己训练参数，并把这些参数存储起来供以后使用，但是该bvlc_alexnet.npy文件中的参数是imageNet训练好了的，使用这些参数训练的模型精确度比我们之前训练的要高。
在Tensorflow中，定义加载参数的程序代码如下，默认的参数就是bvlc_alexnet.npy中存储的权重和偏置值。
```Python
def load_initial_weights(self, session):

    """Load weights from file into network."""

    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if layer should be trained from scratch
        if op_name not in self.SKIP_LAYER:

            with tf.variable_scope(op_name, reuse=True):

                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))
```
在上一节讲述AlexNet的架构的时，曾出现过数据分组处理，这里用程序来描述一下在一个CPU情况下，如何把数据进行分组处理。数据的分组处理都在卷积层中发生，因此首先一个卷积函数，由于在第一层卷积没有分组，所以在函数中需要做分组的判断，如果没有分组，输入数据和权重直接做卷积运算；如果有分组，则把输入数据和权重先划分后做卷积运算，卷积结束后再用`concat()`合并起来，这就是分组的具体操作。
```Python
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,padding='SAME', groups=1):
    """Create a convolution layer."""

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu

```
对于AlexNet中池化层，全链层的代码在alexnet.py已经全部定义好了，这里就不一一列出来了。接着开始如何在Tensorflow中导入图片，在图片数据量大的情况下，Tensorflow会建议把数据转换成tfrecords文件，然后在导入到网络中运算，这样的好处是可以加快计算速度，节约内存空间。但我们没有这样做，因为在训练网络的时候我们没有发现转换成tfrecords文件就明显提高了计算速度，所以这里直接把原生的图片直接转化成三维数据输入到网络中。这样做代码还要简短一点，而图片也是预先存储在硬盘中，需要训练的那一部分再从硬盘中读取到内存中，并没有浪费内存资源。

在Python类中定义图片生成器，需要的参数有图片URL，实际的标签向量和标签个数，batch_size等。首先打乱整个训练集图片的顺序，因为图片名可能是按照某种规律来定义的，打乱图片顺序可以帮助我们更好的训练网络。完成这一步后就可以把图片从RGB色转换成BRG三维数组。
```Python

class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes, shuffle=True, buffer_size=1000):

        self.img_paths = images
        self.labels = labels
        self.num_classes = num_classes
        self.data_size = len(self.labels)
        self.pointer = 0

        # 打乱图片顺序
        if shuffle:
            self._shuffle_lists()
        
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self._parse_function_train, num_threads=8,
                        output_buffer_size=100 * batch_size)

        data = data.batch(batch_size)

        self.data = data

    
    """打乱图片顺序"""
    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])


    """把图片生成三维数组，以及把标签转化为向量"""
    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)
        img_bgr = img_centered[:, :, ::-1]
        
        return img_bgr, one_hot

```

网络搭建完成，数据准备就绪，最后就是开始训练了。由于网络和图片生成器是可以复用的，在训练图片的时候需要用户根据自己的实际情况编写代码调用网络和图片生成器模块，同时定义好损失函数和优化器，以及需要在Tensorboard中观测的各项指标等等操作。下面一节我们将开始进行网络训练。
#5. 用AlexNet识别猫狗图片
##5.1 定义分类
如上一节讲的，datagenerator.py（图片转换模块）和alexnet.py（AlexNet网络模块）已经搭建好了，你在使用的时候无需做修改。现在你只需要根据自己的分类需求编写精调代码，如finetune.py中所示。
假设有3万张猫狗图片训练集和3000张测试集，它们大小不一。我们的目的是使用AlexNet正确的分类猫和狗两种动物，因此，类别标签个数只有2个，并用0代表猫，1代表狗。如果你需要分类其他的动物或者物品，或者anything，你需要标注好图片的实际标签，定义好图片Tensorboard存放的目录，以及训练好的模型和参数的存放目录等等。就像这样：
```Python
import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.contrib.data import Iterator

learning_rate = 1e-4                   # 学习率
num_epochs = 100                       # 代的个数
batch_size = 1024                      # 一次性处理的图片张数
dropout_rate = 0.5                     # dropout的概率
num_classes = 2                        # 类别标签
train_layers = ['fc8', 'fc7', 'fc6']   # 训练层，即三个全链层
display_step = 20                      # 显示间隔次数

filewriter_path = "./tmp/tensorboard"  # 存储tensorboard文件
checkpoint_path = "./tmp/checkpoints"  # 训练好的模型和参数存放目录

if not os.path.isdir(checkpoint_path): #如果没有存放模型的目录，程序自动生成
    os.mkdir(checkpoint_path)

```
接着调用图片生成器，来生成图片数据，并初始化数据：
```Python

train_image_path = 'train/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）
test_image_cat_path = 'test/cat/'  # 指定测试集数据路径（根据实际情况指定测试数据集的路径）
test_image_dog_path = 'test/dog/'

# 打开训练数据集目录，读取全部图片，生成图片路径列表
image_filenames_cat = np.array(glob.glob(train_image_path + 'cat.*.jpg'))
image_filenames_dog = np.array(glob.glob(train_image_path + 'dog.*.jpg'))

# 打开测试数据集目录，读取全部图片，生成图片路径列表
test_image_filenames_cat = np.array(glob.glob(test_image_cat_path + '*.jpg'))
test_image_filenames_dog = np.array(glob.glob(test_image_dog_path + '*.jpg'))

image_path = []
label_path = []
test_image = []
test_label = []

# 遍历训练集图片URL，并把图片对应的实际标签和路径分别存入两个新列表中
for catitem in image_filenames_cat:
    image_path.append(catitem)
    label_path.append(0)
for dogitem in image_filenames_dog:
    image_path.append(dogitem)
    label_path.append(1)

# 遍历测试集图片URL，并把图片路径存入一个新列表中
for catitem in test_image_filenames_cat:
    test_image.append(catitem)
    test_label.append(0)

for dogitem in test_image_filenames_cat:
    test_image.append(dogitem)
    test_label.append(1)

# 调用图片生成器，把训练集图片转换成三维数组
tr_data = ImageDataGenerator(
    images=image_path,
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)

# 调用图片生成器，把测试集图片转换成三维数组
test_data = ImageDataGenerator(
    images=test_image,
    labels=test_label,
    batch_size=batch_size,
    num_classes=num_classes,
    shuffle=False)

# 定义迭代器
iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)
# 定义每次迭代的数据
next_batch = iterator.get_next()

# 初始化数据
training_initalize = iterator.make_initializer(tr_data.data)
testing_initalize = iterator.make_initializer(test_data.data)

```
训练数据准备好以后，让数据通过AlexNet。
```Python

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout概率


# 图片数据通过AlexNet网络处理
model = AlexNet(x, keep_prob, num_classes, train_layers)

# 定义我们需要训练的全连层的变量列表
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


# 执行整个网络图
score = model.fc8

```
接着当然就是定义损失函数，优化器。整个网络需要优化三层全链层的参数，同时在优化参数过程中，使用的是梯度下降算法，而不是反向传播算法。
```Python
# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# 定义需要精调的每一层的梯度
gradients = tf.gradients(loss, var_list)
gradients = list(zip(gradients, var_list))

# 优化器，采用梯度下降算法进行优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# 需要精调的每一层都采用梯度下降算法优化参数
train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# 定义网络精确度
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 以下几步是需要在Tensorboard中观测loss的收敛情况和网络的精确度而定义的
tf.summary.scalar('cross_entropy', loss)
tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

```
最后，训练数据：
```Python

# 定义一代的迭代次数
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 把模型图加入Tensorboard
    writer.add_graph(sess.graph)

    # 把训练好的权重加入未训练的网络中
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # 总共训练100代
    for epoch in range(num_epochs):
        sess.run(iterator.make_initializer(tr_data.data))
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))

        # 开始训练每一代，一代的次数为train_batches_per_epoch的值
        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(optimizer, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

```
训练完成后需要验证模型的精确度，这个时候就得用上测试数据集了。

```Python

# 测试模型精确度
print("{} Start validation".format(datetime.now()))
sess.run(testing_initalize)
test_acc = 0.
test_count = 0

for _ in range(test_batches_per_epoch):
    img_batch, label_batch = sess.run(next_batch)
    acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
    test_acc += acc
    test_count += 1

test_acc /= test_count

print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

```

最后把训练好的模型持久化。
```Python
# 把训练好的模型存储起来
print("{} Saving checkpoint of model...".format(datetime.now()))

checkpoint_name = os.path.join(checkpoint_path,'model_epoch' + str(epoch + 1) + '.ckpt')
save_path = saver.save(sess, checkpoint_name)

print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))

```
到此为止，一个完整的AlexNet就搭建完成了。在准备好训练集和测试集数据后，下面我们开始训练网络。
## 5.2 训练网络
我们总共训练了100代，使用CPU计算进行计算，在台式机上跑了一天左右，完成了3万张图片的训练和3000张图片的测试，网络的识别精确度为71.25%，这个结果不是很好，可能与数据量少有关系。如果你有上十万张的数据集，再增加训练次数，相信你网络的精度应该比我们训练的还要好。下面看看网络的计算图，这是Tensorboard中记录下的，通过该图，你可以对整个网络的架构及运行一目了然。

![2017-10-16-13-54-41](http://qiniu.xdpie.com/2017-10-16-13-54-41.png)

##5.3 验证
网络训练好了以后，当然我们想迫不及待的试试我们网络。首先我们还是得编写自己的验证代码：
```Python
import tensorflow as tf
from alexnet import AlexNet             # import训练好的网络
import matplotlib.pyplot as plt

class_name = ['cat', 'dog']             # 自定义猫狗标签


def test_image(path_image, num_class, weights_path='Default'):
    # 把新图片进行转换
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized = tf.reshape(img_resized, shape=[1, 227, 227, 3])
    
    # 图片通过AlexNet
    model = AlexNet(img_resized, 0.5, 2, skip_layer='', weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./tmp/checkpoints/model_epoch10.ckpt") # 导入训练好的参数
        # score = model.fc8
        print(sess.run(model.fc8))
        prob = sess.run(max)[0]

        # 在matplotlib中观测分类结果
        plt.imshow(img_decoded.eval())
        plt.title("Class:" + class_name[prob])
        plt.show()


test_image('./test/20.jpg', num_class=2) # 输入一张新图片
```

在网上任意下载10张猫狗图片来进行验证，有三张图片识别错误（如下图），验证的精确度70%，效果不是很理想。但是如果你感兴趣，你可以下载我们的代码，用自己的训练集来试试，代码地址为：https://github.com/stephen-v/tensorflow_alexnet_classify 

![2017-10-18-10-16-50](http://qiniu.xdpie.com/2017-10-18-10-16-50.png)

![2017-10-18-10-18-37](http://qiniu.xdpie.com/2017-10-18-10-18-37.png)

![2017-10-18-10-19-57](http://qiniu.xdpie.com/2017-10-18-10-19-57.png)

![2017-10-18-10-21-22](http://qiniu.xdpie.com/2017-10-18-10-21-22.png)

![2017-10-18-10-23-09](http://qiniu.xdpie.com/2017-10-18-10-23-09.png)

![2017-10-18-10-27-53](http://qiniu.xdpie.com/2017-10-18-10-27-53.png)

![2017-10-18-10-26-36](http://qiniu.xdpie.com/2017-10-18-10-26-36.png)

![2017-10-18-10-29-58](http://qiniu.xdpie.com/2017-10-18-10-29-58.png)

![2017-10-18-10-33-15](http://qiniu.xdpie.com/2017-10-18-10-33-15.png)
![2017-10-18-10-38-02](http://qiniu.xdpie.com/2017-10-18-10-38-02.png)