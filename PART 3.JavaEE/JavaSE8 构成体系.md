# JavaSE 8 构成体系

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [JavaSE 8 构成体系](#javase-8-构成体系)
	- [JavaSE 是什么？与JDK JRE是什么关系？](#javase-是什么与jdk-jre是什么关系)
	- [JRE 构成](#jre-构成)
		- [虚拟机](#虚拟机)
		- [lang、util 基础库](#langutil-基础库)
			- [math](#math)
			- [Collections](#collections)
			- [Ref Object](#ref-object)
			- [Regular Expressions](#regular-expressions)
			- [Logging](#logging)
			- [Management](#management)
			- [Concurrency Utilities](#concurrency-utilities)

<!-- /TOC -->

## JavaSE 是什么？与JDK JRE是什么关系？

Java SE 是Java平台标准版的简称（Java Platform, Standard Edition） ，用于开发和部署桌面、服务器以及嵌入设备和实时环境中的Java应用程序。Java SE包括用于开发Java Web服务的类库，同时，Java SE为Java EE提供了基础。

Oracle 有两款产品实现了Java Platform Standard Edition (Java SE) 8:Java SE Development Kit (JDK) 8 and Java SE Runtime Environment (JRE) 8. JDK 8 是JRE 8 的超集 , 包括了JRE 8中的所有。同时也包括更多的工具例如编译器、调试器。而在JRE 8中提供了许多类，以及Java Virtual Machine (JVM)，和其他组件。

![java conceptual](http://qiniu.xdpie.com/f42bd2cc5069c61ceab1bbe1c2e10f6c.png?imageView2/2/w/700)

## JRE 构成

### 虚拟机

hotspot包括server和client两种模式的实现：Java HotSpot Client VM(-client)，为在客户端环境中减少启动时间而优化；
Java HotSpot Server VM(-server)，为在服务器环境中最大化程序执行速度而设计。Server VM启动比Client VM慢，运行比Client VM快。server模式的运行中，垃圾回收处理做的比较好一些。[虚拟机的配置可参看此文](http://blog.csdn.net/magi1201/article/details/41597831)

### lang、util 基础库

**包 java.lang 、 java.util**

#### math
包括浮点运算库`java.lang.Math`和`java.lang.StrictMath`和任意精度的`java.math`.

[详细说明](https://docs.oracle.com/javase/8/docs/technotes/guides/math/index.html)

#### Collections

Java Collections 的接口有两类，一类以Collection作为基础，另一类则以Map做为基础。

![](http://qiniu.xdpie.com/9b8326a50e5aa0daf7184aaa18f85a47.png)

1.Collection 接口

exdends Iterable 接口,同时包括一些基本的操作`int size()`,` boolean isEmpty()`, `boolean contains(Object element)`, `boolean add(E element)`,` boolean remove(Object element)`, ` Iterator<E> iterator()`

2.Set 接口

Set 不能包含重复的元素。有三种主要的Set，`HashSet`、`TreeSet`、`LinkedHashSet` 。其中`HashSet`在插入时候效率最高，但在遍历时效率较低。而且不能保证元素按照插入顺序访问。而`LinkedHashSet`则可以按照顺序访问，同样也是采用HashCode值来存储元素。`TreeSet`使用树结构存储，可以按照值进行排序，并可以对排序进行定制。

3.List 接口

List 可以包含重复元素，并且有序。包含两种类型`ArrayList` ` LinkedList` 。 新增或删除元素的时候`LinkedList`更有优势。而随机访问某个元素的时候`ArrayList`却更好。

4.Queue、Deque 接口

队列以及双向队列

5.Map接口

键值对结合，不能包含重复的键。同`Set`一样有三种特点相同主要的类型:`HashMap` ` LinkedHashMap ` ` TreeMap `

#### Ref Object

**包：java.lang.ref**

对象的引用分为4种级别，从而使程序能更加灵活地控制对象的生命周期。这4种级别由高到低依次为：强引用、软引用、弱引用和虚引用。

1.强引用（StrongReference）

强引用是使用最普遍的引用。如果一个对象具有强引用，那垃圾回收器绝不会回收它。当内存空间不足，Java虚拟机宁愿抛出OutOfMemoryError错误，使程序异常终止，也不会靠随意回收具有强引用的对象来解决内存不足的问题。

2.软引用（SoftReference）

如果一个对象只具有软引用，则内存空间足够，垃圾回收器就不会回收它；如果内存空间不足了，就会回收这些对象的内存。只要垃圾回收器没有回收它，该对象就可以被程序使用。软引用可用来实现内存敏感的高速缓存（下文给出示例）。
软引用可以和一个引用队列（ReferenceQueue）联合使用，如果软引用所引用的对象被垃圾回收器回收，Java虚拟机就会把这个软引用加入到与之关联的引用队列中。

3.弱引用（WeakReference）

弱引用与软引用的区别在于：只具有弱引用的对象拥有更短暂的生命周期。在垃圾回收器线程扫描它所管辖的内存区域的过程中，一旦发现了只具有弱引用的对象，不管当前内存空间足够与否，都会回收它的内存。不过，由于垃圾回收器是一个优先级很低的线程，因此不一定会很快发现那些只具有弱引用的对象。
弱引用可以和一个引用队列（ReferenceQueue）联合使用，如果弱引用所引用的对象被垃圾回收，Java虚拟机就会把这个弱引用加入到与之关联的引用队列中。

4.虚引用（PhantomReference）

“虚引用”顾名思义，就是形同虚设，与其他几种引用都不同，虚引用并不会决定对象的生命周期。如果一个对象仅持有虚引用，那么它就和没有任何引用一样，在任何时候都可能被垃圾回收器回收。
虚引用主要用来跟踪对象被垃圾回收器回收的活动。虚引用与软引用和弱引用的一个区别在于：虚引用必须和引用队列 （ReferenceQueue）联合使用。当垃圾回收器准备回收一个对象时，如果发现它还有虚引用，就会在回收对象的内存之前，把这个虚引用加入到与之 关联的引用队列中。


#### Regular Expressions
正则表达式，此处略去介绍都懂的。

#### Logging
**包：java.util.logging**

日志记录

[具体可参看](http://docs.oracle.com/javase/8/docs/technotes/guides/logging/overview.html)

#### Management

对java平台监控和管理的支持,包括api和一组工具:[jconsole](http://jiajun.iteye.com/blog/810150)、[jps](http://blog.csdn.net/fwch1982/article/details/7947451)


#### Concurrency Utilities

**包：java.util.concurrent**

java并行编程库,提供了强大、高效、高性能的线程管理工具例如`thread pool` `blocking queues`

[具体可参看](http://docs.oracle.com/javase/tutorial/essential/concurrency/index.html)

#### Reflection

**包：java.lang.reflect**

反射允许java 代码去发现加载类的属性、方法、构造的信息，并使用它们。为了安全目的反射只能访问目标对象的公共成员。

[官方文档入口](http://docs.oracle.com/javase/tutorial/reflect/index.html)

[使用示例](http://www.cnblogs.com/rollenholt/archive/2011/09/02/2163758.html)

#### Preferences

**包: java.util.prefs**

用于对应用程序偏好和配置存储和查询。

[官方文档入口](http://docs.oracle.com/javase/8/docs/technotes/guides/preferences/index.html)
[使用示例](http://www.cnblogs.com/littlehb/p/3511689.html)

#### JAR

**包：java.util.jar**

用于将应用程序打包。

[使用示例](http://www.cnblogs.com/adolfmc/archive/2012/10/07/2713562.html)

#### ZIP

**包：java.util.zip **

提供ZIP和GZIP格式的读和写的库。

[使用示例](http://blog.csdn.net/gaowen_han/article/details/7163737/)


### Ohter Base Libraries

#### JavaBeans

JavaBeans 是一种基于组件的架构，JavaBeans的组件可以在软件开发中重用，这样便可以方便的开发和组装程序。

#### Java Security

Java Security API提供了可互操作的算法和安全服务的实现。服务以provider的形式实现，可以以插件的形式植入应用程序中。程序员可以透明地使用这些服务，如此使得程序员可以集中精力在如何把安全组件集成到自己的应用程序中，而不是去实现这些安全功能。此外，除了Java提供的安全服务外，用户可以编写自定义的security provider，按需扩展Java的security平台

[使用示例](http://joshuasabrina.iteye.com/blog/1798245)

#### Serialization

#### Extension Mechanism

#### JMX

JMX是一种JAVA的正式规范，它主要目的是让程序有被管理的功能.

#### JAX

JAXP:是Java XML程序设计的应用程序接口之一，它提供解析和验证XML文档的能力
JAXB:JAXB主要用来实现对象和XML之间的序列化和反序列化
JAX-WS:用于简单创建和发布web service

####　Networking

####　JNI

JNI是Java Native Interface的缩写，它提供了若干的API实现了Java和其他语言的通信

### Integration Libraries



----
参考链接

[1] http://docs.oracle.com/javase/8/docs/

[2] https://docs.oracle.com/javaee/7/tutorial/index.html
