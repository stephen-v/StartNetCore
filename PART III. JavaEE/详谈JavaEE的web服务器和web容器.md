# JavaEE中Web服务器、Web容器、Application服务器区别及联系

>在JavaEE 开发Web中，我们经常会听到Web服务器(Web Server)、Web容器(Web Container)、应用服务器(Application Server)，等容易混淆不好理解名词。本文介绍对三者的理解，以及区别与联系，如果有不正确的地方还请指正。


![Web Server And Application Server](http://qiniu.xdpie.com/3ffccd10dbea438c2bb75be2e3cb37dc.png)

由上图可以看到一个标准的http处理流程：
1. 首先通过Web Server 接受Http请求;
2. 比如html、css等静态资源 Web Server 可自行处理;
3. 当遇到动态资源(jsp等)时候Web Server 将请求转接至Application Server中，由Application Server处理;



## Web服务器(Web Server)

`Web Server` 或者叫 `HTTP Server` ,主要用于操作Http请求，包括接受客户端的请求以及响应。它可以处理请求，也可以将请求转发至其他服务器。

代表：`Nginx`、`apache`、`IIS`

**Web Server市场占有率如下**

![市场占有](http://qiniu.xdpie.com/c1ac33f6e621b522b96f891dabc1227b.png)


## 应用服务器(JavaEE Application Server)

Application Server 具备了 Web Server 处理http请求的能力(但可能没有Web Server专业)同时也支持了JavaEE 技术比如JMS、DI、JPA、Transactions、Concurrency等，同时也包含了Web Container，如下图。

代表:`Bea WebLogic`, `IBM WebSphere`

![web sphere](http://qiniu.xdpie.com/1e91fb876230c84a36041b7bdd8e1e2a.png)

应用服务器可以选择使用上文所说的 `WebLogic` 和 `WebSphere` 这种重量级产品外，也可以使用类似与`Tomcat`、`jetty`这样的web containner 再加上第三方的框架(spring,hibernate等)来构建自己的`Application Server`。


**JavaEE Application Server市场占有率**

![市场占有](http://qiniu.xdpie.com/e3315cadb11608b1710e201e2540a3cb.png)


## 组合应用

一个典型的JavaEE系统可以由两部分构成首先是Web Server 用于处理静态资源，然后是JavaEE Application Server 用于处理业务的动态资源。而这两部分可以是单独的服务器例如Nginx+WebSphere也可以在一个服务器上完成比如Tomcat(Tomcat即可以处理静态资源又可以处理动态的Servlet)。

![](http://qiniu.xdpie.com/47eb5634343824891a177cf10740eb14.png)



----
参考链接

[1]http://stackoverflow.com/questions/12689910/difference-between-web-server-web-container-and-application-server

[2]http://stackoverflow.com/questions/5039354/difference-between-an-application-server-and-a-servlet-container?noredirect=1&lq=1

[3]https://coderanch.com/t/598746/Websphere/application-server-web-server-web

[4]http://www.ibm.com/support/knowledgecenter/zh/SSAW57_7.0.0/com.ibm.websphere.nd.iseries.doc/info/iseriesnd/ae/welc6tech_ovrex.html
