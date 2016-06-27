# ASP.NET Core 运行原理剖析1:Startup 和 Middleware(中间件)

>在上一节[(文章链接)](http://www.cnblogs.com/vipyoumay/p/5613373.html)中已经提到一个ASP.NET Core WebApp 必须要有个Startup类,本节将重点讲解Startup类以及中间件在Startup类中的使用。

## Startup Class
Startup Class中含有两个重要方法：Configure方法用于每次http请求的处理，比如后面要讲的中间件(Middleware)，就是在configure方法中配置。而ConfigureServices方法在Configure方法前调用，它是一个可选的方法，可在configureServices配置一些全局的框架，比如EntityFramework、MVC等。

**一、Configure方法**
在Configure方法中，通过DI(依赖注入),可注入以下对象：
* `IApplicationBuilder`:用于构建应用请求管道。通过IApplicationBuilder下的run方法传入管道处理方法(中间件)。这个是最常用方法，对于一个现实环境的应用基本上离不开中间件比如权限验证、跨域、异常处理等。

* `IHostingEnvironment`:用于访问应用程序的特殊属性，比如`applicationName`,`applicationVersion`。

* `ILoggerFactory`:提供创建日志的接口，可以选用已经实现接口的类或自行实现此接口。

![hosting](http://qiniu.xdpie.com/47d38eaaf04f5086317a03827e44c605.png?imageView2/2/w/700)

**二、ConfigureServices**


**三、Startup Constructor（构造函数）**


## Middleware
