# ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)](#aspnet-core-运行原理剖析2startup-和-middleware中间件)
	- [Startup Class](#startup-class)
		- [1、Configure方法](#1configure方法)
		- [2、ConfigureServices](#2configureservices)
		- [3、Startup Constructor（构造函数）](#3startup-constructor构造函数)
	- [Middleware](#middleware)
		- [1、中间件注册](#1中间件注册)
		- [2、常用中间件](#2常用中间件)

<!-- /TOC -->

>在上一节[(文章链接)](http://www.cnblogs.com/vipyoumay/p/5613373.html)中提到ASP.NET Core WebApp 必须含有Startup类,在本节中将重点讲解Startup类以及Middleware(中间件)在Startup类中的使用。

## Startup Class
Startup Class中含有两个重要方法：Configure方法用于每次http请求的处理，比如后面要讲的中间件(Middleware)，就是在configure方法中配置。而ConfigureServices方法在Configure方法前调用，它是一个可选的方法，可在configureServices依赖注入接口或一些全局的框架，比如EntityFramework、MVC等。

### 1、Configure方法
在Configure方法中，通过入参的依赖注入(DI),可注入以下对象：
* `IApplicationBuilder`:用于构建应用请求管道。通过IApplicationBuilder下的run方法传入管道处理方法即中间件(中间件会在后文中详细介绍)。这是最常用方法，对于一个真实环境的应用基本上离不开中间件比如权限验证、跨域、异常处理等。下面代码调用IApplicationBuilder.use方法注册中间件。拦截每个http请求，输出Hello World。


```cs
public void Configure(IApplicationBuilder app)
{
	app.Run((context) => context.Response.WriteAsync("Hello World!"));
}
```

* `IHostingEnvironment`:用于访问应用程序的特殊属性，比如`applicationName`,`applicationVersion`。


![hosting](http://qiniu.xdpie.com/47d38eaaf04f5086317a03827e44c605.png?imageView2/2/w/700)

* `ILoggerFactory`:提供创建日志的接口，可以选用已经实现接口的类或自行实现此接口,下面代码使用最简单的控制台作为日志输出。

``` cs
        public void Configure(IApplicationBuilder app, IHostingEnvironment env, ILoggerFactory logger)
        {
            var log = logger.CreateLogger("default");
            logger.AddConsole();
            log.LogInformation("start configure");
            app.Run( (context) =>
            {
                return context.Response.WriteAsync("Hello World!");
            });
        }
```

![logger](http://qiniu.xdpie.com/5b3a9f59c5e22cf0bf3d9f83fe0a6359.png?imageView2/2/w/700)


### 2、ConfigureServices

* **IServiceCollection**：整个ASP.NET Core 默认带有依赖注入(DI)，IServiceCollection是依赖注入的入口，下面实现了简易的一个类(Foo)和接口(IFoo),代码清单如下：

```cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApplication1
{
   public interface IFoo
    {
        string GetFoo();
    }
}


```

```cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApplication1
{
    public class Foo : IFoo
    {
        public string GetFoo()
        {
            return "foo";
        }
    }
}

```

在ConfigureServices 中注入

```cs
public void ConfigureServices(IServiceCollection services)
       {
           services.AddTransient<IFoo, Foo>();
       }

```

在Configure方法中注册中间件，让中间件在中间件的执行代码中使用注入的IFoo接口

```cs

app.Run((context) =>
           {
               var str = context.RequestServices.GetRequiredService<IFoo>().GetFoo();
               return context.Response.WriteAsync(str);
           });

```

除了自己的接口外还可以通过扩展方法添加更多的注入方法，比如EntityFramework、mvc框架都实现自己的添加方法。

``` cs
public void ConfigureServices(IServiceCollection services)
{
    // Add framework services.
    services.AddDbContext<ApplicationDbContext>(options =>
        options.UseSqlServer(Configuration.GetConnectionString("DefaultConnection")));

    services.AddIdentity<ApplicationUser, IdentityRole>()
        .AddEntityFrameworkStores<ApplicationDbContext>()
        .AddDefaultTokenProviders();

    services.AddMvc();

    // Add application services.
     services.AddTransient<IFoo, Foo>();

}

```


### 3、Startup Constructor（构造函数）
在构造函数中可以注入IHostingEnvironment、ILoggerFactory功能同上，此处不再赘述。

## Middleware
中间件是一个处理http 请求和 响应的组件，多个中间件构成了处理管道(Handler pipeline)，每个中间件可以决定是否传递至管道中的下一中间件。一旦注册中间件后，每次请求和响应均会被调用。

![调用示意](http://qiniu.xdpie.com/9748b3bdfa96bcfb20e7fc9108a0e177.png?imageView2/2/w/700)

### 1、中间件注册
中间件的注册在startup中的Configure方法完成，在configure方法中使用IApplicationBuilder对象的Run、Map、Use方法传入匿名委托(delegate)。

* Map:含有两个参数pathMatche和configuration，通过请求的url地址匹配相应的configuration。

** Run & Use:添加一个中间件至请求管道。它们在功能很类似但是也存在一些区别，先来看下两个方法的定义。

``` cs
 public static IApplicationBuilder Use(this IApplicationBuilder app, Func<HttpContext, Func<Task>, Task> middleware);

 public static void Run(this IApplicationBuilder app, RequestDelegate handler);

```

Run是通过扩展方法语法来定义，传入入参是RequestDelegate的委托,执行完一个第一个run后是不会激活管道中的第二个run方法，这样代码执行结果只会输出一个“hello world!”

```cs

app.Run((context) => context.Response.WriteAsync("Hello World!"));

app.Run((context) => context.Response.WriteAsync("Hello World 1!"));

```

![run](http://qiniu.xdpie.com/5b432f7cd86c1c89779dd77e54d63524.png?imageView2/2/w/700)

而use方法的入参则是Func<>的委托包含两个入参和一个返回值,这样在第一个函数执行完成后可以选择是否继续执行后续管道中的中间件还是中断。

```cs

app.Use((context, next) =>
					 {
							 context.Response.WriteAsync("ok");
							 return next();
					 });
app.Use((context, next) =>
					 {
							 return context.Response.WriteAsync("ok");
					 });

```

![Use](http://qiniu.xdpie.com/4ffa0cb722bc45c0456c7569134e6222.png?imageView2/2/w/700)

### 2、常用中间件

| Middleware    | 功能描述   |
| :------------- | :------------- |
| Authentication       | 提供权限支持     |
| CORS       | 跨域的配置     |
| Routing      | 配置http请求路由     |
| Session      | 管理用户会话     |
| Static Files      | 提供对静态文件的浏览    |

这里有一些官方的示例,[链接](https://github.com/Microsoft-Build-2016/CodeLabs-WebDev/tree/master/Module2-AspNetCore)


参考链接
[1] https://docs.asp.net/en/latest/fundamentals/middleware.html
[2] http://www.talkingdotnet.com/app-use-vs-app-run-asp-net-core-middleware/
