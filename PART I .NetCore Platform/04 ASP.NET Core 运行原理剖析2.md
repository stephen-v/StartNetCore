# ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)](#aspnet-core-运行原理剖析2startup-和-middleware中间件)
	- [Startup Class](#startup-class)
		- [一、Configure方法](#一configure方法)
		- [二、ConfigureServices](#二configureservices)
		- [三、Startup Constructor（构造函数）](#三startup-constructor构造函数)
	- [Middleware](#middleware)

<!-- /TOC -->

>在上一节[(文章链接)](http://www.cnblogs.com/vipyoumay/p/5613373.html)中已经提到一个ASP.NET Core WebApp 必须要有个Startup类,本节将重点讲解Startup类以及中间件在Startup类中的使用。

## Startup Class
Startup Class中含有两个重要方法：Configure方法用于每次http请求的处理，比如后面要讲的中间件(Middleware)，就是在configure方法中配置。而ConfigureServices方法在Configure方法前调用，它是一个可选的方法，可在configureServices配置一些全局的框架，比如EntityFramework、MVC等。

### 一、Configure方法
在Configure方法中，通过DI(依赖注入),可注入以下对象：
* `IApplicationBuilder`:用于构建应用请求管道。通过IApplicationBuilder下的run方法传入管道处理方法(中间件)。这个是最常用方法，对于一个现实环境的应用基本上离不开中间件比如权限验证、跨域、异常处理等。下面代码调用IApplicationBuilder.use方法注册中间件。拦截每个http请求，输出Hello World。


```cs
   app.Run((context) => context.Response.WriteAsync("Hello World!"));

```

* `IHostingEnvironment`:用于访问应用程序的特殊属性，比如`applicationName`,`applicationVersion`。


![hosting](http://qiniu.xdpie.com/47d38eaaf04f5086317a03827e44c605.png?imageView2/2/w/700)

* `ILoggerFactory`:提供创建日志的接口，可以选用已经实现接口的类或自行实现此接口,下面代码使用最简单的控制台作为日志输出。

``` cs

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env, ILoggerFactory logger)
        {
            var log = logger.CreateLogger("default");
            logger.AddConsole();
            log.LogInformation("start configure");


            app.Run(async (context) =>
            {
                await context.Response.WriteAsync("Hello World!");
            });
        }

```

![logger](http://qiniu.xdpie.com/5b3a9f59c5e22cf0bf3d9f83fe0a6359.png?imageView2/2/w/700)


### 二、ConfigureServices

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


### 三、Startup Constructor（构造函数）


## Middleware
