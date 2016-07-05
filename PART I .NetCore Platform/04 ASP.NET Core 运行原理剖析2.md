# ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [ASP.NET Core 运行原理剖析2:Startup 和 Middleware(中间件)](#aspnet-core-运行原理剖析2startup-和-middleware中间件)
	- [Startup Class](#startup-class)
		- [1、Startup Constructor（构造函数）](#1startup-constructor构造函数)
		- [2、ConfigureServices](#2configureservices)
		- [3、Configure方法](#3configure方法)
	- [Middleware](#middleware)
		- [1、中间件注册](#1中间件注册)
		- [2、常用中间件](#2常用中间件)



>在上一节[(文章链接)](http://www.cnblogs.com/vipyoumay/p/5620373.html)中提到ASP.NET Core WebApp 必须含有Startup类,在本节中将重点讲解Startup类以及Middleware(中间件)在Startup类中的使用。

## Startup Class
Startup Class中含有两个重要方法：Configure方法用于每次http请求的处理，比如后面要讲的中间件(Middleware)，就是在configure方法中配置。而ConfigureServices方法在Configure方法前调用，它是一个可选的方法，可在configureServices依赖注入接口或一些全局的框架，比如EntityFramework、MVC等。**Startup 类的 执行顺序：`构造 -> configureServices->configure`**。


### 1、Startup Constructor（构造函数）

**主要实现一些配置的工作，方法参数如下：**

* **`IHostingEnvironment`:** 用于访问应用程序的特殊属性，比如`applicationName`,`applicationVersion`。通过`IHostingEnvironment`对象下的属性可以在构造中实现配置工作。比如获取当前根路径找到配置json文件地址，然后ConfigurationBuilder初始化配置文件，最后可以通过GetSection()方法获取配置文件。代码清单如下：

```cs

var builder = new ConfigurationBuilder()
							.SetBasePath(env.ContentRootPath)
							 .AddJsonFile("appsettings.json");
					 var configuration = builder.Build();
					 var connStr = configuration.GetSection("Data:DefaultConnection:ConnectionString").Value;

```
根目录下的配置文件如下：
```json

{
	"Data": {
		"DefaultConnection": {
			"ConnectionString": "Server=(localdb)\\MSSQLLocalDB;Database=_CHANGE_ME;Trusted_Connection=True;"
		}
	}
}

```



* **`ILoggerFactory`:** 提供创建日志的接口，可以选用已经实现接口的类或自行实现此接口,下面代码使用最简单的控制台作为日志输出。

``` cs
public Startup(IHostingEnvironment env, ILoggerFactory logger)
 {
		 var log = logger.CreateLogger("default");
		 logger.AddConsole();
		 log.LogInformation("start configure");
 }
```

![logger](http://qiniu.xdpie.com/5b3a9f59c5e22cf0bf3d9f83fe0a6359.png?imageView2/2/w/700)


### 2、ConfigureServices

**主要实现了依赖注入(DI)的配置，方法参数如下：**

* **IServiceCollection**：整个ASP.NET Core 默认带有依赖注入(DI)，IServiceCollection是依赖注入的容器，首先创建一个类(Foo)和接口(IFoo),代码清单如下：

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

在ConfigureServices 中将接口和实现注入至容器

```cs
public void ConfigureServices(IServiceCollection services)
       {
           services.AddTransient<IFoo, Foo>();
       }

```

如果想在每次Http请求后都使用IFoo的GetFoo()方法来处理，上面讲到可以在Configure方法中注册函数，在注册过程中由于使用了依赖注入(DI)，因此可以直接通过`RequestServices.GetRequiredService<IFoo>()`泛型方法将IFoo对象在容器中取出。

```cs

app.Run((context) =>
           {
               var str = context.RequestServices.GetRequiredService<IFoo>().GetFoo();
               return context.Response.WriteAsync(str);
           });

```

除了自己的接口外,还支持通过扩展方法添加更多的注入方法，比如EntityFramework、mvc框架都实现自己的添加方法。

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

### 3、Configure方法

**主要是http处理管道配置和一些系统配置，参数如下：**

* **`IApplicationBuilder`:** 用于构建应用请求管道。通过IApplicationBuilder下的run方法传入管道处理方法。这是最常用方法，对于一个真实环境的应用基本上都需要比如权限验证、跨域、异常处理等。下面代码调用IApplicationBuilder.Run方法注册处理函数。拦截每个http请求，输出Hello World。


```cs
public void Configure(IApplicationBuilder app)
{
	app.Run((context) => context.Response.WriteAsync("Hello World!"));
}
```

* **`IHostingEnvironment`:** 同构造参数

* **`ILoggerFactory`:** 同构造参数






## Middleware
中间件是一个处理http请求和响应的组件，多个中间件构成了处理管道(Handler pipeline)，每个中间件可以决定是否传递至管道中的下一中间件。一旦注册中间件后，每次请求和响应均会被调用。

![调用示意](http://qiniu.xdpie.com/9748b3bdfa96bcfb20e7fc9108a0e177.png?imageView2/2/w/700)

### 1、中间件注册
中间件的注册在startup中的Configure方法完成，在configure方法中使用IApplicationBuilder对象的Run、Map、Use方法传入匿名委托(delegate)。上文示例注册IFoo.GetFoo()方法就是一个典型的中间件。

* **Run & Use:** 添加一个中间件至请求管道。它们在功能很类似但是也存在一些区别，先来看下两个方法的定义。

``` cs
 public static IApplicationBuilder Use(this IApplicationBuilder app, Func<HttpContext, Func<Task>, Task> middleware);

 public static void Run(this IApplicationBuilder app, RequestDelegate handler);

```

Run是通过扩展方法语法来定义，传入入参是RequestDelegate的委托,执行完一个第一个run后是不会激活管道中的第二个run方法，这样代码执行结果只会输出一个“hello world!”

```cs

app.Run((context) => context.Response.WriteAsync("Hello World!"));

app.Run((context) => context.Response.WriteAsync("Hello World 1!"));

```

![run](http://qiniu.xdpie.com/45b355f7238091d3a389747f634055d5.png?imageView2/2/w/700)

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
![Use](http://qiniu.xdpie.com/ffa7a5524a645ef32e4ea82290d66121.png?imageView2/2/w/700)

* **Map:** 含有两个参数pathMatch和configuration，通过请求的url地址匹配相应的configuration。例如可以将url路径是/admin的处理函数指定为如下代码：

```cs
app.Map("/admin", builder =>
					{
							builder.Use((context, next) => context.Response.WriteAsync("admin"));
					});

```

### 2、常用中间件

| Middleware    | 功能描述   |
| :------------- | :------------- |
| Authentication       | 提供权限支持     |
| CORS       | 跨域的配置     |
| Routing      | 配置http请求路由     |
| Session      | 管理用户会话     |
| Static Files      | 提供对静态文件的浏览    |

这里有一些官方的示例,[链接](https://github.com/Microsoft-Build-2016/CodeLabs-WebDev/tree/master/Module2-AspNetCore)


---

> **以上内容有任何错误或不准确的地方请大家指正，不喜勿喷！**

> 作者：帅虫哥 出处： [http://www.cnblogs.com/vipyoumay/p/5640645.html ](http://www.cnblogs.com/vipyoumay/p/5640645.html )

> **本文版权归作者和博客园共有，欢迎转载，但未经作者同意必须保留此段声明，且在文章页面明显位置给出原文连接，否则保留追究法律责任的权利。如果觉得还有帮助的话，可以点一下右下角的【推荐】，希望能够持续的为大家带来好的技术文章！想跟我一起进步么？那就【关注】我吧。**

*参考链接*

[1] https://docs.asp.net/en/latest/fundamentals/middleware.html

[2] http://www.talkingdotnet.com/app-use-vs-app-run-asp-net-core-middleware/
