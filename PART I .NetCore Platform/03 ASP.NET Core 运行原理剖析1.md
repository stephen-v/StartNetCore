# ASP.NET Core 运行原理剖析1:初始化WebApp模版并运行


<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [ASP.NET Core 运行原理剖析1:初始化WebApp模版并运行](#aspnet-core-运行原理剖析1初始化webapp模版并运行)
	- [核心框架](#核心框架)
	- [ASP.NET Core APP 创建与运行](#aspnet-core-app-创建与运行)
	- [总结](#总结)

<!-- /TOC -->

>之前两篇文章分别介绍了，[简析.NET Core 以及与 .NET Framework的关系](http://www.cnblogs.com/vipyoumay/p/5603928.html)和[.NET Core的构成体系](http://www.cnblogs.com/vipyoumay/p/5613373.html),接下来计划用一个系列对ASP.NET Core的运行原理进行剖析。便于自己可以更好的了解ASP.NET Core。




ASP.NET Core 是新一代的 ASP.NET，早期称为 ASP.NET vNext，并且在推出初期命名为ASP.NET 5，但随着 .NET Core 的成熟，以及 ASP.NET 5的命名会使得外界将它视为 ASP.NET 的升级版，但它其实是新一代从头开始打造的 ASP.NET 核心功能，因此微软宣布将它改为与 .NET Core 同步的名称，即 ASP.NET Core。

ASP.NET Core 可运行于 Windows 平台以及非 Windows 平台，如 Mac OSX 以及 Ubuntu Linux 操作系统，是 Microsoft 第一个具有跨平台能力的 Web 开发框架。

微软在一开始开发时就将 ASP.NET Core 开源，因此它也是开源项目的一员，由 .NET 基金会 (.NET Foundation) 所管理。


>正式版的.NET Core已于今天发布(2016年6月27日)，具体可看[微软 .NET Core 1.0 正式发布下载](http://www.codechannels.com/zh/article/microsoft/microsoft-releases-net-core-1-0-final-rtm/)


## 核心框架

ASP.NET Core 以 .NET Core 的基础发展，目前规划的功能有：
* **ASP.NET Core MVC:** ASP.NET Core MVC 提供了开发动态web站点的API，包括了WebPages 和 WebAPI ,最终可运行在IIS 或 自托管(self-hosted)的服务器中。

* **DependencyInjection:** 包含了通用的依赖注入接口,用于在ASP.NET Core MVC中使用。

* **Entity Framework Core:** 与之前版本的EntityFramework版本类似是一个轻量级的ORM框架，包括了Linq,POCO和Codefirst的支持。

* **ASP.NET Core Identity:** 用于在ASP.NET Core web applications构建用户权限系统的框架，包括了membership、login等功能，同时也可以方便的扩展和自定义。


## ASP.NET Core APP 创建与运行


**一、安装the .NET Core SDK for Windows(Linux、MAC)**

以Windows为例，([下载地址](https://www.microsoft.com/net/core#windows)),
安装完成后可以用命令`dotnet -v`查看版本号。

```
C:\Users\stephen>dotnet -v
Telemetry is: Enabled
.NET Command Line Tools (1.0.0-preview1-002702)
Usage: dotnet [common-options] [command] [arguments]
```

**二、命令行生成模版项目**

需要开发一个webapp可以从头开始创建文件，也可以通过命令行生成一个空的项目模版

```
mkdir aspnetcoreapp
cd aspnetcoreapp
dotnet new
```

依次执行命令后，便可在路径下，生成好模版：


![模版](http://qiniu.xdpie.com/17881f1f0b27fcb8f08c220b6390386d.png?imageView2/2/w/700)


模版包括以下三个文件：

* **project.json:**

主掌项目的运行期的配置设置，包含项目的包参考 (Package References)、项目的基本设置、引导指令、包含或排除指定目录、以及建造时的相关事件指令等。



* **Program.cs:**

程序入口文件

* **project.lock.json:**

与project.json相比，是project.json文件中引用包的完整引用列表。



**三、修改project.json**

project.json是用于定义项目需要依赖的资源，每个WebApp 需要一个hosting 程序(IIS、IISExpress等)，而此次使用`Kestrel` ([什么是kestrel?](http://www.cnblogs.com/artech/p/KestrelServer.html))。



**四、下载依赖包部署网站**

在WebApp部署时(dotnet restore)根据project.json的依赖文件,依靠nuget下载依赖包,完成对整个程序的restore。

在`C:\Users\stephen\.nuget\packages`可以看到nuget已经下载到本地的包，在开始部署前nuget是不会加载依赖包，下图可看到目前nuget并没有下载任何包。


![nuget](http://qiniu.xdpie.com/1778b5d28e882d3683ce85765467d006.png?imageView2/2/w/700)


然后执行命令

```
dotnet restore

```

可以看到，nuget已经自动将需要依赖包下载到本地

![nuget](http://qiniu.xdpie.com/18821013e887a7a6189bf605c4fafbda.png?imageView2/2/w/700)





**五、添加Startup.cs文件**

对于一个ASP.NET Core 程序而言，`Startup Class` 是必须的。ASP.NET Core在程序启动时会从assemblies中找到名字叫Startup的类，如果存在多个名为Startup的类，则会先找到项目根名称空间下的Startup类。

在Startup必须定义`Configure`方法，而`configureServices`方法则是可选的，方法会在程序第一次启动时被调用。

在刚才文件路径下添加Startup.cs文件，并复制如下代码:

```cs
using System;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;

namespace aspnetcoreapp
{
    public class Startup
    {
        public void Configure(IApplicationBuilder app)
        {
            app.Run(context =>
            {
                return context.Response.WriteAsync("Hello from ASP.NET Core!");
            });
        }
    }
}
```

**六、Web Hosting 配置**

在Program.cs文件中复制如下代码，指定WebApp宿主程序为`Kestrel`:

```cs
using System;
using Microsoft.AspNetCore.Hosting;

namespace aspnetcoreapp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var host = new WebHostBuilder()
                .UseKestrel()
                .UseStartup<Startup>()
                .Build();

            host.Run();
        }
    }
}
```

**七、编译**

```
dotnet build
```
代码完成后，需要调用Roslyn编译器将代码编译为assemblies，存储至bin文件夹中。按照上一节所述([简析 .NET Core 构成体系](http://www.cnblogs.com/vipyoumay/p/5613373.html)),
ASP.NET Core App 可以编译为IL的assemblies外，还可以通过native直接编译为机器码。


![新增bin文件夹](http://qiniu.xdpie.com/65b91453ff1285a5a47d69f0f0345adc.png?imageView2/2/w/700)


**八、启动**

输入启动命令，Kestrel托管WEB程序,并在5000端口监听，至此整个程序启动起来。
```
dotnet run

```

![run](http://qiniu.xdpie.com/a1723c1c7f8f2df43a1aedd76ed37bfe.png?imageView2/2/w/700)

![run](http://qiniu.xdpie.com/f20026b421619afda28b4038269394b1.png?imageView2/2/w/700)



## 总结

本节介绍了ASP.NET Core 项目从创建、配置、编译、发布、运行的过程，ASP.NET Core与之前的ASP.NET相比具有更高的透明度和灵活性，可以快速的在各个操作系统中开发与运行。

本节使用Windows操作系统，但目前微软也在 linux和mac 下提供了类似的命令行工具([链接地址](https://www.microsoft.com/net/core#ubuntu))，方便在 linux和mac 下开发与部署，在后面文章中会详细讲解，本节不再累述。

----

> **以上内容有任何错误或不准确的地方请大家指正，不喜勿喷！**

> 作者：帅虫哥 出处： [http://www.cnblogs.com/vipyoumay/p/5603928.html](http://www.cnblogs.com/vipyoumay/p/5603928.html)

> **本文版权归作者和博客园共有，欢迎转载，但未经作者同意必须保留此段声明，且在文章页面明显位置给出原文连接，否则保留追究法律责任的权利。如果觉得还有帮助的话，可以点一下右下角的【推荐】，希望能够持续的为大家带来好的技术文章！想跟我一起进步么？那就【关注】我吧。**


**参考链接**

【1】 https://docs.asp.net/en/1.0.0-rc2/getting-started.html
