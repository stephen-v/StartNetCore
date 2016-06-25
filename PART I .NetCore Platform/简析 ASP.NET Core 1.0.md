# ASP.NET Core 1.0 执行原理1——Startup Class
ASP.NET Core 是新一代的 ASP.NET，早期称为 ASP.NET vNext，并且在推出初期命名为 ASP.NET 5，但随着 .NET Core 的成熟，以及 ASP.NET 5 的命名会使得外界将它视为 ASP.NET 的升级版，但它其实是新一代从头开始打造的 ASP.NET 核心功能，因此微软宣布将它改为与 .NET Core 同步的名称，即 ASP.NET Core。

ASP.NET Core 可运行于 Windows 平台以及非 Windows 平台，如 Mac OSX 以及 Ubuntu Linux 操作系统，是 Microsoft 第一个具有跨平台能力的 Web 开发框架。

微软在一开始开发时就将 ASP.NET Core 开源，因此它也是开源项目的一员，由 .NET 基金会 (.NET Foundation) 所管理。


## 核心框架
ASP.NET Core 以 .NET Core 的基础发展，其目前规划的功能有：
* ASP.NET Core MVC：ASP.NET Core MVC 提供了开发动态web站点的API，包括了WebPages 和 WebAPI ,最终可运行在IIS 或 自托管(self-hosted)的服务器中。

* DependencyInjection:包含了通用的依赖注入接口,用于在ASP.NET Core MVC中使用。

* Entity Framework Core：与之前版本的EntityFramework版本类似是一个轻量级的ORM框架，包括了Linq,POCO和Codefirst的支持。

* ASP.NET Core Identity:用于在ASP.NET Core web applications构建用户权限系统的框架，包括了membership、login等功能，同时也可以方便的扩展和自定义。


## ASP.NET Core APP 执行原理简析

>为了了解ASP.NET Core的执行原理，将采用命令行工具运行WebAPP而不是之前的Visual studio debug。

**一、安装the .NET Core SDK for Windows(Linux、MAC)**

在各个平台均有各自的SDK,这里以Windows为例，(https://www.microsoft.com/net/core#windows)[下载地址],
安装完成后可以用命令`dotnet -v`查看版本号。

```
C:\Users\stephen>dotnet -v
Telemetry is: Enabled
.NET Command Line Tools (1.0.0-preview1-002702)
Usage: dotnet [common-options] [command] [arguments]
```

**二、命令行生成模版项目**

```
mkdir aspnetcoreapp
cd aspnetcoreapp
dotnet new
```
生成好的模版文件如下：
![模版](http://qiniu.xdpie.com/70aaf0d83f6984a60401184070009ee8.png?imageView2/2/w/700)


**三、修改project.json**
project.json是用于定义项目需要依赖的资源，Web App 需要一个hosting 程序，需要依赖`Kestrel` (),在App部署时根据project.json的依赖文件依靠nuget下载依赖包。

**四、下载依赖包部署网站**

ASP.NET Core 使用NUGET来下载依赖包，在`C:\Users\stephen\.nuget\packages`可以看到nuget已经下载到本地的包，下图可看到目前nuget并没有下载任何包。
![nuget](http://qiniu.xdpie.com/c0a0cfd5f9117f5e74c2a8eaa08c4be2.png?imageView2/2/w/700)

```
dotnet restore

```
慢慢的收获，nuget把需要依赖包全部下载到本体
![nuget](http://qiniu.xdpie.com/e38ee0cb0a7a4423ca4df4674a527f81.png?imageView2/2/w/700)


**五、添加Startup.cs文件**

对于一个ASP.NET Core 程序而言，Startup Class 是程序的入口。ASP.NET Core在程序启动时会从assemblies中找到名字叫Startup的类，如果存在多个名为Startup的类，则会先找到项目根名称空间下的Startup类。

在Startup必须定义Configure方法，而configureServices方法则是可选的，方法会在程序第一次启动时被调用。

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

**五、Web Hosting 配置**

在Program.cs文件中复制如下代码，泳衣制定宿主程序为Kestrel,同时制定程序入口

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
代码完成后，需要调用Royslin编译器将代码编译为assemblies，存储至bin文件夹中。按照上一节所述[](),
ASP.NET Core App 可以编译为IL的assemblies外，还可以通过native直接编译为机器码。

![新增bin文件夹](http://qiniu.xdpie.com/65b91453ff1285a5a47d69f0f0345adc.png?imageView2/2/w/700)


**六、启动**
输入启动命令，Kestrel托管程序并在5000端口监听，至此整个程序启动起来。
```
dotnet run
```
![run](http://qiniu.xdpie.com/f20026b421619afda28b4038269394b1.png?imageView2/2/w/700)

----

## 总结


**参考链接**

【1】 https://docs.asp.net/en/1.0.0-rc2/getting-started.html
