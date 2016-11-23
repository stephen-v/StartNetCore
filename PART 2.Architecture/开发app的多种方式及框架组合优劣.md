# 几款开源的Hybrid移动app框架分析

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [几款开源的Hybrid移动app框架分析](#几款开源的hybrid移动app框架分析)
	- [Ionic](#ionic)
	- [Onsen UI](#onsen-ui)
		- [与 ionic 相比](#与-ionic-相比)
	- [jQuery Mobile](#jquery-mobile)
	- [Mobile Angular UI](#mobile-angular-ui)
	- [结论](#结论)

<!-- /TOC -->

很多移动开发者喜欢使用原生代码开发，但这种方式并不是一个银弹，在需要快速以及低成本开发的时候Hybrid App（混合应用）就体现了它的优势。

HTML5 移动UI框架(例如Ionic)让你创建垮平台Hybrid App与NativeAPP相似的效果，而使用的则是
HTM5, CSS and JavaScript。如果你已经是一个web开发者了那么选择hybird将是一个较好的开发方式。而且只需要编写一套代码就可以在多个平台中使用。通过Cordova(PhoneGap)提供的javascriptAPI可以访问到照相机或传感器这类硬件设备。最后再编译成原生安装包发到各应用商店。

目前市面上有需要移动端的UI框架可供选择，接下来将介绍几款其中的佼佼者。

## Ionic

![github](http://qiniu.xdpie.com/033082cda2c9309af2b9a57f34a26410.png)

![ionic](http://qiniu.xdpie.com/8f2c308812046c727868973becf477bb.png?imageView2/2/w/700)

在近几年，ionic成为了Hybrid App开发框架中的领军者，并且ionic的开发小组继续更新，并保持领先优势。ionic一直保持免费和开源，而且它还拥有庞大的生态系统，可以在社区中找到大量的资源。

ionic添加了对android材料设计的支持，同时ionic也包括了angular。像其他流行的Hybrid App框架一样，ionic也可以利用cordova来实现对原生硬件的调用。

ionic框架具有可维护性和可扩展性，使用了简单清晰的标记，大量移动端特殊优化的css(Sass),HTML5 and JavaScript 组件。

**优点：**

* 基于Angularjs
* 预置的类原生组件
* 强大的社区

** 缺点：**

* 需要了解Angularjs
* 插件更新较慢
* 动画性能较弱

[官方网站](http://ionicframework.com/)

## Onsen UI
![github](http://qiniu.xdpie.com/14e3212ed3f78cec6f669cab77f773c0.png)

![osen](http://qiniu.xdpie.com/8f80c456263505841abe188b0329d534.png?imageView2/2/w/700)

Onsen UI是相对较新的框架，但是却给Ionic带来了冲击。Onsen采用Apache license开源协。Onsen UI 有通过angular的指令实现了大量的组件也提供基于jQuery的组件 。两个框架很类似但是还是存在一些不同：

### 与 ionic 相比
* 两个框架都依赖与angular指令，但Onsen UI支持jQuery。
* 两个框架都支持Android 4+, iOS 6+，Onsen UI 支持Firefox OS和桌面浏览器。但ionic没有官方的桌面浏览器支持，但还是可以用。
* 都支持分屏显示技术
* 都是扁平是风格，但个人觉得Ionic更好看点。
* ionic支持SASS而Onsen UI 则是基于 Topcoat Css library。
* Onsen UI 文档较好，但ionic的社区较活跃。
* Onsen UI 有一个自己的IDE called Monaca IDE.


[官方网站](https://onsen.io/)

## jQuery Mobile

![github](http://qiniu.xdpie.com/24807452108b9fc2a0fdbe8aa4f8ae3b.png)

![mobile](http://qiniu.xdpie.com/d0b1cca4c321efd7479f18f114a49ff2.png?imageView2/2/w/700)

jQuery 依然在游戏领域与其他移动端框架抗衡。jQuery Mobile 建立在jQuery和jQueryUI的基础上。允许开发者创建webapp获得与平板、pc上无差别的用户体验。因此它无法提供类似移动端原生控件外观和体验的app.

[官方网站](https://jquerymobile.com/)

## Mobile Angular UI

![github](http://qiniu.xdpie.com/e7a0730ad96026cbfd68e8d44c8410db.png)

![angular mobile](http://qiniu.xdpie.com/215b1f9e7139ecd588badae6ba2843d7.png?imageView2/2/w/700)

这是为bootstrap和angular的粉丝而准备的。 通过 Mobile Angular UI ,可以通过bootstrap3和Angular 构建 移动应用。

Mobile Angular UI 提供指令可以构建移动端UI Component 例如 overlays, switches.sidebars,scrollable .

[官方网站](http://mobileangularui.com/)

## 结论
目前市面上最常用的几款开源的移动端框架，总体上来均不错，但如果是要追求最终app的视觉效果则Ionic与Onsen是较好的选择，它们的UI看上去更像原生控件。如果你是jquery的粉丝，并且不想尝试使用其他的那么可以选择jQuery Mobile 简单高效。
如果你熟悉angular与bootstarp那么Mobile Angular UI则是不错的选择。

除了开源的框架外还有一些企业级框架这些框架功能强大但需要相应的费用，比如Sencha Touch 与 Kendo UI 。


----
参考链接

[01] http://noeticforce.com/best-hybrid-mobile-app-ui-frameworks-html5-js-css
