# 基于 Angularjs&Node.js 云编辑器架构设计及开发实践

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [基于 Angularjs&Node.js 云编辑器架构设计及开发实践](#基于-angularjsnodejs-云编辑器架构设计及开发实践)
	- [一、产品背景](#一产品背景)
	- [二、总体架构](#二总体架构)
		- [1. 前端架构](#1-前端架构)
			- [a.前端层次](#a前端层次)
			- [b.核心基础模块设计](#b核心基础模块设计)
			- [c.业务模块设计](#c业务模块设计)
		- [2. Node.js端设计](#2-nodejs端设计)
	- [三、单元测试](#三单元测试)
	- [四、持续集成及自动部署](#四持续集成及自动部署)
	- [五、打包](#五打包)
	- [六、总结](#六总结)

<!-- /TOC -->

## 一、产品背景
产品是一个内部项目，主要是`基于语义网本体的云端编辑器，用于构建语义知识库`。抛开语义网本体概念不谈
，简单说就是一个简易的visual studio 云端编辑器。而图形显示则是在百度脑图的基础上改进的，增加了自己的形状和颜色，以及其他功能。

![总览图](http://qiniu.xdpie.com/03bdd9d074c95f8808c3256474994372.png?imageView2/2/w/700)


> 具体语义网、本体是什么 (http://baike.baidu.com/link?url=6ctsNtk-dthPu-3kiKK_JdMikArIfvbD4VMAQuc685--88X4lggwo58R-q6zKAGlVUcN_RlxQRr5rlPE3B12WK)

## 二、总体架构

整个系统是由三层构成,分别为UI、API、Infrastructure:

**UI层**

UI层主要是用于用户交互界面，目前主要是基于Web的显示方式，未来可扩展移动app。web端采用单页面程序，
所有数据通过ajax请求，最后通过nginx反向代理将前端站点和API站点部署在同一域名下。

`技术：Html5、Css3、Es5`
`框架: agularjs`

**API**

在API层中主要提供UI端所需要的API以及对外的云API。

`技术：node.js`
`框架: express等`

**Infrastructure**

基础层主要提供知识库相关操作，以及数据相关操作,基础层主要是基于Apache jena（Jena是一个语义网本体框架）的封装。以及对阿里云一些服务的封装比如数据库，缓存等。

`技术: java`
`框架:spring spring.data jena 等`

![总体](http://qiniu.xdpie.com/07f23d1d543f59d4ef87e01ceb04135e.png?imageView2/2/w/700)

### 1. 前端架构

#### a.前端层次
前端核心功能是编辑器，由于业务复杂而多变，因此在逻辑上对前端进行了分层处理。系统由三层结构构成，
由下至上分别为infrastructure、configuration、webapp构成，每层又含有若干模块。

![前端分层](http://qiniu.xdpie.com/21c1c700ab04f6e5a21f897dc2186466.png?imageView2/2/w/700)

![代码结构](http://qiniu.xdpie.com/0dac19ff746ced13b4747a6e8583486f.png?imageView2/2/w/700)

在代码文件中`app文件对应webapp层`，`configuration文件对应configuration层`，`lib文件对应infrastructure层`。

**infrastructure层包含了系统所使用的基础框架，包括下面若干框架：**

* Editor: 自行开发WebIDE布局框架，主要用于WebIDE的布局设计等。
* Kity: 百度前端团队的脑图框架，主要用于类图显示等。
* Util: Util主要涵盖了第三方的库例如jQuery、bootstrap、eventbus等除Editor和Kity外其他库均属于Util范畴。


**configuration层包含了系统所需要配置的文件，主要由以下几个部分构成：**

* MenuConfiguration: 用于应用程序顶部菜单和右键菜单的配置。
* FileConfiguration: 用于对编辑器所支持的文件配置。
* ServiceConfiguration: 用于服务器端请求配置。
* Theme: 定义了系统基本皮肤和皮肤扩展。

webapp层包含了界面显示的页面模块均为单页面程序：

* UserDashboard: 用户工作台入口，管理项目，新增项目等。
* EditSpace: 编辑器工作区。

#### b.核心基础模块设计

**Edtior**：是编辑器的功能基础，用于控制编辑器窗口布局、分割、样式等，主要包括三个核心类：

* Editor: 每个编辑器只有一个Editor对象，是编辑器的基础对象，其他对象都可以组合至Editor对象中。
* Tabcontainer：Tabcontainer对象是editor下的一个区域对象，用于控制布局、定位、显示、变化等特性，在Editor中可以有多个Tabcontainer。
* Panel：Panel对象是Tabcontainer下的一个内容对象，用于控制内容，一个Tabcontainer下可以有多个Panel实现对选项卡效果。

```javascript
var Editor = require('/lib/editor-ui/src/editor/editor');
 var editorUi = new Editor('#container');
 editorUi.init('Split5');

 // region top
 console.log(editorUi.getTabContainer('split5-top').getSize());
 module.exports = window.editorUi = editorUi;

 var Panel = require('/lib/editor-ui/src/panel/panel');
 var containerTop = editorUi.getTabContainer('split5-top');
 containerTop.isTabCard = false;
 containerTop.addPanel(new Panel({name: '点击我隐藏', template: editSpaceTemplateConfiguration.toolBarTemplate}));

 // endregion

 // region left
 var containerLeft = editorUi.getTabContainer('split5-bottom-left');
 containerLeft.direction = 'left';
 containerLeft.addPanel(new Panel({name: 'left点击我隐藏', templateHtml: 'left我是内容'}));
 containerLeft.addPanel(new Panel({name: 'left点击我隐藏1', templateHtml: 'left我是内容1'}));
 containerLeft.addPanel(new Panel({name: 'left点击我隐藏2', templateHtml: 'left我是内容2'}));
```

**MenuConfiguration**:用于应用程序顶部菜单和右键菜单的配置,由于菜单功能复杂多变因此采用配置结构。其中顶部工具栏菜单最为复杂，因为根据选中内容不同，需要动态隐藏或显示。我们采用了管道式处理方式，执行逻辑`双击或
单击界面上的任何元素激发依次管道过滤，在管道过滤中每个按钮维护自己的过滤逻辑返回bool值告诉是否通过过滤`。

![toolbar](http://qiniu.xdpie.com/ffa31706c426e9d9f8a2e092a5cb9f0a.png?imageView2/2/w/700)


```json
{
	 "name": "新建本体",
	 "cssClass": "kop-tool-bar-ontology",
	 "disabledCssClass": "kop-tool-bar-ontology-disabled-icon",
	 "commandName": "toolbar/ontology/new",
	 "enableHandler": "newOntologyEnableHandler",
	 "children": []
 }
```

**渲染按钮 的步骤如下：**

* 获取菜单中当前类型需要显示的按钮。
* 根据按钮的类型图标等显示按钮。

**动态改变按钮 （在选中文件或元素不同时按钮会呈现可用和不可用状态） 的步骤如下：**

* 双击或单击事件激活过滤管道
* 执行按钮数组中所有enableHandler方法。
* 返回为false时发起一个buttonstatusChanges事件。
* 菜单controller接受此事件改变按钮数据，最终改变界面元素（双向绑定）。

除此之外还包括`FileConfiguration`和`ServiceConfiguration`,用于文件显示图标和右键菜单的配置

![新建菜单](http://qiniu.xdpie.com/5a3ce4523c7c79f8d4866855d62c36fa.png?imageView2/2/w/700)

![文件图标](http://qiniu.xdpie.com/62f29967f30b7f99d76948cdd2273ff8.png?imageView2/2/w/700)

``` json
var fileConfiguration =[{
  "type":"owl",  //文件类型
  "iconSkin":"/asset/img/ico.jpg"
},...]
```

```json
var serviceConfiguration = {
 "host": "http://dev.onteditor.ad.6starhome.com/api/"
}
```


#### c.业务模块设计

EditSpace是编辑器业务模块，由以下几个子模块构成,每个模块在项目初始化时通过调用editor框架将模版注入至页面中：

* 头部(Headbar)
* 工具栏(Toolbar)
* 项目区(ProjectWindows)
* 导入区(ExportWindows)
* 设计区(DesignWindows)
* 属性区(PropertyWindows)
* 编辑区(EditWindows)
* 状态栏(StatusBar)

在系统初始化时候加入各种模块

```javascript
var Panel = KOP.EditorUI.Panel;
           var TabContainer = KOP.EditorUI.TabContainer;
           // endregion

           // region 左边框初始大小，禁用选中效果
           var leftSpace = editorUi.getTabContainer('split8-main-space-wrap');
           leftSpace.getObject().width(300);
           editorUi.getTabContainer('split8-main-space-edit').getObject().height(250);

           // region 头部模版注入
           var headBarContainer = editorUi.getTabContainer('split8-top');
           headBarContainer.addPanel(new Panel({template: editSpaceTemplateConfiguration.headBarTemplate}));
```

![模块分区](http://qiniu.xdpie.com/48c2974269d3394822ae127665a59a6a.png?imageView2/2/w/700)

![代码清单](http://qiniu.xdpie.com/4776542e241495f8581e2ffd9a194437.png?imageView2/2/w/700)

在每个模块中包含了`单元测试(test)` `事件处理中心(event-handler)` `界面(html)` `控制器(controller)` 如下图所示：

![](http://qiniu.xdpie.com/5f6da80ebcf3a74dfc88759699e6713d.png?imageView2/2/w/700)

事件处理中心接受全局的事件，比如之前配置好的按钮的事件，其他模块与之通信的事件,均是在事件模块中处理。

```javascript
function onNewClass($scope) {
           $scope.$on('toolbar/class/new', function (e, param) {
                     });
       }
```
### 2. Node.js端设计

Nodejs端主要是对各种业务的组装，通过express框架暴露RestfulAPI,这里不再赘述。

## 三、单元测试

可参看之前文章 http://www.cnblogs.com/vipyoumay/p/5331787.html

## 四、持续集成及自动部署

* 代码管理: gitlab(本地部署)
* CI服务器: jenkins(本地部署)
* 构建部署：docker(本地部署)

> 整个配置太多了就不讲解了，有需要可以留言

## 五、打包

采用grunt对前端打包,如何打包请自行百度。

## 六、总结

整个项目工作量很大，我们团队6人，在规定时间内（包括需求设计一共3个月）完成，持续集成自动化部署给我们节约了很多时间，
另外另外架构的设计也为我们在需求微调赢的了时间。

设计包括的内容很多，这里不能一一说完，如果有兴趣可评论或私信交流。
