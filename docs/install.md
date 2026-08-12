# 安装

## 核心功能

Python 中创建、写入和读取记录只需安装核心包：

```bash
pip install logqbit
```

## 图形界面

LogBrowser 和实时绘图需要 GUI extra：

```bash
pip install "logqbit[gui]"
```

安装后可用下面的命令确认可启动：

```bash
logqbit browser
```

如果要先看看示例数据，可以运行：

```bash
logqbit browser-demo
```

它会在当前目录创建 `logqbit_example/` 并打开浏览器。

## 更新

```bash
pip install -U "logqbit[gui]"
```

只使用核心功能时，将命令中的 `"logqbit[gui]"` 改为 `logqbit` 即可。
