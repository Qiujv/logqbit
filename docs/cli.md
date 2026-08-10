# 命令行工具

## LogBrowser


启动 LogBrowser，浏览 `LogFolder` 记录：

```bash
logqbit browser
```

Browser 在独立进程中运行，需要让 Browser 在当前进程中运行并保持终端连接时，可以使用 foreground 模式：

```bash
logqbit browser --foreground
```

也可以通过 Python 模块启动：

```bash
python -m logqbit.gui.browser ./runs
```

界面和绘图交互见 [LogBrowser 使用指南](browser.md)。

## Live Plotter

旧的独立实时绘图窗口仍保留兼容入口，但不再是主要界面：

```bash
logqbit-live-plotter
```

## 实用命令

在当前目录创建 `logqbit_example/`，生成示例数据并启动图形化浏览器。

```bash
logqbit browser-demo
```

复制模板脚本到当前目录或指定目录。当前可用模板适合 LabRAD 数据迁移。

```bash
logqbit copy-template move_from_labrad
logqbit copy-template move_from_labrad -o ./tools/
```

在 Windows 上创建桌面快捷方式。

```bash
logqbit shortcuts
logqbit shortcuts -o "C:\MyShortcuts"
```
