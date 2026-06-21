# 命令行工具和 GUI

## 图形界面命令

启动 LogBrowser，浏览一组 `LogFolder` 记录。

```bash
logqbit-browser ./runs
```

也可以通过主命令的便捷入口启动同一个浏览器。

```bash
logqbit browser ./runs
```

启动实时绘图窗口。

```bash
logqbit-live-plotter
```

## 实用命令

在当前目录创建 `logqbit_example/`，写入示例数据，并尝试启动浏览器。

```bash
logqbit browser-demo
```

复制模板脚本到当前目录或指定目录。当前可用模板适合 LabRAD 数据迁移。

```bash
logqbit copy-template move_from_labrad
logqbit copy-template move_from_labrad -o ./tools/
```

在 Windows 上创建带图标的桌面快捷方式。

```bash
logqbit shortcuts
logqbit shortcuts -o "C:\MyShortcuts"
```
