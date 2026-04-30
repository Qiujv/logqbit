# 命令行工具

## 图形界面命令

### `logqbit-browser [directory]`

启动图形化日志浏览器。

```bash
logqbit-browser ./data/logqbit_data
```

### `logqbit-live-plotter`

启动实时绘图窗口。

```bash
logqbit-live-plotter
```

### `logqbit browser [directory]`

这是 `logqbit-browser` 的简写入口。

```bash
logqbit browser ./data/logqbit_data
```

## 实用命令

### `logqbit browser-demo`

在当前目录创建 `logqbit_example/`，写入示例数据，并尝试启动浏览器。

```bash
logqbit browser-demo
```

当前实现会生成多组示例数据，包括：

- 线性关系示例
- 带噪正弦信号
- 二维参数扫描
- 大规模一维数据
- 大规模二维数据

### `logqbit copy-template <name>`

复制模板脚本到当前目录或指定目录。

```bash
logqbit copy-template move_from_labrad
logqbit copy-template move_from_labrad -o ./tools/
```

当前可用模板适合 LabRAD 数据迁移。

### `logqbit shortcuts`

在 Windows 上创建带图标的桌面快捷方式。

```bash
logqbit shortcuts
logqbit shortcuts -o "C:\MyShortcuts"
```

## 使用建议

- 如果你的环境没有安装 GUI 依赖，优先使用 Python API 和非 GUI 命令。
- `browser-demo` 和 `shortcuts` 最终都会触发 GUI 相关导入。
- 在已有实验环境中，如果只需要核心读写能力，可以结合 `uv pip install --no-deps` 使用。