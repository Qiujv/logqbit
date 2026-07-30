# 安装

## 核心功能

只使用 `LogFolder`、`LogMetadata`、`Registry` 和 `LogCatalog` 等 Python
接口时，直接安装核心包：

```bash
uv pip install logqbit
```

## GUI 功能

浏览器、实时绘图和 Windows 桌面快捷方式使用单独的 `gui` extra：

```bash
uv pip install "logqbit[gui]"
```

它会额外安装 `PySide6`、`pyqtgraph`、`numba` 和 `send2trash`。

## 不解析依赖的安装方式

如果你已有一个稳定环境，不希望安装 logqbit 时触发依赖升级、降级或重新解析，可以使用：

```bash
uv pip install --no-deps logqbit
```

这会跳过依赖安装，只安装 logqbit 本体。

适用场景：

- 你已有一套手工维护的实验环境。
- 你已经自行准备好 LogQbit 所需的核心依赖。

注意事项：

- `--no-deps` 不是“无依赖运行”。
- 如果环境里没有 `numpy`、`pandas`、`pyarrow`、`ruamel-yaml`、`tqdm` 等核心依赖，核心功能也无法运行。
- `--no-deps` 也不会自动安装 `gui` extra 中的组件。

## 哪些功能依赖 GUI 组件

以下功能依赖 GUI 相关组件：

- `logqbit-browser`
- `logqbit-live-plotter`
- `logqbit browser`
- `logqbit browser-demo`
- `logqbit shortcuts`

如果你后续需要这些功能，可以把 GUI extra 补装到现有环境：

```bash
uv pip install "logqbit[gui]"
```
