# 安装

## 常规安装

推荐直接从 PyPI 安装：

```bash
uv pip install logqbit
```

## 不解析依赖的安装方式

如果你已有一个稳定环境，不希望安装 logqbit 时触发依赖升级、降级或重新解析，可以使用：

```bash
uv pip install --no-deps logqbit
```

这会跳过依赖安装，只安装 logqbit 本体。

适用场景：

- 你已有一套手工维护的实验环境。
- 你不希望 GUI 栈影响现有环境。
- 你只需要 `LogFolder`、`Registry` 等核心接口。

注意事项：

- `--no-deps` 不是“无依赖运行”。
- 如果环境里没有 `numpy`、`pandas`、`pyarrow`、`ruamel-yaml`、`tqdm` 等核心依赖，核心功能也无法运行。
- GUI 相关功能还依赖 `PySide6` 和 `pyqtgraph`。

## 哪些功能依赖 GUI 组件

以下功能依赖 GUI 相关组件：

- `logqbit-browser`
- `logqbit-live-plotter`
- `logqbit browser`
- `logqbit browser-demo`
- `logqbit shortcuts`

如果你后续需要这些功能，可以再补装：

```bash
uv pip install pyside6 pyqtgraph
```
