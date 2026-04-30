# LogQbit

LogQbit 是一个面向实验室场景的轻量数据记录工具包，适合记录中小规模实验数据。

它的核心设计目标是：

- 用简单的接口和运行逻辑实验数据。
- 基于目录的实验数据组织，用通用可读的格式储存数据。
- 在需要时再启用浏览器和实时绘图等 GUI 能力。

完整文档见：https://qiujv.github.io/logqbit/

## 安装

常规安装：

```bash
pip install logqbit
```

如果你不想让安装过程修改当前 Python 环境，可以跳过依赖解析：

```bash
pip install --no-deps logqbit
```

通常 `logqbit.logfolder.LogFolder`、`logqbit.registry.Registry` 等核心接口仍然可用；
但浏览器、实时绘图和快捷方式等 GUI 功能仍然依赖 `PySide6`、`pyqtgraph` 等组件。

更详细的安装说明见：https://qiujv.github.io/logqbit/install/

## 用例

### 用例 1：记录一组实验数据

```python
from pathlib import Path

from logqbit.logfolder import LogFolder

lf = LogFolder.new("./demo_data", title="My Experiment")

lf.add_row(x=0.0, y=1.2)
lf.add_row(x=1.0, y=1.8)
lf.flush()

lf.meta.plot_axes = ["x"]
lf.const["temperature"] = "300 K"

print(lf.df)
```

### 用例 2：打开数据查看器

```bash
logqbit browser
```

或者也可以通过 `logqbit shortcuts` 创建桌面快捷方式（仅 Windows 可用）。

### 用例 3：快速生成示例数据并打开浏览器

```bash
logqbit browser-demo
```

这会在当前目录创建 `logqbit_example/`，生成多组示例数据，并启动图形化浏览器。

更多内容：

- 文档首页：https://qiujv.github.io/logqbit/
- 命令行工具：https://qiujv.github.io/logqbit/cli/
- LabRAD 迁移：https://qiujv.github.io/logqbit/migration_guide/
- Python API：https://qiujv.github.io/logqbit/api/
