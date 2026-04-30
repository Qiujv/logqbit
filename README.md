# LogQbit

LogQbit 是一个面向实验室场景的轻量数据记录工具包，适合记录中小规模实验数据，并提供日志目录、元数据、常量配置和可选的 GUI 浏览/实时绘图能力。

它的核心设计目标是：

- 用尽量少的样板代码记录结构化实验数据。
- 让 `LogFolder` 和 `Registry` 这类核心接口保持简单直接。
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

project_folder = Path("test")
project_folder.mkdir(exist_ok=True)

lf = LogFolder.new(project_folder, title="My Experiment")

lf.add_row(x=0.0, y=1.2)
lf.add_row(x=1.0, y=1.8)
lf.flush()

lf.meta.plot_axes = ["x"]
lf.const["temperature"] = "300 K"

print(lf.df)
```

### 用例 2：快速生成示例数据并打开浏览器

```bash
logqbit browser-demo
```

这个命令会在当前目录创建 `logqbit_example/`，生成多组示例数据，并尝试启动图形化浏览器。

更多内容：

- 文档首页：https://qiujv.github.io/logqbit/
- 核心用法：https://qiujv.github.io/logqbit/core_usage/
- 命令行工具：https://qiujv.github.io/logqbit/cli/
- LabRAD 迁移：https://qiujv.github.io/logqbit/migration_guide/
