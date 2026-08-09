# LogQbit

LogQbit 是一个面向实验室场景的轻量数据记录工具包，适合记录中小规模实验数据。

它的核心设计目标是：

- 用简单的接口记录和管理实验数据。
- 基于目录的实验数据组织，用通用可读的格式储存数据。
- 在需要时再启用浏览器和实时绘图等 GUI 能力。

完整文档见：https://qiujv.github.io/logqbit/

## 安装

常规安装：

```bash
pip install logqbit
```

如果还需要 GUI：

```bash
pip install "logqbit[gui]"
```

更详细的安装说明见：https://qiujv.github.io/logqbit/install/

## 用例

### 用例 1：记录一组实验数据

```python
from logqbit import LogFolder

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

如果希望启动后立即释放当前终端，可以使用 detached 模式：

```bash
logqbit browser --detach
```

或者也可以通过 `logqbit shortcuts` 创建桌面快捷方式（仅 Windows 可用）。

### 用例 3：生成示例数据并打开浏览器

```bash
logqbit browser-demo
```

这会在当前目录创建 `logqbit_example/`，生成五组示例数据并启动图形化浏览器。其中两组
包含一百万个数据点，首次生成可能需要一点时间。

更多内容：

- 文档首页：https://qiujv.github.io/logqbit/
- LogBrowser：https://qiujv.github.io/logqbit/browser/
- 命令行工具：https://qiujv.github.io/logqbit/cli/
- LabRAD 迁移：https://qiujv.github.io/logqbit/migration_guide/
- Python API：https://qiujv.github.io/logqbit/api/
