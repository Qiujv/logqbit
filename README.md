# LogQbit

*LogQbit* 是一个轻量且可扩展的实验数据记录工具包。
它最初用于记录 量子比特（qubit）实验测量数据，但凭借灵活的数据格式和实时可视化功能，同样适用于 任意中小规模（≤MB级）实验数据 的采集与管理。

*LogQbit* is a lightweight and extensible data logging toolkit for lab-scale experiments.
It was originally developed for recording quantum qubit measurement data, 
but its flexible format and real-time visualization tools make it suitable for any small to medium (≤MB-level) experimental dataset.

通过使用 `logqbit`，你可以：

- 以最少的样板代码记录结构化实验数据；

- 使用集成的实时绘图工具可视化数据流；

- 通过交互式日志浏览器查看与分析记录的数据。

With `logqbit`, you can:

- Record structured experimental data with minimal boilerplate.

- Visualize data streams in real time with an integrated live plotter.

- Browse and analyze logged results through an interactive log browser.

无论是量子比特读出、参数扫描，还是传感器输出记录，
`logqbit` 都能为你提供一个简洁而可靠的实验数据采集与回溯工作流。

Whether you are monitoring qubit readouts, scanning a parameter sweep, or simply logging sensor outputs, 
`logqbit` provides a simple and robust workflow for capturing and revisiting your experimental data.

## 安装 Installation

从 PyPI 安装 / Install from PyPI:

```bash
pip install logqbit
```

## 命令行工具 Command-Line Tools

安装后，以下命令可用：

After installation, the following commands are available:

- `logqbit-browser [directory]` - 启动交互式日志浏览器 GUI / Launch the interactive log browser GUI
- `logqbit-live-plotter` - 启动实时绘图窗口 / Launch the live plotting window
- `logqbit browser-demo` - 创建示例数据并启动浏览器 / Create example data and launch browser
- `logqbit copy-template <name>` - 复制迁移/工具模板到工作目录 / Copy migration/utility templates to your working directory
- `logqbit shortcuts` - 在桌面创建带自定义图标的快捷方式 / Create desktop shortcuts with custom icons

### 快速体验 Quick Demo

想快速体验 logqbit browser 的功能？运行以下命令：

Want to quickly experience logqbit browser? Run:

```bash
logqbit browser-demo
```

这将创建示例数据并自动启动浏览器：

This will create example data and automatically launch the browser:

- `logqbit_example/` - 包含 3 个示例日志文件夹 / Contains 3 example log folders
  - **示例 0** - 线性关系：`y = 2x + 1` 和 `z = x²` / Linear relationship: `y = 2x + 1` and `z = x²`
  - **示例 1** - 带噪声的正弦信号 / Noisy sinusoidal signal
  - **示例 2** - 2D 参数扫描（共振模拟）/ 2D parameter scan (resonance simulation)

浏览器将自动打开并显示示例数据。

The browser will automatically open and display the example data.

### 从 LabRAD 迁移数据 Data Migration from LabRAD

如果你有 LabRAD 格式的现有数据，可以轻松迁移：

If you have existing data in LabRAD format, you can easily migrate it:

```bash
# 复制迁移模板 / Copy the migration template
logqbit copy-template move_from_labrad

# 编辑 move_from_labrad.py 设置路径 / Edit move_from_labrad.py to set your paths
# 然后运行 / Then run it
python move_from_labrad.py
```

详细说明请参阅 [迁移指南](docs/migration_guide.md)。

See [Migration Guide](docs/migration_guide.md) for detailed instructions.

### 创建桌面快捷方式 Create Desktop Shortcuts

在桌面创建 LogQbit Browser 和 Live Plotter 的快捷方式（带自定义图标）：

Create desktop shortcuts for LogQbit Browser and Live Plotter with custom icons:

```bash
# 在桌面创建快捷方式 / Create shortcuts on desktop
logqbit shortcuts

# 在指定目录创建快捷方式 / Create shortcuts in a specific directory
logqbit shortcuts -o "C:\MyShortcuts"
```

这将创建以下快捷方式：

This will create the following shortcuts:

- **LogQbit Browser.lnk** - 日志浏览器快捷方式 / Log browser shortcut
- **LogQbit Live Plotter.lnk** - 实时绘图工具快捷方式 / Live plotter shortcut

每个快捷方式都配有对应的自定义图标。

Each shortcut comes with its corresponding custom icon.

## 文档 Documentation

- [首页 index](docs/index.md)
- [迁移指南 Migration Guide](docs/migration_guide.md) - 从 LabRAD 迁移数据 / Migrate data from LabRAD format

## 快速开始 Quick Start

### 基本用法 Basic Usage

```python
from pathlib import Path
from logqbit.logfolder import LogFolder

# 创建项目文件夹 / Create project folder
project_folder = Path('test')
project_folder.mkdir(exist_ok=True)

# 创建新的日志文件夹 / Create a new log folder
lf = LogFolder.new(project_folder)

# 添加数据行 / Add data rows
lf.add_row(x=1.5, y=2.0)
lf.add_row(x=233, y=[1, 3, 5])

# 访问数据 / Access data
print(lf.df)

# [可选] 设置元数据 / [Optional] Set metadata
lf.meta.title = "My Experiment"
lf.meta.star = 1
lf.meta.plot_axes = ["x"]

# [可选] 添加常量配置 / [Optional] Add constant configuration
lf.const["temperature"] = "300 K"
lf.const["description"] = "Test measurement"
```

### API 说明 API Reference

**LogFolder 主要属性和方法 Main Attributes and Methods:**

- `lf.df` - 获取完整数据帧 / Get full dataframe
- `lf.meta` - 元数据（标题、星标、标签等）/ Metadata (title, star, tags, etc.)
- `lf.const` - 常量配置（实验参数等）/ Constant configuration (experiment parameters, etc.)
- `lf.add_row(**kwargs)` - 添加数据行 / Add data row
- `lf.flush()` - 强制保存数据 / Force save data
