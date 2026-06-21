# LogQbit 文档

LogQbit 是一个面向实验室实验流程的轻量数据记录工具包。它把每次实验记录成一个目录，
用通用文件格式保存数据、元数据和常量参数，并提供浏览器界面查看记录。

## 核心概念

Python 代码里的主要入口是 `LogFolder`。一个 `LogFolder` 对应磁盘上的一个实验记录目录：

```text
runs/0/
├── data.feather
├── metadata.json
└── const.yaml
```

- `data.feather`: 表格数据，适合用 pandas 读取和分析。
- `metadata.json`: 浏览器相关的轻量状态，例如标题、收藏、回收站状态和绘图轴。
- `const.yaml`: 实验常量、配置参数和人工可读的说明。

最小写入示例：

```python
from logqbit.logfolder import LogFolder

with LogFolder.new("./runs", title="cooldown") as log:
    log.add_row(time=0.0, temperature=300.0)
    log.add_row(time=1.0, temperature=295.2)
    log.add_const(operator="alice", sample="device-a")
```

如果只想读取已经写好的数据，最简单的方式是直接用 pandas：

```python
import pandas as pd

df = pd.read_feather("./runs/0/data.feather")
```

## 浏览和辅助文件

LogQbit 的主要图形界面是 LogBrowser，用来浏览一组实验记录、查看数据表、检查常量和元数据，
并打开绘图视图：

```bash
logqbit-browser ./runs
```

也可以使用主命令的便捷入口：

```bash
logqbit browser ./runs
```

实验记录目录里也可以放用户自己的辅助文件，例如截图、照片、分析摘要或仪器配置导出：

```text
runs/0/
├── data.feather
├── metadata.json
├── const.yaml
├── device_photo.png
└── notes.txt
```

LogBrowser 会直接预览常见图片文件，例如 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`
和 `.gif`。其它额外文件会显示在 `Files` 标签中，可以从 GUI 里打开。

## 文档目录

- [安装](install.md)
- [Python API](api.md)
- [命令行工具和 GUI](cli.md)
- [LabRAD 迁移](migration_guide.md)

## 项目主页

- GitHub: https://github.com/Qiujv/logqbit
- 文档站点: https://qiujv.github.io/logqbit/
