# LogQbit

LogQbit 基于文件/文件夹保存实验数据，并提供图形化的数据浏览器。

## 安装

Python 中创建、写入和读取记录只需安装核心包：

```bash
pip install logqbit
```

数据浏览器等图形化界面需要 GUI extra：

```bash
pip install "logqbit[gui]"
```

随后可通过以下命令启动浏览器：

```bash
logqbit browser
```

创建快捷方式及其他功能介绍见[命令行工具](cli.md)和[LogBrowser 使用指南](browser.md)。

## 快速开始

```python
from logqbit import LogFolder

log = LogFolder.new("./runs", title="cooldown")
log.add_row(time=0.0, temperature=300.0)
log.add_row(time=1.0, temperature=295.2)
log.add_const(sample="device-a", operator="alice")
log.meta.plot_axes = ["time"]
```

上例会在 `runs/` 下创建一条新记录。
记录目录通常包含以下三个文件：

```text
1/
├── data.feather
├── metadata.json
└── const.yaml
```

存入的数据可以直接获取：

```python
print(log.df)
print(log.const["sample"])
```

读取已有记录使用 `LogRecord`：

```python
from logqbit import LogRecord

record = LogRecord("./runs/1")

df = record.df
print(record.meta.title)
print(record.row_count, record.columns)
```

也可以直接通过 `pandas` 读取：

```python
import pandas as pd

df = pd.read_feather("./runs/1/data.feather")
```

追加写入、批量浏览等更多功能见[核心 API](core.md)。

## 项目链接

- GitHub: <https://github.com/Qiujv/logqbit>
- 文档站点: <https://qiujv.github.io/logqbit/>
- PyPI: <https://pypi.org/project/logqbit/>
