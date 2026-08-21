# LogQbit

LogQbit 是一个用目录组织实验记录的轻量数据工具包。每条记录可以包含表格数据、
实验常量和用于浏览、筛选与绘图的信息；既可在 Python 中写入和读取，也可在
LogBrowser 中查看。

## 安装

Python 中创建、写入和读取记录只需安装核心包：

```bash
pip install logqbit
```

LogBrowser 和实时绘图需要 GUI extra：

```bash
pip install "logqbit[gui]"
```

## 快速开始

```python
from logqbit import LogFolder

log = LogFolder.new("./runs", title="cooldown")
log.add_row(time=0.0, temperature=300.0)
log.add_row(time=1.0, temperature=295.2)
log.add_const(sample="device-a", operator="alice")
log.meta.plot_axes = ["time"]
```

上例会在 `runs/` 下创建一条新记录。数据保存在 `data.feather`，常量保存在
`const.yaml`，记录标题和绘图设置保存在 `metadata.json`。通常不需要手动编辑这些
文件。

写入、打开和批量浏览记录的 Python 接口见[核心 API](core.md)。使用图形界面时，
可直接打开记录的父目录：

```bash
logqbit browser ./runs
```

## 文档导航

- [核心 API](core.md)：使用 `LogFolder`、`LogRecord` 和 `LogCatalog`。
- [LogBrowser 使用指南](browser.md)：浏览、整理和绘制已有记录。
- [命令行工具](cli.md)：打开 Browser、生成示例和复制迁移模板。
- [从 LabRAD 迁移](migration_guide.md)：将已有 LabRAD 数据转换为 LogQbit 记录。

## 项目链接

- GitHub: <https://github.com/Qiujv/logqbit>
- 文档站点: <https://qiujv.github.io/logqbit/>
- PyPI: <https://pypi.org/project/logqbit/>
