# LogQbit 文档

LogQbit 用目录保存实验记录。每条记录可以包含表格数据、实验常量和少量用于浏览、筛选与绘图的信息；既可在 Python 中写入和读取，也可在 LogBrowser 中查看。

## 从这里开始

1. 按[安装说明](install.md)安装核心包；要使用图形界面，请安装 GUI extra。
2. 用 `LogFolder` 创建记录并写入数据。
3. 运行 `logqbit browser ./runs` 浏览记录。

最小示例：

```python
from logqbit import LogFolder

log = LogFolder.new("./runs", title="cooldown")
log.add_row(time=0.0, temperature=300.0)
log.add_row(time=1.0, temperature=295.2)
log.add_const(sample="device-a", operator="alice")
log.meta.plot_axes = ["time"]
```

上例会在 `runs/` 下创建一条新记录。数据保存在 `data.feather`，常量保存在 `const.yaml`，记录标题和绘图设置保存在 `metadata.json`。通常不需要手动编辑这些文件。

## 文档目录

- [安装](install.md)
- [Core API](core.md)
- [LogBrowser 使用指南](browser.md)
- [命令行工具](cli.md)
- [从 LabRAD 迁移](migration_guide.md)

## 项目主页

- GitHub: <https://github.com/Qiujv/logqbit>
- 文档站点: <https://qiujv.github.io/logqbit/>
- PyPI: <https://pypi.org/project/logqbit/>
