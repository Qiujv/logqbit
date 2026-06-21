# Python API

LogQbit 的核心入口是 `LogFolder`。一个 `LogFolder` 对应磁盘上的一个实验记录目录，
目录里通常包含三类文件：

- `data.feather`: 表格数据。
- `metadata.json`: 与 LogBrowser 交互的轻量元数据。
- `const.yaml`: 实验常量和配置参数。

## LogFolder

`LogFolder` 负责创建记录目录、追加数据行、管理元数据和常量，并在需要时把缓冲数据写入磁盘。

### 创建新记录

```python
from logqbit.logfolder import LogFolder

with LogFolder.new("./runs", title="cooldown") as log:
    log.add_row(time=0.0, temperature=300.0)
    log.add_row(time=1.0, temperature=295.2)

    log.add_const(operator="alice", sample="device-a")
    log.meta.plot_axes = ["time"]
```

`LogFolder.new(parent)` 会在 `parent` 下创建下一个数字目录，例如 `0/`、`1/`、`2/`。
推荐使用 `with` 语句；退出代码块时会自动 `close()`，确保末段数据完成写入并关闭后台线程。

### 打开已有记录

```python
log = LogFolder("./runs/0", create=False)
print(log.df)
log.close()
```

如果路径不存在并且 `create=False`，会抛出 `FileNotFoundError`。

### 追加数据

标量参数会追加一行：

```python
log.add_row(x=1.0, y=2.0)
```

列表、数组或其他有长度的参数会一次追加多行：

```python
log.add_row(
    x=[0.0, 1.0, 2.0],
    y=[1.2, 1.8, 2.1],
)
```

多行写入会交给 pandas 检查列长度是否一致。

### 读取和保存

```python
df = log.df
log.flush()
log.close()
```

- `log.df` 返回当前完整 dataframe，包括还没有写入磁盘的缓冲行。
- `log.flush()` 立即同步写入 `data.feather`，调用会阻塞直到写入完成。
- `log.close()` 会先 flush，再停止后台 autosave 线程。它是幂等的，可以重复调用。

如果只需要读取已经写好的数据文件，最简单的方式是直接用 pandas：

```python
import pandas as pd

df = pd.read_feather("./runs/0/data.feather")
```

这适合做只读分析、导出脚本或不需要创建 `LogFolder` 对象的场景。

普通脚本自然退出时，LogQbit 也会通过 `atexit` 尝试关闭仍然活跃的 `LogFolder`。
对象被垃圾回收时还有 `weakref.finalize` 兜底。不过需要强保证时，仍然推荐使用 `with`
或显式调用 `close()`。

### 常量和元数据

```python
log.add_const(temperature="300 K", bias=0.1)
log.add_const_to_head(run_group="calibration")

log.const["instrument/name"] = "scope-a"
log.meta.star = True
log.meta.plot_axes = ["time"]
```

- `log.const` 是 `log.reg` 的别名，类型是 `Registry`，对应 `const.yaml`。
- `add_const()` 会把键值追加到 YAML 文件并立即保存。
- `add_const_to_head()` 会把键值插入到 YAML 顶部，适合放最重要的运行参数。
- `log.meta` 对应 `metadata.json`，主要用于和 LogBrowser 交互，例如标题、收藏、
  回收站状态、绘图轴等 GUI 展示相关的轻量状态。

## Registry

`Registry` 是基于 YAML 文件的轻量键值注册表。它支持使用 `/` 分隔路径访问嵌套字段：

```python
from logqbit.registry import Registry

reg = Registry("const.yaml")
reg["device/name"] = "sample-a"
print(reg["device/name"])
```

直接通过 `get()`、`set()` 或 `[]` 操作时，`Registry` 会在读写前后和文件同步。
如果要做一批本地修改，可以操作 `root`，最后手动保存：

```python
reg.root["operator"] = "alice"
reg.root["temperature"] = "300 K"
reg.save()
```

`reload()` 会在文件变化后重新读取磁盘内容。本地未保存的修改会被新的磁盘内容覆盖。
`undo()` 和 `redo()` 可以撤销或重做最近的保存快照。

## DataFrameBuffer

`DataFrameBuffer` 是 `LogFolder` 内部使用的低层组件，负责把追加进来的 dataframe
片段缓冲在内存里，并后台 autosave 到 feather 文件。普通用户通常不需要直接使用它；
优先使用 `LogFolder.add_row()`、`LogFolder.flush()` 和 `LogFolder.close()`。

如果需要单独使用 dataframe 缓冲，可以这样写：

```python
import pandas as pd

from logqbit.dataframe import DataFrameBuffer

buffer = DataFrameBuffer("data.feather")
buffer.add_one_row({"x": 1.0, "y": 2.0})
buffer.add_multi_rows(pd.DataFrame({"x": [2.0, 3.0], "y": [4.0, 6.0]}))
buffer.flush()
buffer.close()
```

后台线程的状态机很小：等待数据变 dirty，等待当前 autosave interval 合并连续追加，
如果仍然 dirty 就写盘。`flush()` 会跳过等待，在调用线程同步写入。

## API Reference

::: logqbit.logfolder.LogFolder

::: logqbit.metadata.LogMetadata

::: logqbit.registry.Registry

::: logqbit.dataframe.DataFrameBuffer
