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
from logqbit import LogFolder

log = LogFolder.new("./runs", title="cooldown")
log.add_row(time=0.0, temperature=300.0)
log.add_row(time=1.0, temperature=295.2)

log.add_const(operator="alice", sample="device-a")
log.meta.plot_axes = ["time"]
```

`LogFolder.new(parent)` 会在 `parent` 下创建下一个数字目录，例如 `0/`、`1/`、`2/`。
创建后可以直接持续追加数据；后台线程会自动保存，程序正常退出时也会完成最后一次写入。

### 打开已有记录

```python
log = LogFolder("./runs/0", create=False)
print(log.df)
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

也可以直接追加一个已有 dataframe：

```python
import pandas as pd

log.add_df(pd.DataFrame({"x": [3.0, 4.0], "y": [2.4, 2.8]}))
```

### 参数扫描

`capture()` 可以执行简单的参数扫描。可迭代参数会参与笛卡尔积扫描，标量参数会作为常量
传给测量函数：

```python
def measure(frequency, bias):
    return {"signal": frequency * bias}


log = LogFolder.new("./runs", title="frequency sweep")
log.capture(
    measure,
    {
        "frequency": [4.0, 4.5, 5.0],
        "bias": 0.1,
    },
)
```

测量函数返回的字典会和当前扫描坐标合并为数据行。`capture()` 还会把标量参数和扫描维度
写入 `const.yaml`；仅当 `plot_axes` 为空时，才自动使用扫描参数作为绘图轴。扫描完成后会
立即 flush。

### 读取和保存

```python
df = log.df
log.flush()
```

- `log.df` 返回当前完整 dataframe 的副本，包括还没有写入磁盘的缓冲行，但不会触发 flush。
- `log.flush()` 立即同步写入 `data.feather`，调用会阻塞直到写入完成。

没有追加过数据的记录不会创建空的 `data.feather`，但已经同步写入的
`metadata.json` 和 `const.yaml` 仍会保留。

每个 `LogFolder` 实例都有独立的 dataframe buffer、内存快照和 autosave 线程。buffer
创建时只读取一次已有的 `data.feather`；创建后只写入而不会重新读取。因此，不要同时用
多个实例或进程写入同一路径：任何其他 writer 造成的修改，都可能在该 buffer 下次写入
完整内存快照时被覆盖。

```python
first = LogFolder("./runs/0", create=False)
second = LogFolder("./runs/0", create=False)
first.add_row(time=2.0, temperature=291.4)
second.add_row(time=3.0, temperature=291.8)
```

上例中的两个实例互相看不到对方创建后的追加；两者都写盘时，最后一次写入会覆盖前一次
写入的完整 dataframe。

如果只需要读取已经写好的数据文件，最简单的方式是直接用 pandas：

```python
import pandas as pd

df = pd.read_feather("./runs/0/data.feather")
```

这适合做只读分析、导出脚本或不需要创建 `LogFolder` 对象的场景。

最后一个 buffer 的 Python 强引用消失时，LogQbit 会自动 flush 并停止对应的 autosave
线程；普通脚本自然退出时也会执行这项清理。IPython/Jupyter 的输出历史、异常 traceback
或回调可能继续持有对象，使自动清理延后。`with LogFolder(...)` 会在退出代码块时 flush，
但不会让该对象失效，之后仍可继续使用。如果需要立即确认数据已经写入，应主动调用
`flush()`。进程崩溃或被强制终止时不能依赖自动清理，因此重要阶段仍应主动 flush。

长期运行的交互式环境可以检查和关闭 autosave worker：

```python
workers = LogFolder.inspect_workers()
for worker in workers:
    print(worker.worker_id, worker.path, worker.owner_alive, worker.last_error)

# 按 worker ID 关闭，或不传参数关闭当前进程内的全部 worker。
failed = LogFolder.close_workers([workers[0].worker_id])
```

诊断快照还包含线程名称、线程 ID、存活状态和 dirty 状态。`owner_alive=False` 表示
`DataFrameBuffer` 包装对象已经被回收，但 finalizer 未能成功停止该 worker。人工关闭仍有
owner 的 worker 会使对应实例失效，之后继续使用会抛出 `RuntimeError`。关闭方法会尝试
所有选中的 worker，并返回未能成功关闭的最新状态。

### 常量

```python
log.add_const(temperature="300 K", bias=0.1)
log.add_const_to_head(run_group="calibration")

log.const["instrument/name"] = "scope-a"
```

- `log.const` 是 `log.reg` 的别名，类型是 `Registry`，对应 `const.yaml`。
- `add_const()` 会把键值追加到 YAML 文件并立即保存。
- `add_const_to_head()` 会把键值插入到 YAML 顶部，适合放最重要的运行参数。

## LogMetadata

`log.meta` 是一个 `LogMetadata`，对应记录目录中的 `metadata.json`。常用字段会在赋值时
立即保存，主要用于控制 LogBrowser 的显示和绘图：

```python
log.meta.title = "cooldown"
log.meta.star = 2
log.meta.trash = False
log.meta.plot_axes = ["time"]
log.meta.plot_fields = ["temperature"]
```

`plot_axes` 和 `plot_fields` 读取时是不可变 tuple；赋值时可以传一个字符串或任意字符串
序列。单个字符串会被当成一个列名，而不是拆成字符。

如果要一次更新多个字段，使用 `update()` 只写一次文件：

```python
from logqbit import LogMetadata

meta = LogMetadata("./runs/0/metadata.json", create=False)
meta.update(
    title="cooldown",
    star=1,
    plot_axes="time",
    plot_fields=["temperature", "pressure"],
)
```

`reload()` 会在磁盘文件发生变化时重新加载。直接创建的 `LogMetadata` 默认严格报告无效
JSON；Catalog 和 LogBrowser 则使用容错读取，避免单个损坏记录阻止整个列表打开。

## LogCatalog

`LogCatalog` 用于快速浏览一个父目录下的已有记录，不会创建 `LogFolder` 的写缓冲或
autosave 线程。每个 `LogRecord` 同时包含轻量的内存摘要，以及显式的磁盘读取入口。

Catalog 以 `metadata.json` 文件作为日志目录的标记，因此目录名不必是数字；没有
`metadata.json` 的普通子目录会被忽略。

```python
from logqbit import LogCatalog

catalog = LogCatalog()
records = catalog.refresh("./runs")

for record in records:
    print(record.log_id, record.title, record.row_count)
```

结果中数字目录排在前面并按数值递增，其他目录随后按名称排序。

`LogRecord` 缓存 dataframe 的行数、列名和文件版本；未变化的日志会复用同一个 record
实例。Metadata 直接来自 `record.meta` 当前已经加载的内存值。

```python
record = records[0]
df = record.read_dataframe()
```

`read_dataframe()` 是明确的一次读取，不保留隐藏的进程级缓存。LogBrowser 的 detail
view 会按窗口和文件版本缓存完整 dataframe，因此多个 detail 窗口的刷新状态互不影响。
记录还没有 `data.feather`，或数据读取失败时，返回 `None`。

```python
record.meta.update(
    title="cooldown",
    star=2,
    trash=False,
    plot_axes=["time"],
    plot_fields=["temperature"],
)
```

`LogFolder.meta` 和 `LogRecord.meta` 使用同一个 `LogMetadata` 接口。Catalog 和
LogBrowser 会在各自的刷新边界同步外部修改；client 代码也可以主动同步：

```python
record.meta.reload()
```

## Registry

`Registry` 是基于 YAML 文件的轻量键值注册表。它支持使用 `/` 分隔路径访问嵌套字段：

```python
from logqbit import Registry

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

## 低层组件：DataFrameBuffer

`DataFrameBuffer` 是 `LogFolder` 内部使用的低层组件，负责把追加进来的 dataframe
片段缓冲在内存里，并后台 autosave 到 feather 文件。普通用户通常不需要直接使用它；
优先使用 `LogFolder.add_row()` 和 `LogFolder.flush()`。

如果需要单独使用 dataframe 缓冲，可以这样写：

```python
import pandas as pd

from logqbit.dataframe import DataFrameBuffer

buffer = DataFrameBuffer("data.feather")
buffer.add_one_row({"x": 1.0, "y": 2.0})
buffer.add_multi_rows(pd.DataFrame({"x": [2.0, 3.0], "y": [4.0, 6.0]}))
buffer.flush()
```

通常不需要主动关闭：最后一个强引用消失后，buffer 会自动 flush 并停止线程。如果需要
确定性地释放这个低层对象，也可以调用 `buffer.close()`；关闭后的 buffer 不能继续使用。

后台线程的状态机很小：等待数据变 dirty，等待当前 autosave interval 合并连续追加，
如果仍然 dirty 就写盘。临时写入失败时会保留 dirty 状态并重试；`flush()` 会跳过等待，
在调用线程同步写入并直接报告错误。每次构造都会创建独立 buffer 和线程；
最后一个强引用消失时，它会自动 flush 并停止后台线程。指向同一路径的多个 buffer 不会
重新读取或合并彼此的数据，最后写入者会覆盖文件。

## API Reference

::: logqbit.logfolder.LogFolder

::: logqbit.catalog.LogCatalog

::: logqbit.catalog.LogRecord

::: logqbit.metadata.LogMetadata

::: logqbit.registry.Registry

::: logqbit.dataframe.DataFrameBuffer
