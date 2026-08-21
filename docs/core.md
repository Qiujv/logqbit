# 核心 API

一条 LogQbit 记录是一个目录，通常包含三个标准文件：

```text
0/
├── data.feather
├── metadata.json
└── const.yaml
```

核心 API 从三个角度操作这些记录：

| API | 用途 |
| --- | --- |
| `LogFolder` | 创建记录、追加 DataFrame 数据并自动保存 |
| `LogRecord` | 查看已有记录，读取数据，访问/修改metadata 和 constants |
| `LogCatalog` | 扫描发现记录，批量获得和刷新 `LogRecord` |

## 创建和写入：LogFolder

```python
from logqbit import LogFolder

log = LogFolder.new("./runs", title="frequency sweep")
log.add_row(frequency=4.0, signal=0.12)
log.add_row(frequency=4.5, signal=0.18)
log.add_const(sample="device-a", temperature="20 mK")
log.meta.plot_axes = ["frequency"]
log.meta.plot_fields = ["signal"]
```

`LogFolder.new()` 会在父目录下创建下一个数字 ID。新增数据会在后台自动保存；如果
下一步必须立即读取已经写入磁盘的数据，调用 `log.flush()`。

`add_row()` 接受一行标量值，也可接受等长数组一次写入多行：

```python
log.add_row(
    frequency=[4.0, 4.5, 5.0],
    signal=[0.12, 0.18, 0.15],
)
```

已有 pandas DataFrame 可直接追加：

```python
import pandas as pd

log.add_df(pd.DataFrame({"frequency": [5.5, 6.0], "signal": [0.10, 0.08]}))
```

### 写入注意事项

一个记录目录在同一时刻只能有一个 writer。不要同时用多个 `LogFolder` 实例或进程写入同一路径；每个实例都会保留自己的内存数据，之后保存时可能覆盖其他 writer 已写入的内容。

读取同一条记录没有这个限制。若需要将数据交给另一个进程或脚本读取，先调用 `log.flush()`，再读取 `data.feather`。

### 参数扫描

`capture()` 适用于规则的参数扫描。可迭代参数会依次扫描，标量参数作为固定条件
传入测量函数：

```python
def measure(frequency, bias):
    return {"signal": frequency * bias}


log = LogFolder.new("./runs", title="frequency sweep")
log.capture(
    measure,
    {"frequency": [4.0, 4.5, 5.0], "bias": 0.1},
)
```

测量函数应返回包含测量结果的字典；扫描坐标会自动一并写入。

## 打开已有记录：LogRecord

`LogRecord` 无法添加数据，也不会创建缺失的文件：

```python
from logqbit import LogRecord

record = LogRecord("./runs/0")

df = record.df
print(record.meta.title)
print(record.row_count, record.columns)
```

`record.df` 在首次访问时读取 `data.feather`，并缓存这个内存快照。修改返回的
DataFrame 只影响本地缓存，不会写回磁盘。如果记录没有 `data.feather`，它返回
`None`。

需要读取当前文件而不使用缓存时，调用：

```python
fresh_df = record.read_dataframe()
```

`refresh()` 会检查磁盘状态并原地刷新记录；当 `data.feather` 发生变化时，已经缓存的
`df` 会自动失效：

```python
record.refresh()
df = record.df
```

如果磁盘没有变化，但仍要丢弃或替换本地缓存，可以显式操作 cached property：

```python
del record.df
record.df = record.read_dataframe()
```

只需要表格数据、不需要记录的 metadata 或 constants 时，也可以直接使用 pandas：

```python
import pandas as pd

df = pd.read_feather("./runs/0/data.feather")
```

## Metadata、constants 和绘图设置

`LogFolder` 与 `LogRecord` 都通过 `meta` 访问 `metadata.json`：

```python
record.meta.title = "frequency sweep"
record.meta.star = 1
record.meta.plot_axes = ["frequency"]
record.meta.plot_fields = ["signal"]
record.meta.plot_groupby = ["device"]
```

多个设置可一次更新：

```python
record.meta.update(
    title="frequency sweep",
    plot_axes=["frequency"],
    plot_fields=["signal"],
)
```

`const` 是 `reg` 的别名，用于访问 `const.yaml` 中的实验常量和配置：

```python
print(record.const["sample"])
record.reg["instrument/name"] = "scope-a"
```

`LogRecord` 只访问已有文件；如果 `metadata.json` 或 `const.yaml` 不存在，访问对应
属性会抛出 `FileNotFoundError`。通过 `meta` 和 `const` 对已有文件赋值仍会同步到
磁盘，因此 `LogRecord` 不是严格不可写对象。

## 批量浏览：LogCatalog

`LogCatalog` 扫描父目录中含有 `metadata.json` 的直接子目录，并返回按 ID 排序的
`LogRecord`：

```python
from logqbit import LogCatalog

catalog = LogCatalog("./runs")
records = catalog.refresh()

for record in records:
    print(record.log_id, record.title, record.row_count)
```

Catalog 会复用同一路径的 `LogRecord`。再次调用 `refresh()` 时，已有对象会原地同步；
没有变化的 `data.feather` 不会被重复检查。

使用 [LogBrowser](browser.md) 可在图形界面中完成浏览、筛选和绘图。

## 单独使用 Registry

`Registry` 是一个独立的 YAML 键值存储，适合保存配置、实验常量或其他结构化参数。
`LogFolder.const` 和 `LogRecord.const` 都是 Registry。

```python
from logqbit import Registry

reg = Registry("config.yaml")
reg["experiment/operator"] = "alice"
reg["experiment/temperature"] = "20 mK"

print(reg["experiment/operator"])
```

带 `/` 的键会自动创建嵌套 YAML 结构。通过 `[]`、`get()` 和 `set()` 读写时，
Registry 会与文件同步，适合少量独立修改。

需要批量修改时，直接编辑 `root`，最后调用一次 `save()`：

```python
reg = Registry("config.yaml")
settings = reg.root
settings["operator"] = "alice"
settings["device"] = {"name": "scope-a", "channel": 1}
reg.save()
```

`root` 的修改只保留在当前实例中；调用 `reload()` 或使用另一个实例写入同一文件前，
应先 `save()`。同一个 YAML 文件也应避免同时被多个 writer 修改，否则后保存的内容
可能覆盖先保存的内容。

## API Reference

::: logqbit.logfolder.LogFolder

::: logqbit.catalog.LogRecord

::: logqbit.catalog.LogCatalog

::: logqbit.metadata.LogMetadata

::: logqbit.registry.Registry
