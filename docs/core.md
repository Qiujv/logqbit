# Core API

大多数程序只需要 `LogFolder`。它代表一条实验记录，并把数据、常量和浏览器使用的信息保存在同一个目录中。

## 创建和写入记录

```python
from logqbit import LogFolder

log = LogFolder.new("./runs", title="frequency sweep")
log.add_row(frequency=4.0, signal=0.12)
log.add_row(frequency=4.5, signal=0.18)
log.add_const(sample="device-a", temperature="20 mK")
log.meta.plot_axes = ["frequency"]
log.meta.plot_fields = ["signal"]
```

`LogFolder.new()` 会在父目录下创建下一个数字 ID。新增数据会在后台自动保存；如果下一步必须立即读取已经写入磁盘的数据，调用 `log.flush()`。

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

## 读取记录

```python
from logqbit import LogFolder

log = LogFolder("./runs/0", create=False)
df = log.df
print(df.head())
```

只需要读取已保存的数据时，也可以直接使用 pandas：

```python
import pandas as pd

df = pd.read_feather("./runs/0/data.feather")
```

## 常量和绘图设置

`log.const` 用于保存实验常量，支持普通键和以 `/` 分隔的嵌套键：

```python
log.add_const(operator="alice")
log.const["instrument/name"] = "scope-a"
```

`log.meta` 用于记录标题、星标和 Browser 的绘图选择：

```python
log.meta.title = "frequency sweep"
log.meta.star = 1
log.meta.plot_axes = ["frequency"]
log.meta.plot_fields = ["signal"]
log.meta.plot_groupby = ["device"]
```

多个设置可一次更新：

```python
log.meta.update(
    title="frequency sweep",
    plot_axes=["frequency"],
    plot_fields=["signal"],
)
```

## Registry

`Registry` 是一个独立的 YAML 键值存储，适合保存配置、实验常量或其他结构化参数。`LogFolder.const` 就是一个 Registry。

```python
from logqbit import Registry

reg = Registry("config.yaml")
reg["experiment/operator"] = "alice"
reg["experiment/temperature"] = "20 mK"

print(reg["experiment/operator"])
```

带 `/` 的键会自动创建嵌套 YAML 结构。通过 `[]`、`get()` 和 `set()` 读写时，Registry 会与文件同步，适合少量独立修改。

需要批量修改时，直接编辑 `root`，最后调用一次 `save()`：

```python
reg = Registry("config.yaml")
settings = reg.root
settings["operator"] = "alice"
settings["device"] = {"name": "scope-a", "channel": 1}
reg.save()
```

`root` 的修改只保留在当前实例中；调用 `reload()` 或使用另一个实例写入同一文件前，应先 `save()`。同一个 YAML 文件也应避免同时被多个 writer 修改，否则后保存的内容可能覆盖先保存的内容。

## 参数扫描

`capture()` 适用于规则的参数扫描。可迭代参数会依次扫描，标量参数作为固定条件传入测量函数：

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

## 浏览已有记录

`LogCatalog` 用于读取一个目录下的记录摘要，适合批量检查而不需要写入数据的场景：

```python
from logqbit import LogCatalog

records = LogCatalog().refresh("./runs")
for record in records:
    print(record.log_id, record.title, record.row_count)
```

使用 [LogBrowser](browser.md) 可在图形界面中完成同样的浏览、筛选和绘图工作。

## API Reference

::: logqbit.logfolder.LogFolder

::: logqbit.registry.Registry

::: logqbit.catalog.LogCatalog

::: logqbit.catalog.LogRecord

::: logqbit.metadata.LogMetadata
