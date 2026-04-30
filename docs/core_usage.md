# 核心用法

## `LogFolder`

`LogFolder` 是最常用的入口，用于在一个日志目录中维护：

- `data.feather` 数据文件
- `metadata.json` 元数据
- `const.yaml` 常量配置

### 创建新日志目录

```python
from pathlib import Path

from logqbit.logfolder import LogFolder

project_dir = Path("runs")
project_dir.mkdir(exist_ok=True)

log = LogFolder.new(project_dir, title="Resonator Scan")
```

### 写入数据

```python
log.add_row(freq=5.01, amp=0.12)
log.add_row(freq=5.02, amp=0.15)
log.flush()
```

`add_row()` 同时支持标量行和等长数组输入。

### 读取数据

```python
df = log.df
print(df.tail())
```

### 设置元数据

```python
log.meta.title = "Resonator Scan"
log.meta.star = 1
log.meta.plot_axes = ["freq"]
```

### 设置常量配置

```python
log.const["temperature"] = "20 mK"
log.const["attenuation_db"] = 60
```

`log.const` 是 `log.reg` 的别名，本质上是一个 `Registry`。

## `Registry`

`Registry` 用于管理 YAML 配置，适合存放实验参数、层级配置和附加说明。

### 基本读写

```python
from logqbit.registry import Registry

reg = Registry("const.yaml")
reg["instrument/source"] = "SGS100A"
reg["instrument/power_dbm"] = -30

print(reg["instrument/source"])
```

### 本地批量修改

```python
reg.reload()
reg.root["sweep"] = {"start": 4.0, "stop": 8.0}
reg.save()
```

直接操作 `root` 时属于本地修改，需要手动 `save()`。

## 何时适合只安装核心能力

如果你只需要：

- 在脚本中写入实验数据
- 维护 `const.yaml` 和 `metadata.json`
- 在现有环境里读取和分析日志目录

那么通常不需要立即安装 GUI 组件。

如果你需要交互式浏览器或实时绘图，再安装 `PySide6` 和 `pyqtgraph` 即可。