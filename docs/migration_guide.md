# 从 LabRAD 迁移数据

本指南介绍如何把已有的 LabRAD 数据目录迁移为 logqbit 的目录结构。

## 快速开始

### 1. 复制迁移模板

```bash
logqbit copy-template move_from_labrad
```

如果你想把模板放到指定目录：

```bash
logqbit copy-template move_from_labrad -o /path/to/your/project/
```

### 2. 编辑模板脚本

打开 `move_from_labrad.py`，修改顶部配置，例如：

```python
from pathlib import Path

folder_in = Path("//moli/data")
# folder_out = Path("./my_converted_data").resolve()
```

默认情况下，输出目录会自动生成类似 `./logqbit_{machine_name}` 的路径。

### 3. 运行迁移

```bash
python move_from_labrad.py
```

如果你习惯在 Jupyter/IPython 里执行，也可以直接打开这个脚本。

## 迁移后会得到什么

迁移脚本会执行以下转换：

- 将 `.csv` 数据转换为 `.feather`。
- 保留 `const.yaml` 中的常量配置。
- 生成 `metadata.json`，写入标题、标签、绘图轴等元数据。
- 从 `session.ini` 中保留星标和回收站标签。
- 记录创建时间与机器名。
- 支持中断后续跑。

## 目录结构示例

迁移前：

```text
data/
├── experiment.dir/
│   ├── 00001 - my experiment.csv
│   ├── 00002 - another run.csv
│   └── session.ini
└── another.dir/
    └── ...
```

迁移后：

```text
logqbit_moli/
├── experiment/
│   ├── 1/
│   │   ├── data.feather
│   │   ├── const.yaml
│   │   └── metadata.json
│   ├── 2/
│   │   ├── data.feather
│   │   ├── const.yaml
│   │   └── metadata.json
│   └── ...
└── another/
    └── ...
```

## 恢复中断任务

如果迁移过程中断，重新运行脚本即可。脚本会尝试检测上次处理到的位置并继续转换。

## 迁移完成后

你可以直接用浏览器打开迁移后的目录：

```bash
logqbit browser ./logqbit_moli/experiment
```

也可以在 Python 中访问：

```python
from logqbit.logfolder import LogFolder

log = LogFolder("./logqbit_moli/experiment", create=False)
print(log.df.head())
```
