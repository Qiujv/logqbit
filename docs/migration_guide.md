# 从 LabRAD 迁移数据 Data Migration from LabRAD

本指南介绍如何将现有的 LabRAD 数据迁移到 logqbit 格式。

This guide explains how to migrate your existing LabRAD data to the logqbit format.

## 快速开始 Quick Start

### 1. 复制迁移模板 Copy the Migration Template

```bash
# 复制模板脚本到当前目录 / Copy the template script to your current directory
logqbit copy-template move_from_labrad

# 或指定输出位置 / Or specify an output location
logqbit copy-template move_from_labrad -o /path/to/your/project/
```

### 2. 编辑配置 Edit the Configuration

打开 `move_from_labrad.py` 并编辑顶部的配置部分：

Open `move_from_labrad.py` and edit the configuration section at the top:

```python
# 输入：你的 LabRAD 数据目录 / Input: Your LabRAD data directory
folder_in = Path("//moli/data")  # 改为你的数据路径 / Change this to your data path

# 输出目录将自动生成为 ./logqbit_{machine_name}
# Output directory will be auto-generated as ./logqbit_{machine_name}
# 或者你可以直接指定：/ Or you can specify it directly:
# folder_out = Path("./my_converted_data").resolve()
```

### 3. 运行迁移 Run the Migration

```bash
# 作为脚本运行 / If running as a script
python move_from_labrad.py

# 或在 Jupyter/IPython 中（文件包含 cell 标记）
# Or in Jupyter/IPython (the file contains cell markers)
jupyter notebook move_from_labrad.py
```

## 转换内容 What Gets Converted

迁移脚本将：

The migration script will:

- ✅ 将 `.csv` 数据文件转换为 `.feather` 格式（zstd 压缩）/ Convert `.csv` data files to `.feather` format (compressed with zstd)
- ✅ 保留 `const.yaml` 配置 / Preserve `const.yaml` configuration
- ✅ 提取元数据（标题、标签、绘图轴）到 `metadata.json` / Extract metadata (title, tags, plot axes) to `metadata.json`
- ✅ 从 `session.ini` 保留星标/回收站标签 / Preserve star/trash tags from `session.ini`
- ✅ 记录创建时间和机器名 / Record creation time and machine name
- ✅ 支持恢复中断的迁移 / Support resuming interrupted migrations

## 目录结构 Directory Structure

### 迁移前（LabRAD 格式）Before (LabRAD format):
```
data/
├── experiment.dir/
│   ├── 00001 - my experiment.csv
│   ├── 00002 - another run.csv
│   └── session.ini
└── another.dir/
    └── ...
```

### 迁移后（logqbit 格式）After (logqbit format):
```
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

## 高级用法 Advanced Usage

### 恢复中断的迁移 Resume Interrupted Migration

脚本会自动检测最后转换的文件并从那里恢复。如果迁移被中断，只需再次运行脚本。

The script automatically detects the last converted file and resumes from there. Just run the script again if it was interrupted.


## 故障排除 Troubleshooting

### "模板未找到"错误 "Template not found" error

确保正确安装了 logqbit：

Make sure you've installed logqbit correctly:
```bash
pip install --upgrade logqbit
```

## 迁移后 After Migration

迁移完成后，可以使用 Log Browser 打开转换后的数据：

Once migration is complete, you can open your converted data with the log browser:

```bash
logqbit browser ./logqbit_moli/experiment
```

或在 Python 中：

Or in Python:
```python
from logqbit import LogFolder

log = LogFolder("./logqbit_moli/experiment")
# 像往常一样访问数据 / Access your data as usual
```
