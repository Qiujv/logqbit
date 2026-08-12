# 从 LabRAD 迁移

LogQbit 附带一个 LabRAD 迁移模板。它是可修改的起点，请先用一小部分数据试运行，再处理完整数据集。

## 使用模板

复制模板到当前目录或指定目录：

```bash
logqbit copy-template move_from_labrad
logqbit copy-template move_from_labrad -o ./tools/
```

打开复制出的 `move_from_labrad.py`，根据自己的数据位置和输出位置修改开头的配置，然后运行：

```bash
python move_from_labrad.py
```

## 迁移结果

每条数据会成为一个 LogQbit 记录目录，通常包含：

```text
experiment/
└── 1/
    ├── data.feather
    ├── const.yaml
    └── metadata.json
```

迁移后的目录可直接打开：

```bash
logqbit browser ./logqbit_data/experiment
```

如果迁移中断，可重新运行模板。开始完整迁移前，建议保留原始数据并确认少量结果的表格、常量和绘图显示符合预期。
