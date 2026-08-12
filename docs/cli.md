# 命令行工具

运行 `logqbit --help` 可查看当前可用命令。

## 打开 LogBrowser

```bash
logqbit browser ./runs
```

目录参数可省略。通常 Browser 会在独立进程中启动，不占用终端；需要在当前终端运行时使用：

```bash
logqbit browser ./runs --foreground
```

## 创建示例数据

```bash
logqbit browser-demo
```

命令会在当前目录的 `logqbit_example/` 中添加示例记录，并打开 Browser。重复运行会追加新的示例记录。

## 复制迁移模板

```bash
logqbit copy-template move_from_labrad
logqbit copy-template move_from_labrad -o ./tools/
```

该模板用于将已有 LabRAD 数据迁移为 LogQbit 记录；复制后请按自己的目录结构编辑。

## Windows 快捷方式

```bash
logqbit shortcuts
logqbit shortcuts -o "C:\MyShortcuts"
```

这会创建 LogBrowser 的 Windows 快捷方式。该命令需要安装 GUI extra。
