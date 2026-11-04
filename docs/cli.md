# 命令行工具

运行 `logqbit --help` 可查看当前可用命令。

## 打开离线文档

```bash
logqbit docs
```

该命令会在默认浏览器中打开随 LogQbit 安装的离线文档，无需联网或安装 MkDocs。

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

## 格式化历史 metadata

新写入的 `metadata.json` 使用可读的 Unicode 和四空格缩进，方便直接搜索和查看文件内容。旧文件仍可正常读取；如需一次转换历史记录，可运行：

```bash
logqbit format-metadata-json ./runs --dry-run
logqbit format-metadata-json ./runs
```

命令递归扫描指定目录下的 `metadata.json`，包括指定目录本身，不跟随符号链接。预览只报告结果，不修改文件；正式运行会保留未知字段和文件权限，对已符合格式的文件不做修改。无效 JSON、重复键或无法无损转换的数值会被跳过并报告；其他文件继续处理，有错误时命令返回非零状态。

转换会修改文件内容和修改时间。运行前请备份重要数据并暂停对这些记录的写入：命令会检测部分并发修改，但不能保证避免所有并发写入冲突。

## Windows 快捷方式

```bash
logqbit shortcuts
logqbit shortcuts -o "C:\MyShortcuts"
```

这会创建 LogBrowser 的 Windows 快捷方式。该命令需要安装 GUI extra。
