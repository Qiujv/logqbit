# LogBrowser 使用指南

通过以下命令启动数据浏览器 LogBrowser：

```bash
logqbit browser
```

请确保已安装 GUI extra：

```bash
pip install "logqbit[gui]"
```

你也可以通过以下命令来创建对应的桌面快捷方式：

```bash
logqbit shortcuts
```

调试时可以使用 `logqbit browser ./runs --foreground`，让 Browser 在当前进程中运行并
保持终端连接。


主界面分左右两部分：左侧记录列表，右侧显示选中的记录详情。
界面各处都设有右键菜单。

## 记录列表

Browser 把含有 `metadata.json` 的直接子目录识别为有效记录。
左侧列表显示 ID、标题、行数和绘图轴，右键表头可以启用更多信息列，单击表头可以按列排序。
双击记录会打开独立的 detail 窗口；
多选记录后可以通过右键菜单批量设置星标、回收站状态或导出记录。

常用操作也绑定了快捷键如下：

| 快捷键 | 操作 |
| --- | --- |
| `F5` | 刷新界面信息 |
| `F2` | 修改标题 |
| `S` | 切换一颗星 |
| `0`–`3` | 设置星标数量 |
| `T` | 切换回收站标记 |
| `Delete` | 将目录移入系统回收站 |
| `←` / `→` | 切换 detail 标签页 |
| `Ctrl + Enter` | 打开记录文件夹 |

目录中新增或移除记录时，列表会自动刷新。
选中记录的内容修改也会触发记录列表和详情页得刷新；
你也可以点击右上角的 `Refresh`按钮或者按`F5`键刷新整个界面。

## 详情页

右侧详情页包含三个固定标签页：

- `Const.`：查看 `const.yaml`。
- `Data`：查看 dataframe。初始只渲染前 100 行，点击 `Show More Rows` 每次再显示
  1000 行，避免大表格一次创建太多界面元素。
- `Plot`：根据数据列和 metadata 绘图。

记录目录中的常见图片文件在额外标签页中展示；其他文件可以通过 `Files...` 菜单打开。

## 选择绘图列

Plot 顶部的 TagBar 由两个分隔符划分为三段：

```text
axes | fields | ignored
```

拖动列名即可改变用途，图像会立即更新。右键点击 TagBar 并选择 `Save`，会把当前
`plot_axes` 和 `plot_fields` 写入 `metadata.json`。如果 metadata 没有提供有效选择，
TagBar 会按原始列顺序尽量取第一列作为 axis、第二列作为 field，让 Plot 页先有图可画。

- 一个 axis 加一个或多个 fields 会绘制 1D 曲线。
- 两个或更多 axes 加至少一个 field 会绘制 2D 色图；只使用前两个 axes 和第一个 field。

## 1D 拟合


绘制 1D 曲线时才会启用 `fit exp` 和 `fit x²`：

1. 点击拟合按钮进入对应模式，再次点击按钮、按 `Esc` 或单击鼠标右键可退出。
2. 用鼠标左键拖出矩形，框选参与拟合的数据点。
3. 图中会显示选中点、拟合曲线和结果。

指数拟合显示衰减时间 `τ`，二次拟合显示极值位置 `x`。

## cursor 与截面

点击 `cursor` 按钮会显示一个可拖动的游标，会展示绘图数据在游标处的值。
再次点击按钮即可关闭 cursor。

在 2D 色图还会同时展示水平、垂直截面。

为了给截面留出空间，cursor 启用期间 colorbar 会隐藏；关闭 cursor 后恢复。

## 复制图像

`Copy plot` 会把当前主绘图区复制到系统剪贴板。
