# LogBrowser 使用指南

LogBrowser 用来浏览一组 `LogFolder` 记录，查看数据、常量和附加文件，并直接绘制一维或
二维数据。先安装 GUI extra：

```bash
pip install "logqbit[gui]"
```

然后把记录的父目录传给 Browser：

```bash
logqbit browser ./runs
```

`logqbit-browser ./runs` 是同一个界面的独立入口。未指定目录时，Browser 会打开最近使用
的目录；首次运行则使用当前目录。

需要让命令立即返回、不占用当前终端时，可以使用 `logqbit browser ./runs --detach` 或
`logqbit-browser-detached ./runs`。Windows 桌面快捷方式默认使用这一启动模式。

## 记录列表

Browser 把含有 `metadata.json` 的直接子目录识别为记录。左侧列表显示 ID、标题、行数和
绘图轴，可以按列排序。双击记录会打开独立的 detail 窗口；多选记录后可以通过右键菜单
批量设置星标、回收站状态或导出记录。

右键菜单还可以：

- 修改标题、显示隐藏列。
- 在文件管理器中打开记录目录。
- 将记录复制到另一个目录并重新编号。
- 将记录目录移入系统回收站。

常用快捷键如下：

| 快捷键 | 操作 |
| --- | --- |
| `F2` | 修改标题 |
| `S` | 切换一颗星 |
| `0`–`3` | 设置星标数量 |
| `T` | 切换回收站标记 |
| `Delete` | 将目录移入系统回收站 |
| `←` / `→` | 切换 detail 标签页 |

父目录新增或移除记录时，列表会自动刷新。当前记录的文件变化也会刷新 detail view 和对应
的行数；需要立即重新扫描全部状态时，可以点击主界面的 `Refresh`。

## Detail view

右侧 detail view 包含三个固定标签页：

- `Const.`：查看 `const.yaml`。
- `Data`：查看 dataframe。初始只渲染前 100 行，点击 `Show More Rows` 每次再显示
  1000 行，避免大表格一次创建太多界面元素。
- `Plot`：根据数据列和 metadata 绘图。

记录目录中的常见图片会成为额外标签页；其他文件可以通过 `Files...` 菜单打开。Detail
view 顶部的 watch 开关控制是否跟随当前记录的文件变化。

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

## 1D 拟合与 cursor

`cursor` 按钮会显示一条可拖动竖线。拖动期间隐藏读数，松开后吸附到最近的数据位置，并
显示各条曲线在该位置的值。再次点击按钮即可关闭 cursor。

只有恰好一个 field 时才会启用 `fit exp` 和 `fit x²`：

1. 点击拟合按钮进入对应模式。
2. 用鼠标左键拖出矩形，框选参与拟合的数据点。
3. 图中会显示选中点、拟合曲线和结果。

指数拟合显示衰减时间 `τ`，二次拟合显示极值位置 `x`。完成一次拟合后仍保持在同一拟合
模式，可以继续框选；再次点击按钮、按 `Esc` 或单击鼠标右键可退出。拟合和 cursor
互斥，启用其中一个会关闭另一个。

## 2D cursor 与截面

2D 图默认显示 colorbar。启用 `cursor` 后会出现横线、竖线和可同时移动两个方向的中心
方块，同时显示当前位置的数值以及水平、垂直截面。拖动期间读数和截面暂时隐藏，松开后
更新到最近的有效数据点。

为了给截面留出空间，cursor 启用期间 colorbar 会隐藏；关闭 cursor 后恢复。

## 复制图像

`Copy plot` 会把当前主绘图区复制到系统剪贴板。复制出的图片顶部会临时加入当前记录的
完整路径，界面本身不会因此改变标题或布局。
