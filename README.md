# LogQbit

## 说明

*LogQbit* 是一个轻量且可扩展的实验数据记录工具包。
它最初用于记录 量子比特（qubit）实验测量数据，但凭借灵活的数据格式和实时可视化功能，同样适用于 任意中小规模（≤MB级）实验数据 的采集与管理。

*LogQbit* is a lightweight and extensible data logging toolkit for lab-scale experiments.
It was originally developed for recording quantum qubit measurement data, 
but its flexible format and real-time visualization tools make it suitable for any small to medium (≤MB-level) experimental dataset.

通过使用 `logqbit`，你可以：

- 以最少的样板代码记录结构化实验数据；

- 使用集成的实时绘图工具可视化数据流；

- 通过交互式日志浏览器查看与分析记录的数据。

With `logqbit`, you can:

- Record structured experimental data with minimal boilerplate.

- Visualize data streams in real time with an integrated live plotter.

- Browse and analyze logged results through an interactive log browser.

无论是量子比特读出、参数扫描，还是传感器输出记录，
`logqbit` 都能为你提供一个简洁而可靠的实验数据采集与回溯工作流。

Whether you are monitoring qubit readouts, scanning a parameter sweep, or simply logging sensor outputs, 
`logqbit` provides a simple and robust workflow for capturing and revisiting your experimental data.


## 安装 Python 项目依赖

在该仓库目录下，运行如下指令以安装依赖的第三方python库

```
pip install -r requirements.txt
```