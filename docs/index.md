# LogQbit 文档 Documentation

欢迎来到 LogQbit 文档！

Welcome to LogQbit documentation!

## 目录 Table of Contents

### 入门指南 Getting Started

- [README](../README.md) - 项目概述和快速开始 / Project overview and quick start
- [安装 Installation](../README.md#安装-installation) - 安装说明 / Installation instructions

### 指南 Guides

- [迁移指南 Migration Guide](migration_guide.md) - 从 LabRAD 迁移数据 / Migrate data from LabRAD format

### 命令行工具 Command-Line Tools

LogQbit 提供以下命令行工具：

LogQbit provides the following command-line tools:

#### GUI 工具 GUI Tools

- **`logqbit-browser [directory]`** - 启动交互式日志浏览器 / Launch the interactive log browser
  - 浏览和分析记录的实验数据 / Browse and analyze logged experimental data
  - 可视化数据表格和图表 / Visualize data tables and plots
  - 管理元数据（星标、标题等）/ Manage metadata (stars, titles, etc.)

- **`logqbit-live-plotter`** - 启动实时绘图窗口 / Launch the live plotting window
  - 实时可视化正在运行的实验 / Real-time visualization of running experiments
  - 支持多轴绘图 / Support for multi-axis plotting
  - 自动刷新数据流 / Automatic data stream refresh

#### 实用工具 Utilities

- **`logqbit browser-demo`** - 创建示例数据并启动浏览器 / Create example data and launch browser
  - 快速体验浏览器功能 / Quick demo of browser features
  - 创建 3 个示例日志文件夹 / Creates 3 example log folders:
    - `0` - 线性关系：`y = 2x + 1`, `z = x²` / Linear relationship: `y = 2x + 1`, `z = x²`
    - `1` - 带噪声的正弦信号 / Noisy sinusoidal signal
    - `2` - 2D 参数扫描（共振）/ 2D parameter scan (resonance)
  - 自动启动浏览器显示示例 / Automatically launches browser to display examples
  - 示例 / Example:
    ```bash
    logqbit browser-demo
    # 浏览器将自动打开 / Browser will automatically open
    ```

- **`logqbit copy-template <name>`** - 复制模板到工作目录 / Copy templates to working directory
  - 可用模板 / Available templates:
    - `move_from_labrad` - LabRAD 数据迁移脚本 / LabRAD data migration script
  - 示例 / Example:
    ```bash
    logqbit copy-template move_from_labrad
    logqbit copy-template move_from_labrad -o /path/to/output/
    ```

- **`logqbit browser [directory]`** - 启动浏览器的简短命令 / Short command to launch browser
  - 等同于 `logqbit-browser` / Equivalent to `logqbit-browser`

- **`logqbit shortcuts`** - 创建桌面快捷方式 / Create desktop shortcuts
  - 在桌面创建带自定义图标的快捷方式 / Create shortcuts on desktop with custom icons
  - 快捷方式 / Shortcuts created:
    - `LogQbit Browser.lnk` - 日志浏览器 / Log browser
    - `LogQbit Live Plotter.lnk` - 实时绘图工具 / Live plotter
  - 每个快捷方式配有对应的 SVG 图标 / Each shortcut has its corresponding SVG icon
  - 示例 / Example:
    ```bash
    # 在桌面创建快捷方式 / Create shortcuts on desktop
    logqbit shortcuts
    
    # 在指定目录创建 / Create in specific directory
    logqbit shortcuts -o "C:\MyShortcuts"
    ```

### API 参考 API Reference

（待补充）

(To be added)

### 数据格式 Data Format

（待补充）

(To be added)

## 贡献 Contributing

欢迎贡献！请访问 [GitHub 仓库](https://github.com/Qiujv/logqbit)。

Contributions are welcome! Please visit the [GitHub repository](https://github.com/Qiujv/logqbit).

## 许可证 License

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
