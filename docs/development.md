# 开发者文档

本页记录文档系统的本地预览、构建和发布方式，方便维护者在修改 `docs/`、`README.md` 或 `mkdocs.yml` 时快速验证结果。

## 文档来源

当前文档站由以下内容组成：

- `docs/` 目录中的 Markdown 页面。
- 仓库根目录的 `mkdocs.yml` 导航与站点配置。
- `README.md` 中的项目简介与安装入口。

## 本地预览

先安装开发依赖：

```bash
uv sync --group dev
```

启动本地预览服务：

```bash
uv run --group dev mkdocs serve --dev-addr 127.0.0.1:8000
```

默认访问地址：

```text
http://127.0.0.1:8000/logqbit/
```

说明：

- 修改 `docs/`、`README.md` 或 `mkdocs.yml` 后，页面会自动重建。
- 当前地址带有 `/logqbit/` 前缀，是因为站点配置了 `site_url: https://qiujv.github.io/logqbit/`。

## 本地构建

如果只想验证静态构建结果，可以运行：

```bash
uv run --group dev mkdocs build --strict
```

构建产物会输出到仓库根目录下的 `site/`。

`--strict` 会把链接错误、缺页等问题提升为构建失败，适合在提交前做一次检查。

## GitHub Actions 与 Pages

仓库中当前有两类 workflow：

- `.github/workflows/publish.yml`：用于打 tag 后构建并发布 PyPI 包。
- `.github/workflows/docs.yml`：用于构建 MkDocs 站点并部署到 GitHub Pages。

`docs.yml` 会在以下场景触发：

- `main` 分支推送了文档相关改动。
- 在 GitHub 页面手动触发 `workflow_dispatch`。

工作流的大致步骤是：

- checkout 仓库
- 安装 Python 与 `uv`
- `uv sync --group dev`
- `uv run --group dev mkdocs build --strict`
- 上传 `site/` 并部署到 GitHub Pages

## 常见维护操作

### 新增一页文档

1. 在 `docs/` 下新增一个 Markdown 文件。
2. 在 `mkdocs.yml` 的 `nav` 中加入对应条目。
3. 本地运行 `mkdocs serve` 或 `mkdocs build --strict` 验证。

### 调整导航顺序

直接修改 `mkdocs.yml` 中的 `nav` 顺序即可。

### 更新首页入口

如果 README 中的安装方式、文档入口或示例变化较大，建议同步更新 `README.md`，保持 GitHub 首页与 Pages 站点一致。