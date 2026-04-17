"""
PColorMeshItem 2D 绘图 — pyqtgraph 原生非均匀网格 + OpenGL 渲染
"""
import sys
import time

import numpy as np
import pyqtgraph as pg
from numba import njit, prange
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


@njit(cache=True)
def is_lexsorted(x, y):
    """x 全局升序；y 在每列内单调（升或降均可，各列独立）。"""
    N = len(x)
    prev_x = x[0]
    prev_y = y[0]
    y_dir = 0  # 0=未确定, 1=升序, -1=降序；每列重置
    for i in range(1, N):
        xi = x[i]
        if xi < prev_x:
            return False
        if xi == prev_x:
            yi = y[i]
            if yi > prev_y:
                if y_dir == -1:
                    return False
                y_dir = 1
            elif yi < prev_y:
                if y_dir == 1:
                    return False
                y_dir = -1
            prev_y = yi
        else:
            y_dir = 0  # 新列，重置方向
            prev_x = xi
            prev_y = y[i]
    return True


@njit(parallel=True, cache=True)
def _build_grids_rect(ys, zs, col_starts, col_sizes, max_ny, nx_col,
                      top_y, step_c):
    """各列独立并行，直接构建矩形色块的最终数组。
    z_final: (max_ny, 2*nx_col-1)  奇数列为分隔列，保持 NaN 不渲染
    y_final: (max_ny+1, 2*nx_col)  每数据列左右角点取相同 y → 上下边水平 → 矩形
    """
    z_final = np.full((max_ny, 2 * nx_col - 1), np.nan)
    y_final = np.empty((max_ny + 1, 2 * nx_col))

    for c in prange(nx_col):
        s = col_starts[c]
        n = col_sizes[c]
        c2 = c + c  # 偶数列索引

        for r in range(n):
            z_final[r, c2] = zs[s + r]
            yv = ys[s + r]
            y_final[r, c2] = yv
            y_final[r, c2 + 1] = yv

        top = top_y[c]
        y_final[n, c2] = top
        y_final[n, c2 + 1] = top

        sc = step_c[c]
        for r in range(n + 1, max_ny + 1):
            val = top + (r - n) * sc
            y_final[r, c2] = val
            y_final[r, c2 + 1] = val

    return z_final, y_final

data_shape = (1000, 10000)  # sort(SKIP): 0.010s | col_grp: 0.014s | extrap: 0.000s | build_rect(nb): 0.137s | TOTAL: 0.162s
# data_shape = (1000, 1000)  # sort(SKIP): 0.001s | col_grp: 0.002s | extrap: 0.000s | build_rect(nb): 0.010s | TOTAL: 0.013s
# data_shape = (1000, 100)  # sort(SKIP): 0.000s | col_grp: 0.000s | extrap: 0.000s | build_rect(nb): 0.001s | TOTAL: 0.001s

# Sorting time: 0.772, 0.056, 0.005 seconds for 10M, 1M, 100K points respectively.

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.status_label = QLabel("准备就绪")
        layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.useOpenGL(True)
        layout.addWidget(self.plot_widget)

        self.generate_data(*data_shape)
        self.plot()
        self.plot()  # 第二次plot耗时明显减少，主要是 OS 内存页面已映射（第一次分配大数组触发 page fault）、CPU cache 热身、Python 字节码已编译。

    def generate_data(self, nx, ny):
        self.status_label.setText(f"生成 {nx}×{ny} = {nx*ny:,} 个数据点...")
        QApplication.processEvents()

        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 10, ny)[::-1]
        xx, yy = np.meshgrid(x, y)
        yy = yy + 0.5 * xx + 0.1

        z = np.sin(xx) * np.cos(yy) + 0.1 * np.random.randn(ny, nx)

        self.x_data = xx.T.ravel()[:-99]
        self.y_data = yy.T.ravel()[:-99]
        self.z_data = z.T.ravel()[:-99]

        self.status_label.setText(f"数据已生成：{len(self.x_data):,} 个点")

    def plot(self):
        self.plot_widget.clear()

        num_points = len(self.x_data)
        self.status_label.setText(f"PColorMesh 绘制中...（{num_points:,} 个点）")
        QApplication.processEvents()

        start_time = time.time()
        t = time.perf_counter

        x_data, y_data, z_data = self.x_data, self.y_data, self.z_data
        N = len(x_data)

        # ── 1. 排序（numba 单趟早退检测）──
        t0 = t()
        is_sorted = is_lexsorted(x_data, y_data)
        if is_sorted:
            xs, ys, zs = x_data, y_data, z_data
        else:
            sort_idx = np.lexsort((y_data, x_data))
            xs, ys, zs = x_data[sort_idx], y_data[sort_idx], z_data[sort_idx]
        t1 = t()

        # ── 列分组 ──
        change = np.empty(N, dtype=np.bool_)
        change[0] = True
        change[1:] = xs[1:] != xs[:-1]
        xu = xs[change]
        col_starts = np.flatnonzero(change)
        nx_col = len(xu)
        col_ends = np.empty(nx_col, dtype=col_starts.dtype)
        col_ends[:-1] = col_starts[1:]
        col_ends[-1] = N
        col_sizes = col_ends - col_starts
        max_ny = int(col_sizes.max())

        ref_col = int(np.argmax(col_sizes))
        ref_y = ys[col_starts[ref_col]:col_ends[ref_col]]
        typical_dy = float(np.median(np.diff(ref_y))) if len(ref_y) > 1 else 1.0
        t2 = t()

        # ── 2. 计算每列外推参数 ──
        last_y = ys[col_ends - 1]
        prev_idx = np.maximum(col_ends - 2, col_starts)
        step_c = np.where(col_sizes > 1, last_y - ys[prev_idx], typical_dy)
        top_y = last_y + step_c

        # x corners（zero-copy broadcast，仅需 1D 数组）
        x_edges = np.empty(nx_col + 1)
        x_edges[:nx_col] = xu
        x_edges[-1] = xu[-1] + (xu[-1] - xu[-2] if nx_col > 1 else 1.0)
        x_edges_rect = np.repeat(x_edges, 2)[1:-1]          # shape (2*nx_col,)
        final_x = np.broadcast_to(x_edges_rect, (max_ny + 1, 2 * nx_col))
        t3 = t()

        # ── 3. numba: 一次遍历直接构建矩形网格的最终 z / y 数组 ──
        final_z, final_y = _build_grids_rect(
            ys, zs, col_starts, col_sizes, max_ny, nx_col,
            top_y, step_c)
        t4 = t()

        prepare_time = time.time() - start_time
        skipped = "SKIP" if is_sorted else "SORT"
        print(f"  sort({skipped}): {t1-t0:.3f}s | col_grp: {t2-t1:.3f}s | "
              f"extrap: {t3-t2:.3f}s | build_rect(nb): {t4-t3:.3f}s | "
              f"TOTAL: {prepare_time:.3f}s")

        # ── 4. 创建 PColorMeshItem ──
        cmap = pg.colormap.get('viridis', source='matplotlib')
        pcm = pg.PColorMeshItem(final_x, final_y, final_z, colorMap=cmap)
        self.plot_widget.addItem(pcm)

        plot_item = self.plot_widget.getPlotItem()
        plot_item.enableAutoRange(enable=True)
        plot_item.autoRange()

        elapsed = time.time() - start_time
        self.status_label.setText(
            f"PColorMesh 完成：{num_points:,} 点 → {nx_col}×{max_ny} mesh，"
            f"准备 {prepare_time:.3f}s，总计 {elapsed:.3f}s"
        )



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PColorMesh 2D Plot (OpenGL)")
        self.setGeometry(100, 100, 1400, 900)
        self.setCentralWidget(PlotWidget())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
