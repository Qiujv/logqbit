"""Tests for plotter module helper functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QSizePolicy, QToolButton

from logqbit.catalog import PlotColumns, resolve_plot_columns
from logqbit.gui.browser.plot.fitting import fit_exponential, fit_quadratic
from logqbit.gui.browser.plot.grouping import iter_plot_groups
from logqbit.gui.browser.plot.mesh import (
    _build_grids_rect,
    _is_lexsorted,
    build_plot_mesh,
    warmup_plotter_jit,
)
from logqbit.gui.browser.plot.manager import (
    COLOR_BAR_HEIGHT_FACTOR,
    PLOT_AUTO_RANGE_PADDING,
    PlotManager,
    TagBar,
)


class TestTagBar:
    def test_set_columns_deduplicates_without_reordering(self) -> None:
        tag_bar = TagBar()

        tag_bar.set_columns(
            ["x", "y", "signal", "reference"],
            ["y", "x", "y"],
            ["reference", "y", "signal", "reference"],
            [],
        )

        assert tag_bar.axes == ["y", "x"]
        assert tag_bar.fields == ["reference", "signal"]
        assert tag_bar.groupby == []
        assert tag_bar._split()[2] == []

    def test_set_columns_uses_first_ignored_column_when_fields_are_empty(
        self,
    ) -> None:
        tag_bar = TagBar()

        tag_bar.set_columns(
            ["x", "signal", "reference"],
            ["x", "x"],
            [],
            [],
        )

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert tag_bar.groupby == []
        assert tag_bar._split()[2] == ["reference"]

    def test_set_columns_reserves_groupby_before_default_roles(self) -> None:
        tag_bar = TagBar()

        tag_bar.set_columns(
            ["device", "x", "signal"],
            [],
            [],
            ["device"],
        )

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert tag_bar.groupby == ["device"]
        assert tag_bar._split()[2] == ["device"]
        assert tag_bar.groupby_button.text() == "group by (1)"
        assert tag_bar._groupby_checks["device"].isChecked()

    def test_groupby_dropdown_moves_columns_out_of_plot_roles(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(
            ["x", "signal", "device"],
            ["x"],
            ["signal"],
            [],
        )

        tag_bar._groupby_checks["device"].setChecked(True)

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert tag_bar.groupby == ["device"]
        assert tag_bar._split()[2] == ["device"]
        assert tag_bar.groupby_button.text() == "group by (1)"

    def test_dragging_groupby_column_into_axes_unchecks_it(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(
            ["x", "signal", "device"],
            ["x"],
            ["signal"],
            ["device"],
        )
        device_index = next(
            index
            for index in range(tag_bar._list.count())
            if tag_bar._list.item(index).text() == "device"
        )
        tag_bar._loading = True
        device_item = tag_bar._list.takeItem(device_index)
        tag_bar._list.insertItem(0, device_item)
        tag_bar._loading = False

        tag_bar._on_model_changed()

        assert tag_bar.axes == ["device", "x"]
        assert tag_bar.groupby == []
        assert tag_bar.groupby_button.text() == "group by"


def test_resolve_plot_columns_fills_axes_before_fields() -> None:
    resolved = resolve_plot_columns(
        ["x", "signal", "reference"],
        [],
        [],
    )

    assert resolved == PlotColumns(
        axes=("x",),
        fields=("signal",),
        groupby=(),
        ignored=("reference",),
    )


def test_iter_plot_groups_preserves_order_multiple_keys_and_missing() -> None:
    frame = pd.DataFrame(
        {
            "device": ["B", "A", "B", None],
            "sweep": [2, 1, 2, 1],
            "x": [0, 1, 2, 3],
        }
    )

    groups = list(iter_plot_groups(frame, ["device", "sweep"]))

    assert [group.label for group in groups] == [
        "device=B, sweep=2",
        "device=A, sweep=1",
        "device=<NA>, sweep=1",
    ]
    assert [group.frame.index.tolist() for group in groups] == [[0, 2], [1], [3]]


class TestFits:
    def test_exponential_reports_decay_time_with_offset(self) -> None:
        x = np.linspace(2.0, 12.0, 80)
        y = 1.5 + 4.0 * np.exp(-(x - 2.0) / 2.75)

        result = fit_exponential(x, y)

        assert result.value == pytest.approx(2.75, rel=1e-5)
        assert result.label.startswith("τ = ")
        assert result.x[[0, -1]] == pytest.approx([2.0, 12.0])

    @pytest.mark.parametrize("coefficient", [-2.0, 2.0])
    def test_quadratic_reports_extremum(self, coefficient: float) -> None:
        x = np.linspace(1_000_000.0, 1_000_010.0, 51)
        y = coefficient * (x - 1_000_004.25) ** 2 + 3.0

        result = fit_quadratic(x, y)

        assert result.value == pytest.approx(1_000_004.25)
        assert result.label.startswith("x = ")

    def test_fit_rejects_too_few_selected_points(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            fit_exponential(np.array([0.0, 1.0, 2.0]), np.ones(3))


class TestPlotManagerFitAndColorBar:
    def test_save_tag_bar_persists_groupby_with_other_plot_roles(self) -> None:
        manager = PlotManager()
        updates: list[dict[str, object]] = []
        manager._plot_record = SimpleNamespace(
            meta=SimpleNamespace(update=lambda **changes: updates.append(changes))
        )
        manager.tag_bar.set_columns(
            ["x", "signal", "device"],
            ["x"],
            ["signal"],
            ["device"],
        )

        manager._save_tag_bar()

        assert updates == [
            {
                "plot_axes": ["x"],
                "plot_fields": ["signal"],
                "plot_groupby": ["device"],
            }
        ]
        manager.widget.deleteLater()

    def test_grouped_1d_plots_each_group_and_labels_cursor_series(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "device": ["A", "A", "B", "B"],
                "x": [0.0, 1.0, 0.0, 1.0],
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )

        manager._refresh_plot_1d("x", ["signal"], ["device"])

        assert [series.name for series in manager.cursor_controller._series] == [
            "signal | device=A",
            "signal | device=B",
        ]
        assert manager.fit_controller._field == "signal | device=A"
        assert manager._legend is not None
        assert [label.text for _, label in manager._legend.items] == [
            "device=A",
            "device=B",
        ]
        assert manager._legend.brush().color().alpha() == 200
        assert manager._legend.pen().style() == Qt.NoPen
        assert manager._legend.layout.horizontalSpacing() == 1
        assert manager._legend.layout.verticalSpacing() == 0
        assert manager._legend.layout.getContentsMargins() == (2.0, 2.0, 2.0, 2.0)
        assert "2 groups, 2 curves" in manager.plot_status_label.text()
        manager.widget.deleteLater()

    def test_grouped_2d_cursor_uses_first_largest_group(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "device": ["small"] * 4 + ["large"] * 6 + ["same-size"] * 6,
                "x": [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
                "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "z": list(range(16)),
            }
        )

        manager._refresh_plot_2d("x", "y", "z", ["device"])

        assert len(manager._mesh_items) == 3
        assert manager.cursor_controller._mesh is not None
        assert manager.cursor_controller._mesh.point_count == 6
        assert manager.cursor_controller._group_label == "device=large"
        assert manager._legend is not None
        assert manager.plot_widget.getPlotItem().listDataItems() == []
        assert manager._mesh_levels == pytest.approx((0.0, 15.0))
        assert "cursor uses device=large (6 points)" in manager.plot_status_label.text()

        manager.cursor_button.click()
        assert "device=large" in manager.section_readout.text()
        manager.widget.deleteLater()

    def test_fit_selection_uses_points_inside_both_rectangle_axes(self) -> None:
        manager = PlotManager()
        x = np.arange(-2.0, 4.0)
        y = x**2
        manager.fit_controller.set_series(x, y, "signal", "#1E90FF")
        manager.quadratic_fit_button.click()
        manager.fit_view_box.setXRange(-3.0, 3.0, padding=0)
        selection = QRectF(-1.1, -0.1, 2.2, 1.2)

        manager.fit_controller._fit_selection(
            "quadratic",
            selection,
        )

        assert "using 3 points" in manager.plot_status_label.text()
        assert "x = " in manager.plot_status_label.text()
        assert "minimum" not in manager.plot_status_label.text()
        assert "maximum" not in manager.plot_status_label.text()
        assert manager.quadratic_fit_button.isChecked()
        assert manager.fit_view_box._fit_kind == "quadratic"

        selected_points = manager.fit_controller._overlays[1]
        assert selected_points.opts["size"] == 6
        assert selected_points.opts["brush"].color().name() == "#1e90ff"
        result_text = manager.fit_controller._overlays[3]
        assert (
            result_text.pos().y() <= selection.top()
            or result_text.pos().y() >= selection.bottom()
        )

        manager.quadratic_fit_button.click()
        assert not manager.quadratic_fit_button.isChecked()
        assert manager.fit_view_box._fit_kind is None
        manager.widget.deleteLater()

    def test_fit_buttons_use_first_plotted_1d_field(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {"x": [0.0, 1.0], "a": [1.0, 2.0], "b": [2.0, 3.0]}
        )

        manager._refresh_plot_1d("x", ["a"])
        assert not manager.exponential_fit_button.isHidden()
        assert manager.exponential_fit_button.isEnabled()
        assert manager.quadratic_fit_button.isEnabled()
        assert (
            manager.plot_status_label.sizePolicy().horizontalPolicy()
            == QSizePolicy.Ignored
        )

        manager._refresh_plot_1d("x", ["a", "b"])
        assert manager.exponential_fit_button.isEnabled()
        assert manager.quadratic_fit_button.isEnabled()
        assert manager.fit_controller._field == "a"
        manager.widget.deleteLater()

    def test_plot_actions_use_compact_tool_buttons(self) -> None:
        manager = PlotManager()

        for button in (
            manager.exponential_fit_button,
            manager.quadratic_fit_button,
            manager.cursor_button,
            manager.copy_plot_button,
        ):
            assert isinstance(button, QToolButton)

        manager.widget.deleteLater()

    def test_points_context_menu_is_hidden(self) -> None:
        manager = PlotManager()
        plot_item = manager.plot_widget.getPlotItem()

        points_action = next(
            action
            for action in plot_item.ctrlMenu.actions()
            if action.text() == "Points"
        )

        assert not points_action.isVisible()
        manager.widget.deleteLater()

    def test_view_context_menu_has_save_plot_action(self) -> None:
        manager = PlotManager()

        assert manager.save_plot_action.text() == "Save plot"
        assert manager.save_plot_action in manager.fit_view_box.getMenu(None).actions()
        manager.widget.deleteLater()

    def test_view_context_menu_mirrors_log_mode_controls(self) -> None:
        manager = PlotManager()
        plot_item = manager.plot_widget.getPlotItem()
        menu_actions = manager.fit_view_box.getMenu(None).actions()

        assert manager.log_x_action in menu_actions
        assert manager.log_y_action in menu_actions
        manager.log_x_action.trigger()
        assert plot_item.ctrl.logXCheck.isChecked()
        plot_item.ctrl.logYCheck.setChecked(True)
        assert manager.log_y_action.isChecked()

        manager.log_x_action.trigger()
        plot_item.ctrl.logYCheck.setChecked(False)
        assert not plot_item.ctrl.logXCheck.isChecked()
        assert not manager.log_y_action.isChecked()
        manager.widget.deleteLater()

    def test_copy_shortcut_is_scoped_to_plot_widget(self) -> None:
        manager = PlotManager()

        assert manager.copy_plot_shortcut.key() == QKeySequence.Copy
        assert manager.copy_plot_shortcut.context() == Qt.WidgetWithChildrenShortcut
        assert manager.copy_plot_shortcut.parent() is manager.plot_widget
        manager.widget.deleteLater()

    def test_double_click_zooms_to_fit_all_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = PlotManager()
        plot_item = manager.plot_widget.getPlotItem()
        calls: list[float] = []
        monkeypatch.setattr(
            plot_item, "autoRange", lambda *, padding: calls.append(padding)
        )

        manager.fit_view_box.zoom_fit_requested.emit()

        assert calls == [PLOT_AUTO_RANGE_PADDING]
        manager.widget.deleteLater()

    def test_1d_cursor_and_fit_modes_are_mutually_exclusive(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [1.0, 2.0, 3.0]})
        manager._refresh_plot_1d("x", ["z"])

        manager.cursor_button.click()
        assert manager.cursor_button.isChecked()
        assert manager.cursor_controller._vertical_line is not None

        manager.exponential_fit_button.click()
        assert not manager.cursor_button.isChecked()
        assert manager.exponential_fit_button.isChecked()

        manager.cursor_button.click()
        assert manager.cursor_button.isChecked()
        assert not manager.exponential_fit_button.isChecked()
        manager.widget.deleteLater()

    def test_copy_plot_temporarily_adds_record_path_to_title(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotManager()
        record_path = Path("/logs/example-record")
        manager._plot_record = SimpleNamespace(path=record_path)
        observed: dict[str, object] = {}

        class FakeExporter:
            def __init__(self, plot_item) -> None:
                self.plot_item = plot_item
                self._parameters = {"width": 400}

            def parameters(self):
                return self._parameters

            def export(self, *, copy: bool) -> None:
                observed["copy"] = copy
                observed["title"] = self.plot_item.titleLabel.text
                observed["visible"] = self.plot_item.titleLabel.isVisible()
                observed["width"] = self._parameters["width"]

        monkeypatch.setattr(
            "logqbit.gui.browser.plot.manager.ImageExporter", FakeExporter
        )

        manager.copy_plot_to_clipboard()

        assert observed == {
            "copy": True,
            "title": str(record_path),
            "visible": True,
            "width": 800,
        }
        assert not manager.plot_widget.getPlotItem().titleLabel.isVisible()
        manager.widget.deleteLater()

    def test_save_plot_writes_png_with_record_path_title(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotManager()
        manager._plot_record = SimpleNamespace(path=tmp_path)
        observed: dict[str, object] = {}

        class FakeExporter:
            def __init__(self, plot_item) -> None:
                self.plot_item = plot_item
                self._parameters = {"width": 400}

            def parameters(self):
                return self._parameters

            def export(self, *, toBytes: bool):
                observed["to_bytes"] = toBytes
                observed["title"] = self.plot_item.titleLabel.text
                observed["width"] = self._parameters["width"]

                class FakeImage:
                    def save(self, path: str, image_format: str) -> bool:
                        observed["path"] = path
                        observed["format"] = image_format
                        return True

                return FakeImage()

        monkeypatch.setattr(
            "logqbit.gui.browser.plot.manager.ImageExporter", FakeExporter
        )

        manager.save_plot_action.trigger()

        assert observed == {
            "to_bytes": True,
            "title": str(tmp_path),
            "width": 800,
            "path": str(tmp_path / "plot.png"),
            "format": "PNG",
        }
        assert (
            manager.plot_status_label.text() == f"Saved plot to {tmp_path / 'plot.png'}"
        )
        assert not manager.plot_widget.getPlotItem().titleLabel.isVisible()
        manager.widget.deleteLater()

    def test_2d_export_disables_antialiasing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = PlotManager()
        manager._mesh_item = object()
        observed: dict[str, object] = {}

        class FakeExporter:
            def __init__(self, plot_item) -> None:
                self._parameters = {"width": 400, "antialias": True}

            def parameters(self):
                return self._parameters

        monkeypatch.setattr(
            "logqbit.gui.browser.plot.manager.ImageExporter", FakeExporter
        )

        exporter = manager._create_image_exporter(manager.plot_widget.getPlotItem())
        observed.update(exporter.parameters())

        assert observed == {"width": 800, "antialias": False}
        manager.widget.deleteLater()

    def test_save_plot_does_not_overwrite_existing_image(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "plot.png").touch()
        (tmp_path / "plot-1.png").touch()
        manager = PlotManager()
        manager._plot_record = SimpleNamespace(path=tmp_path)
        saved_paths: list[str] = []

        class FakeImage:
            def save(self, path: str, image_format: str) -> bool:
                saved_paths.append(path)
                return True

        monkeypatch.setattr(manager, "_render_plot_image", FakeImage)

        manager.save_plot()

        assert saved_paths == [str(tmp_path / "plot-2.png")]
        manager.widget.deleteLater()

    def test_color_bar_is_reused_and_removed_when_switching_to_1d(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )

        manager._refresh_plot_2d("x", "y", "z")
        color_bar = manager._color_bar
        assert color_bar is not None
        assert color_bar.getAxis("left").labelText == "z"
        assert color_bar.axis.labelText == ""
        assert color_bar.maximumHeight() == round(
            manager.plot_widget.getPlotItem().vb.height() * COLOR_BAR_HEIGHT_FACTOR
        )
        assert manager.exponential_fit_button.isHidden()
        assert manager.quadratic_fit_button.isHidden()

        manager._refresh_plot_2d("x", "y", "z")
        assert manager._color_bar is color_bar

        manager._refresh_plot_1d("x", ["z"])
        assert manager._color_bar is None
        manager.widget.deleteLater()

    def test_2d_cursor_replaces_color_bar_and_target_moves_both_lines(self) -> None:
        manager = PlotManager()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )
        manager._refresh_plot_2d("x", "y", "z")
        assert manager.plot_layout.columnStretch(1) == 0
        assert manager.plot_layout.rowStretch(1) == 0

        manager.cursor_button.click()
        controller = manager.cursor_controller
        assert manager._color_bar is None
        assert not manager.horizontal_section_widget.isHidden()
        assert not manager.vertical_section_widget.isHidden()
        assert manager.plot_layout.columnStretch(1) == 1
        assert manager.plot_layout.rowStretch(1) == 1

        controller._target.setPos(0.1, 0.2)
        assert controller._vertical_line.value() == pytest.approx(0.1)
        assert controller._horizontal_line.value() == pytest.approx(0.2)
        assert controller._horizontal_curve.isVisible() is False
        assert controller._vertical_curve.isVisible() is False
        assert not manager.section_readout.text()
        assert not manager.horizontal_section_widget.isHidden()

        controller._finish_2d_drag()
        assert controller._vertical_line.value() == pytest.approx(0.0)
        assert controller._horizontal_line.value() == pytest.approx(0.0)
        assert controller._horizontal_curve.isVisible()
        assert controller._vertical_curve.isVisible()
        assert len(controller._horizontal_curve.xData) == 2
        assert "z = 1" in manager.section_readout.text()

        manager.cursor_button.click()
        assert manager._color_bar is not None
        assert manager.horizontal_section_widget.isHidden()
        assert manager.vertical_section_widget.isHidden()
        assert manager.plot_layout.columnStretch(1) == 0
        assert manager.plot_layout.rowStretch(1) == 0
        manager.widget.deleteLater()


class TestPlotMeshSections:
    def test_sections_use_logical_columns_with_descending_and_ragged_y(self) -> None:
        mesh = build_plot_mesh(
            np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
            np.array([2.0, 1.0, 0.0, 3.0, 1.0]),
            np.array([20.0, 10.0, 0.0, 31.0, 11.0]),
        )

        x, y, z = mesh.vertical_section(0.8)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx([3.0, 1.0])
        assert z == pytest.approx([31.0, 11.0])

        section_x, section_z = mesh.horizontal_section(1.2)
        assert section_x == pytest.approx([0.0, 1.0])
        assert section_z == pytest.approx([10.0, 11.0])

        assert mesh.nearest_point(0.8, 2.4) == pytest.approx((1.0, 3.0, 31.0))

    def test_horizontal_section_does_not_clamp_outside_each_column(self) -> None:
        mesh = build_plot_mesh(
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([10.0, 11.0, 22.0, 23.0]),
        )

        _, z = mesh.horizontal_section(0.5)

        assert z[0] == pytest.approx(10.0)
        assert np.isnan(z[1])


class TestIsLexsorted:
    def test_already_sorted(self) -> None:
        x = np.array([1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 1.0, 2.0])
        assert _is_lexsorted(x, y) is True

    def test_not_sorted_by_x(self) -> None:
        x = np.array([2.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 2.0])
        assert _is_lexsorted(x, y) is False

    def test_not_sorted_by_y_within_same_x(self) -> None:
        # y reverses direction mid-column (1→3→2): not monotonic → False
        x = np.array([1.0, 1.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 2.0, 1.0])
        assert _is_lexsorted(x, y) is False

    def test_y_descending_within_column_is_ok(self) -> None:
        # y monotonically descending within x=1 is now allowed
        x = np.array([1.0, 1.0, 2.0])
        y = np.array([2.0, 1.0, 1.0])
        assert _is_lexsorted(x, y) is True

    def test_single_element(self) -> None:
        x = np.array([1.0])
        y = np.array([1.0])
        assert _is_lexsorted(x, y) is True

    def test_strictly_increasing_x(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 3.0, 1.0])  # y can be anything when x changes
        assert _is_lexsorted(x, y) is True


class TestBuildGridsRect:
    def _make_inputs(self, x_data, y_data, z_data):
        """Compute all inputs needed by _build_grids_rect from flat x/y/z arrays."""
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        z_data = np.asarray(z_data, dtype=float)
        N = len(x_data)

        change = np.empty(N, dtype=np.bool_)
        change[0] = True
        change[1:] = x_data[1:] != x_data[:-1]
        xu = x_data[change]
        col_starts = np.flatnonzero(change)
        nx_col = len(xu)
        col_ends = np.append(col_starts[1:], N)
        col_sizes = col_ends - col_starts
        max_ny = int(col_sizes.max())

        ref_col = int(np.argmax(col_sizes))
        ref_y = y_data[col_starts[ref_col] : col_ends[ref_col]]
        typical_dy = float(np.median(np.diff(ref_y))) if len(ref_y) > 1 else 1.0

        last_y = y_data[col_ends - 1]
        prev_idx = np.maximum(col_ends - 2, col_starts)
        step_c = np.where(col_sizes > 1, last_y - y_data[prev_idx], typical_dy)
        top_y = last_y + step_c

        return (
            x_data,
            y_data,
            z_data,
            col_starts,
            col_sizes,
            max_ny,
            nx_col,
            top_y,
            step_c,
        )

    def test_uniform_grid_shape(self) -> None:
        """A 3x2 grid: z_final shape (2, 5), y_final shape (3, 6)."""
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        z = np.arange(6.0)

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        assert z_final.shape == (max_ny, 2 * nx_col - 1)  # (2, 5)
        assert y_final.shape == (max_ny + 1, 2 * nx_col)  # (3, 6)

    def test_z_values_placed_correctly(self) -> None:
        """Data columns sit at even indices; odd separator columns are NaN."""
        x = np.array([0.0, 0.0, 1.0, 1.0])
        y = np.array([0.0, 1.0, 0.0, 2.0])
        z = np.array([10.0, 20.0, 30.0, 40.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, _ = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        # Column 0 (c2=0): z values 10, 20
        assert z_final[0, 0] == pytest.approx(10.0)
        assert z_final[1, 0] == pytest.approx(20.0)
        # Column 1 (c2=2): z values 30, 40
        assert z_final[0, 2] == pytest.approx(30.0)
        assert z_final[1, 2] == pytest.approx(40.0)
        # Odd separator column is NaN
        assert np.isnan(z_final[0, 1])

    def test_y_corners_horizontal_edges(self) -> None:
        """Left and right y corners per cell must be equal (horizontal edges)."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        _, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        # Column 0 occupies c2=0,1; left and right y must match
        for r in range(max_ny + 1):
            assert y_final[r, 0] == pytest.approx(y_final[r, 1])
        # First 3 rows match input y
        assert y_final[0, 0] == pytest.approx(1.0)
        assert y_final[1, 0] == pytest.approx(2.0)
        assert y_final[2, 0] == pytest.approx(3.0)
        # Top extrapolated: 3 + 1 = 4
        assert y_final[3, 0] == pytest.approx(4.0)

    def test_unequal_column_sizes_no_crash(self) -> None:
        """Columns with different point counts should not crash."""
        x = np.array([0.0, 0.0, 0.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 5.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])

        args = self._make_inputs(x, y, z)
        _, _, _, col_starts, col_sizes, max_ny, nx_col, top_y, step_c = args

        z_final, y_final = _build_grids_rect(
            y, z, col_starts, col_sizes, max_ny, nx_col, top_y, step_c
        )

        assert z_final.shape == (max_ny, 2 * nx_col - 1)
        # Short column (x=1, c2=2): first row filled, rest NaN
        assert z_final[0, 2] == pytest.approx(4.0)
        assert np.isnan(z_final[1, 2])
        assert np.isnan(z_final[2, 2])
        # y corners for short column extrapolated (not NaN)
        assert not np.isnan(y_final[1, 2])
        assert not np.isnan(y_final[2, 2])
        assert not np.isnan(y_final[3, 2])


def test_warmup_plotter_jit() -> None:
    warmup_plotter_jit()
