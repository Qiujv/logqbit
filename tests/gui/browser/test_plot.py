"""Tests for plotter module helper functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtWidgets import QSizePolicy, QStyleOptionViewItem
import pyqtgraph as pg

from logqbit.catalog import PlotColumns, resolve_plot_columns
from logqbit.gui.browser.plot.fitting import fit_exponential, fit_quadratic
from logqbit.gui.browser.plot.view import PlotViewBox
from logqbit.gui.browser.plot.controls import TagBar
from logqbit.gui.browser.plot.rendering import iter_plot_groups, MAX_PLOT_GROUPS
from logqbit.gui.browser.plot.mesh import (
    _build_grids_rect,
    _is_lexsorted,
    build_plot_mesh,
    warmup_plotter_jit,
)
from logqbit.gui.browser.plot.view import (
    PLOT_AUTO_RANGE_PADDING,
    PlotView,
)


class TestTagBar:
    def test_tag_items_are_compact_but_short_tags_remain_clickable(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(["x", "signal"], ["x"], ["signal"], [])

        item = tag_bar._list.item(0)
        metrics = tag_bar._list.fontMetrics()
        option = QStyleOptionViewItem()
        option.font = tag_bar._list.font()
        size = tag_bar._list.itemDelegate().sizeHint(
            option,
            tag_bar._list.model().index(tag_bar._list.row(item), 0),
        )

        assert tag_bar._list.spacing() == 2
        assert tag_bar._list.styleSheet() == ""
        assert size.width() >= metrics.horizontalAdvance("00")

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

    def test_dragging_tag_clears_selection(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(["x", "signal", "reference"], ["x"], ["signal"], [])
        signal_item = tag_bar._list.item(2)
        tag_bar._list.setCurrentItem(signal_item)
        signal_item.setSelected(True)

        tag_bar._loading = True
        moved_item = tag_bar._list.takeItem(2)
        tag_bar._list.insertItem(3, moved_item)
        tag_bar._loading = False
        tag_bar._on_model_changed()

        assert not tag_bar._list.selectedItems()

    def test_canceling_tag_drag_clears_selection(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(["x", "signal"], ["x"], ["signal"], [])
        signal_item = tag_bar._list.item(2)
        tag_bar._list.setCurrentItem(signal_item)
        signal_item.setSelected(True)

        tag_bar._list.startDrag(Qt.MoveAction)

        assert not tag_bar._list.selectedItems()

    def test_clicking_field_moves_it_to_start_of_ignored_columns(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(
            ["x", "signal", "reference", "note"],
            ["x"],
            ["signal", "reference"],
            [],
        )
        changes: list[None] = []
        tag_bar.changed.connect(lambda: changes.append(None))

        tag_bar._toggle_item_role(tag_bar._list.item(2))

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["reference"]
        assert tag_bar._split()[2] == ["signal", "note"]
        assert changes == [None]

    def test_clicking_axis_does_not_change_roles(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(["x", "signal"], ["x"], ["signal"], [])
        changes: list[None] = []
        tag_bar.changed.connect(lambda: changes.append(None))

        tag_bar._toggle_item_role(tag_bar._list.item(0))

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert changes == []

    def test_clicking_non_field_moves_it_to_end_of_fields(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(
            ["x", "signal", "reference"],
            ["x"],
            ["signal"],
            [],
        )

        tag_bar._toggle_item_role(tag_bar._list.item(4))

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal", "reference"]
        assert tag_bar._split()[2] == []

    def test_clicking_groupby_column_moves_it_to_fields(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(
            ["x", "signal", "device"],
            ["x"],
            ["signal"],
            ["device"],
        )

        tag_bar._toggle_item_role(tag_bar._list.item(4))

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal", "device"]
        assert tag_bar.groupby == []

    def test_clicking_separator_does_not_change_roles(self) -> None:
        tag_bar = TagBar()
        tag_bar.set_columns(["x", "signal"], ["x"], ["signal"], [])
        changes: list[None] = []
        tag_bar.changed.connect(lambda: changes.append(None))

        tag_bar._toggle_item_role(tag_bar._list.item(1))

        assert tag_bar.axes == ["x"]
        assert tag_bar.fields == ["signal"]
        assert changes == []


class TestCursorClickHandling:
    def test_single_click_is_deferred_and_double_click_cancels_it(self) -> None:
        class ClickEvent:
            def __init__(self, position: QPointF, *, double: bool) -> None:
                self._position = position
                self._double = double
                self.accepted = False

            def button(self):
                return Qt.LeftButton

            def double(self) -> bool:
                return self._double

            def pos(self) -> QPointF:
                return self._position

            def accept(self) -> None:
                self.accepted = True

        view_box = PlotViewBox()
        clicks: list[QPointF] = []
        zooms: list[None] = []
        view_box.cursor_click_requested.connect(clicks.append)
        view_box.zoom_fit_requested.connect(lambda: zooms.append(None))

        single = ClickEvent(QPointF(1, 2), double=False)
        view_box.mouseClickEvent(single)
        assert single.accepted
        assert clicks == []

        view_box.mouseClickEvent(ClickEvent(QPointF(1, 2), double=True))
        view_box._emit_cursor_click()

        assert clicks == []
        assert zooms == [None]

    def test_cursor_click_moves_to_the_nearest_1d_data_point(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [1.0, 2.0, 3.0]})
        manager._refresh_plot_1d("x", ["z"])
        manager.cursor_button.click()

        position = manager.view_box.mapFromView(QPointF(1.8, 2.0))
        manager.view_box.cursor_click_requested.emit(position)

        assert manager.cursor_controller._vertical_line.value() == pytest.approx(2.0)
        manager.deleteLater()

    def test_data_refresh_keeps_a_user_cursor_and_view_range(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
        manager._refresh_plot_1d("x", ["y"])
        manager.cursor_button.click()
        cursor_line = manager.cursor_controller._vertical_line
        assert cursor_line is not None
        manager.view_box.setRange(xRange=(0.25, 0.75), yRange=(1.25, 1.75))
        original_x_range = manager.view_box.viewRange()[0]
        manager._user_controls_view = True

        manager._plot_frame = pd.DataFrame(
            {"x": [0.0, 1.0, 2.0], "y": [10.0, 20.0, 30.0]}
        )
        manager._refresh_plot_1d("x", ["y"])

        assert manager.cursor_controller._vertical_line is cursor_line
        assert manager.view_box.viewRange()[0] == pytest.approx(original_x_range)
        np.testing.assert_allclose(
            manager.plot_widget.getPlotItem().listDataItems()[0].getData()[1],
            [10.0, 20.0, 30.0],
        )
        manager.deleteLater()

    def test_preview_coordinates_use_a_tenth_of_the_minor_tick_spacing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotView()
        axis = manager.plot_widget.getAxis("bottom")
        monkeypatch.setattr(
            axis,
            "tickSpacing",
            lambda *_args: [(1.0, 0.0), (0.2, 0.1)],
        )

        assert manager.cursor_controller._snap_preview_coordinate(
            0.371,
            "bottom",
        ) == pytest.approx(0.38)
        manager.deleteLater()


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


class TestPlotViewFitAndColorBar:
    def test_save_tag_bar_persists_groupby_with_other_plot_roles(self) -> None:
        manager = PlotView()
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
        manager.deleteLater()

    def test_grouped_1d_plots_each_group_and_labels_cursor_series(self) -> None:
        manager = PlotView()
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
        assert manager.renderer._legend is not None
        assert [label.text for _, label in manager.renderer._legend.items] == [
            "device=A",
            "device=B",
        ]
        assert manager.renderer._legend.brush().color().alpha() == 200
        assert manager.renderer._legend.pen().style() == Qt.NoPen
        assert manager.renderer._legend.layout.horizontalSpacing() == 1
        assert manager.renderer._legend.layout.verticalSpacing() == 0
        assert manager.renderer._legend.layout.getContentsMargins() == (
            2.0,
            2.0,
            2.0,
            2.0,
        )
        assert "2 groups, 2 curves" in manager.plot_status_label.text()
        manager.deleteLater()

    def test_grouped_1d_plots_only_the_latest_fifty_groups(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        devices = np.repeat(np.arange(MAX_PLOT_GROUPS + 2), 2)
        manager._plot_frame = pd.DataFrame(
            {
                "device": devices,
                "x": np.tile([0.0, 1.0], len(devices) // 2),
                "signal": np.arange(len(devices), dtype=float),
            }
        )

        manager._refresh_plot_1d("x", ["signal"], ["device"])

        assert len(manager.cursor_controller._series) == MAX_PLOT_GROUPS
        assert manager.cursor_controller._series[0].name == "signal | device=2"
        assert manager.cursor_controller._series[-1].name == "signal | device=51"
        assert "showing latest 50 of 52" in manager.plot_status_label.text()
        manager.deleteLater()

    def test_grouped_2d_cursor_uses_first_largest_group(self) -> None:
        manager = PlotView()
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

        assert len(manager.renderer._mesh_items) == 3
        assert manager.cursor_controller._mesh is not None
        assert manager.cursor_controller._mesh.point_count == 6
        assert manager.cursor_controller._group_label == "device=large"
        assert manager.renderer._legend is not None
        assert manager.plot_widget.getPlotItem().listDataItems() == []
        assert manager.renderer._mesh_levels == pytest.approx((0.0, 15.0))
        assert "cursor uses device=large (6 points)" in manager.plot_status_label.text()

        manager.cursor_button.click()
        assert "device=large" in manager.section_readout.text()
        manager.deleteLater()

    def test_fit_selection_uses_points_inside_both_rectangle_axes(self) -> None:
        manager = PlotView()
        x = np.arange(-2.0, 4.0)
        y = x**2
        manager.fit_controller.set_series(x, y, "signal", "#1E90FF")
        manager.quadratic_fit_button.click()
        manager.view_box.setXRange(-3.0, 3.0, padding=0)
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
        assert manager.view_box._fit_kind == "quadratic"

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
        assert manager.view_box._fit_kind is None
        manager.deleteLater()

    def test_data_refresh_keeps_completed_fit_overlays(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {"x": [0.0, 1.0, 2.0, 3.0], "y": [9.0, 4.0, 1.0, 0.0]}
        )
        manager._refresh_plot_1d("x", ["y"])
        manager.fit_controller._fit_selection("quadratic", QRectF(-1, -1, 5, 11))
        overlays = tuple(manager.fit_controller._overlays)
        assert overlays
        manager._user_controls_view = True

        manager._plot_frame = pd.DataFrame(
            {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 4.0, 9.0]}
        )
        manager._refresh_plot_1d("x", ["y"])

        assert tuple(manager.fit_controller._overlays) == overlays
        manager.deleteLater()

    def test_fit_buttons_use_first_plotted_1d_field(self) -> None:
        manager = PlotView()
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
        manager.deleteLater()

    def test_datetime_columns_use_date_axes_and_seconds_since_epoch(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-09-09 12:00", "2026-09-09 12:01"]),
                "value": [1.0, 2.0],
            }
        )

        manager._refresh_plot_1d("time", ["value"])

        plot_item = manager.plot_widget.getPlotItem()
        assert isinstance(plot_item.getAxis("bottom"), pg.DateAxisItem)
        assert not isinstance(plot_item.getAxis("left"), pg.DateAxisItem)
        timestamps = plot_item.listDataItems()[0].getData()[0]
        assert timestamps[1] - timestamps[0] == pytest.approx(60.0)
        assert 1_000_000_000 < timestamps[0] < 2_000_000_000
        assert not manager.log_x_action.isEnabled()
        assert manager.exponential_fit_button.isEnabled()

        manager._plot_frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})
        manager._refresh_plot_1d("x", ["y"])
        assert not isinstance(plot_item.getAxis("bottom"), pg.DateAxisItem)
        assert manager.log_x_action.isEnabled()
        manager.deleteLater()

    def test_numeric_grouped_plots_reuse_existing_axes(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        plot_item = manager.plot_widget.getPlotItem()
        bottom_axis = plot_item.getAxis("bottom")
        left_axis = plot_item.getAxis("left")
        manager._plot_frame = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "x": [0.0, 1.0, 0.0, 1.0],
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )

        manager._refresh_plot_1d("x", ["signal"], ["group"])

        manager._plot_frame = pd.DataFrame(
            {
                "group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "x": [0.0, 0.0, 1.0, 1.0] * 2,
                "y": [0.0, 1.0, 0.0, 1.0] * 2,
                "signal": list(range(8)),
            }
        )
        manager._refresh_plot_2d("x", "y", "signal", ["group"])

        assert plot_item.getAxis("bottom") is bottom_axis
        assert plot_item.getAxis("left") is left_axis
        manager.deleteLater()

    def test_quadratic_datetime_fit_reports_a_readable_extremum(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        timestamps = pd.date_range("2026-09-09 12:00", periods=5, freq="min")
        manager._plot_frame = pd.DataFrame(
            {
                "time": timestamps,
                "value": [4.0, 1.0, 0.0, 1.0, 4.0],
            }
        )

        manager._refresh_plot_1d("time", ["value"])
        x_values = manager.fit_controller._x
        assert x_values is not None
        manager.fit_controller._fit_selection(
            "quadratic",
            QRectF(x_values[0] - 1, -1, x_values[-1] - x_values[0] + 2, 6),
        )

        assert "x = 2026-09-09 12:02:00" in manager.plot_status_label.text()
        manager.deleteLater()

    def test_datetime_field_uses_a_date_y_axis(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "time": pd.to_datetime(["2026-09-09 12:00", "2026-09-09 12:01"]),
            }
        )

        manager._refresh_plot_1d("x", ["time"])

        assert isinstance(manager.plot_widget.getAxis("left"), pg.DateAxisItem)
        assert not manager.log_y_action.isEnabled()
        manager.deleteLater()

    def test_datetime_and_numeric_fields_do_not_share_a_y_axis(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "value": [1.0, 2.0],
                "time": pd.to_datetime(["2026-09-09 12:00", "2026-09-09 12:01"]),
            }
        )

        manager._refresh_plot_1d("x", ["value", "time"])

        assert manager.plot_status_label.text() == (
            "Datetime and numeric fields cannot share a y axis."
        )
        assert manager.plot_widget.getPlotItem().listDataItems() == []
        manager.deleteLater()

    def test_view_context_menu_mirrors_log_mode_controls(self) -> None:
        manager = PlotView()
        plot_item = manager.plot_widget.getPlotItem()
        menu_actions = manager.view_box.getMenu(None).actions()

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
        manager.deleteLater()

    def test_marker_size_menu_updates_1d_markers_and_hides_for_2d(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]})

        manager._refresh_plot_1d("x", ["y"])

        item = manager.plot_widget.getPlotItem().listDataItems()[0]
        large_action = next(
            action
            for action in manager.marker_size_actions.actions()
            if action.text() == "Large"
        )
        assert manager.marker_size_menu.menuAction().isVisible()
        assert item.opts["symbol"] == "o"
        assert item.opts["symbolSize"] == 6

        large_action.trigger()

        assert item.opts["symbolSize"] == 8
        assert large_action.isChecked()

        values = np.arange(2002, dtype=float)
        manager._plot_frame = pd.DataFrame({"x": values, "y": values})
        manager._refresh_plot_1d("x", ["y"])

        assert (
            manager.plot_widget.getPlotItem().listDataItems()[0].opts["symbol"] is None
        )

        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )
        manager._refresh_plot_2d("x", "y", "z")

        assert not manager.marker_size_menu.menuAction().isVisible()
        manager.deleteLater()

    def test_double_click_zooms_to_fit_all_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = PlotView()
        plot_item = manager.plot_widget.getPlotItem()
        calls: list[float] = []
        monkeypatch.setattr(
            plot_item, "autoRange", lambda *, padding: calls.append(padding)
        )

        manager.view_box.zoom_fit_requested.emit()

        assert calls == [PLOT_AUTO_RANGE_PADDING]
        manager.deleteLater()

    def test_1d_cursor_and_fit_modes_are_mutually_exclusive(self) -> None:
        manager = PlotView()
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
        manager.deleteLater()

    def test_copy_plot_temporarily_adds_record_path_to_title(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotView()
        record_path = Path("/logs/example-record")
        manager._plot_record = SimpleNamespace(path=record_path, title="Example title")
        plot_item = manager.plot_widget.getPlotItem()
        initial_minimum_width = plot_item.layout.minimumWidth()
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
                observed["size"] = self.plot_item.titleLabel.opts["size"]
                observed["width"] = self._parameters["width"]

        monkeypatch.setattr("logqbit.gui.browser.plot.view.ImageExporter", FakeExporter)

        manager.copy_plot_to_clipboard()

        assert observed == {
            "copy": True,
            "title": f"Example title — {record_path.as_posix().replace('/', '/<wbr>')}",
            "visible": True,
            "size": (
                f"{plot_item.getAxis('bottom').label.document().defaultFont().pointSizeF():g}pt"
            ),
            "width": 800,
        }
        assert not plot_item.titleLabel.isVisible()
        assert plot_item.titleLabel.text == ""
        assert plot_item.layout.minimumWidth() == initial_minimum_width
        manager.deleteLater()

    def test_save_plot_writes_png_with_record_path_title(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manager = PlotView()
        manager._plot_record = SimpleNamespace(path=tmp_path, title="Example title")
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

        monkeypatch.setattr("logqbit.gui.browser.plot.view.ImageExporter", FakeExporter)

        manager.save_plot_action.trigger()

        assert observed == {
            "to_bytes": True,
            "title": f"Example title — {tmp_path.as_posix().replace('/', '/<wbr>')}",
            "width": 800,
            "path": str(tmp_path / "plot.png"),
            "format": "PNG",
        }
        assert (
            manager.plot_status_label.text() == f"Saved plot to {tmp_path / 'plot.png'}"
        )
        assert not manager.plot_widget.getPlotItem().titleLabel.isVisible()
        manager.deleteLater()

    def test_2d_export_disables_antialiasing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = PlotView()
        manager.renderer._mesh_item = object()
        observed: dict[str, object] = {}

        class FakeExporter:
            def __init__(self, plot_item) -> None:
                self._parameters = {"width": 400, "antialias": True}

            def parameters(self):
                return self._parameters

        monkeypatch.setattr("logqbit.gui.browser.plot.view.ImageExporter", FakeExporter)

        exporter = manager._create_image_exporter(manager.plot_widget.getPlotItem())
        observed.update(exporter.parameters())

        assert observed == {"width": 800, "antialias": False}
        manager.deleteLater()

    def test_save_plot_does_not_overwrite_existing_image(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "plot.png").touch()
        (tmp_path / "plot-1.png").touch()
        manager = PlotView()
        manager._plot_record = SimpleNamespace(path=tmp_path)
        saved_paths: list[str] = []

        class FakeImage:
            def save(self, path: str, image_format: str) -> bool:
                saved_paths.append(path)
                return True

        monkeypatch.setattr(manager, "_render_plot_image", FakeImage)

        manager.save_plot()

        assert saved_paths == [str(tmp_path / "plot-2.png")]
        manager.deleteLater()

    def test_2d_cursor_replaces_color_bar_and_click_moves_both_lines(self) -> None:
        manager = PlotView()
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
        assert manager.renderer._color_bar is None
        assert not manager.horizontal_section_widget.isHidden()
        assert not manager.vertical_section_widget.isHidden()
        assert manager.plot_layout.columnStretch(1) == 1
        assert manager.plot_layout.rowStretch(1) == 1

        controller._update_2d(0.1, 0.2)
        assert controller._vertical_line.value() == pytest.approx(0.0)
        assert controller._horizontal_line.value() == pytest.approx(0.0)
        assert controller._horizontal_curve.isVisible()
        assert controller._vertical_curve.isVisible()
        assert len(controller._horizontal_curve.xData) == 2
        assert "z = 1" in manager.section_readout.text()

        manager.cursor_button.click()
        assert manager.renderer._color_bar is not None
        assert manager.horizontal_section_widget.isHidden()
        assert manager.vertical_section_widget.isHidden()
        assert manager.plot_layout.columnStretch(1) == 0
        assert manager.plot_layout.rowStretch(1) == 0
        manager.deleteLater()

    def test_replacing_2d_mesh_disconnects_old_color_bar_signal(self) -> None:
        manager = PlotView()
        manager._plot_record = object()
        manager._plot_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )
        manager._refresh_plot_2d("x", "y", "z")
        old_mesh = manager.renderer._mesh_item
        color_bar = manager.renderer._color_bar

        manager._refresh_plot_2d("x", "y", "z")

        assert old_mesh is not None
        assert color_bar is manager.renderer._color_bar
        levels = color_bar.levels()
        old_mesh.setLevels((10.0, 20.0))
        assert color_bar.levels() == levels
        manager.deleteLater()


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
