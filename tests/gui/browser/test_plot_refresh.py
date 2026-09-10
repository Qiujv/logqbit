"""Exercise plot lifecycle through the record-detail loading boundary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QRectF

from logqbit.catalog import LogRecord
from logqbit.logfolder import LogFolder
from logqbit.gui.browser.detail.view import RecordDetailView, TAB_PLOT, TAB_CONST


def make_record(directory: Path) -> LogRecord:
    with LogFolder.new(directory, title="refresh") as log:
        for x in range(4):
            log.add_row(x=float(x), y=float(x * x), z=float(x + 1))
        log.meta.update(plot_axes=["x"], plot_fields=["y"])
    return LogRecord(log.path).refresh()


@pytest.mark.parametrize("interaction", ["cursor", "fit"])
def test_data_refresh_preserves_interactions_but_metadata_resets_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interaction: str
) -> None:
    record = make_record(tmp_path)
    detail = RecordDetailView()
    detail.set_watch_enabled(False)
    try:
        detail.load_record(record)
        detail.set_current_tab(TAB_PLOT)
        plot = detail.plot_view
        if interaction == "cursor":
            plot.cursor_button.click()
            overlays = (plot.cursor_controller._vertical_line,)
        else:
            plot.quadratic_fit_button.click()
            plot.view_box.selection_finished.emit("quadratic", QRectF(-1, -1, 5, 12))
            overlays = tuple(plot.fit_controller._overlays)
        assert overlays
        plot.view_box.setRange(xRange=(0.25, 1.75), yRange=(0.5, 4.0), padding=0)
        original_range = plot.view_box.viewRange()[0]
        control_updates = []
        original_set_columns = plot.tag_bar.set_columns

        def set_columns(*args):
            control_updates.append(args)
            return original_set_columns(*args)

        monkeypatch.setattr(plot.tag_bar, "set_columns", set_columns)

        pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [10.0, 20.0, 30.0, 40.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        ).to_feather(record.data_path)
        detail.refresh_current_record()

        if interaction == "cursor":
            assert plot.cursor_controller._vertical_line is overlays[0]
        else:
            assert tuple(plot.fit_controller._overlays) == overlays
        assert plot.view_box.viewRange()[0] == pytest.approx(original_range)
        assert not control_updates
        np.testing.assert_allclose(
            plot.renderer._data_items[0].getData()[1],
            [10.0, 20.0, 30.0, 40.0],
        )

        record.meta.update(plot_fields=["z"])
        detail.refresh_current_record()
        assert not plot.cursor_controller.active
        assert not plot.fit_controller._overlays
        assert len(control_updates) == 1
        assert plot.tag_bar.fields == ["z"]
        np.testing.assert_allclose(
            plot.renderer._data_items[0].getData()[1],
            [1.0, 2.0, 3.0, 4.0],
        )
    finally:
        detail.close()
        detail.deleteLater()


def test_hidden_plot_renders_latest_data_and_switching_record_clears_cursor(
    tmp_path: Path,
) -> None:
    record = make_record(tmp_path)
    other = make_record(tmp_path)
    detail = RecordDetailView()
    detail.set_watch_enabled(False)
    try:
        detail.load_record(record)
        detail.set_current_tab(TAB_PLOT)
        plot = detail.plot_view
        plot.cursor_button.click()
        detail.set_current_tab(TAB_CONST)
        for value in [10.0, 20.0]:
            pd.DataFrame(
                {"x": [0.0, 1.0], "y": [value, value + 1], "z": [2.0, 3.0]}
            ).to_feather(record.data_path)
            detail.refresh_current_record()
        assert plot._needs_refresh
        detail.set_current_tab(TAB_PLOT)
        assert not plot._needs_refresh
        np.testing.assert_allclose(
            plot.renderer._data_items[0].getData()[1], [20.0, 21.0]
        )
        detail.load_record(other)
        assert not plot.cursor_controller.active
        np.testing.assert_allclose(
            plot.renderer._data_items[0].getData()[1],
            [0.0, 1.0, 4.0, 9.0],
        )
    finally:
        detail.close()
        detail.deleteLater()
