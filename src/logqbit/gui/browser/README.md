# Browser ownership

The package follows the user workflow: navigate records, inspect one record,
then interact with its data and plots. GUI modules are internal implementation
boundaries; changing them does not require compatibility aliases.

- `window/view.py` assembles the main window and coordinates navigation,
  record operations, and standalone detail windows.
- `window/navigation.py` owns the table/model, filters, selection, pins,
  directory catalog cache, and directory refresh scheduling.
- `window/records.py` owns record commands, their dialogs and menu actions.
- `window/merge.py` owns merge/append preparation, publication, and cleanup.
- `window/preferences.py` owns persistent settings and native Qt themes.
- `detail/view.py` owns one current record, metadata/data version caches,
  its watcher, and tab lifecycle. Standalone windows reuse this component.
- `detail/const.py`, `data.py`, and `files.py` own their respective content
  and local operations. File changes notify the detail container.
- `plot/view.py` owns the plot panel, mouse input, refresh policy,
  interaction coordination, and image export.
- `plot/controls.py` owns field roles and grouping controls.
- `plot/rendering.py` prepares plot coordinates/groups and owns data graphics,
  axes, legends and color-bar signal connections. It returns data for controllers
  without reaching into the panel, metadata, or interaction buttons.
- `plot/cursor.py` and `fitting.py` own their interaction overlays.
  `plot/mesh.py` contains mesh construction and section queries.

## Refresh boundaries

Catalog refresh scans list summaries and reuses unchanged `LogRecord` handles.
A detail view owns its full DataFrame cache and refreshes its record directly;
its notification updates the corresponding list row. Do not make local detail
refresh depend on the catalog or make list refresh load full DataFrames.

Metadata changes synchronize plotting controls. Feather-only changes update
plot data without replacing unsaved control choices. Hidden plots defer drawing
until selected. Once the user takes control of the view, data refresh retains
its range and interaction overlays; record or role changes reset that policy.
Completed fit overlays intentionally remain results of the original selection.

Only the renderer removes data items. Cursor and fit controllers remove their
own overlays. Detach color-bar signals before removing the mesh from the scene.
Do not replace this with a blanket `PlotItem.clear()` during data refresh.

Keep selection debounce, list refresh debounce, and detail refresh debounce
with their respective owners: they serve different purposes and must not be
combined into a shared watcher service.
