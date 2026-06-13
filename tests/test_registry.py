from pathlib import Path

import pytest

from logqbit.registry import Registry

test_data = """
create_time: '2024-01-01 12:00:00'
create_machine: test_machine
period_ns: 50_000
Device:
  Q1:
    frr: 5.856 GHz
    centers: !numpy
      - [-362.36833, 306.784]
      - [-264.632, 74.66033]
      - [-2433.79567, -3025.139]
"""

@pytest.fixture
def temp_yaml(tmp_path: Path) -> Path:
    dst = tmp_path / "test.yaml"
    with open(dst, "w", encoding="utf-8") as f:
        f.write(test_data)
    return dst


def test_get_unit_value(temp_yaml):
    pytest.importorskip("labrad.units", reason="requires labrad.units for unit parsing")
    reg = Registry(temp_yaml)
    val = reg.get("Device/Q1/frr")
    assert hasattr(val, "unit")
    assert val.unit.name == "GHz"
    assert pytest.approx(val._value, rel=0, abs=1e-9) == 5.856


def test_set_and_persistence(temp_yaml):
    reg = Registry(temp_yaml)
    # set a new nested key and ensure it's written to file
    reg["new_section/answer"] = 42

    # creating a new Registry from the file should see the persisted value
    reg2 = Registry(temp_yaml)
    assert reg2.get("new_section/answer") == 42


def test_local_change_not_saved_until_save(temp_yaml):
    reg = Registry(temp_yaml)
    # make a local-only change
    reg.root["local_only"] = "temp"

    # new Registry instance (reads file) should not see local_only
    reg2 = Registry(temp_yaml)
    with pytest.raises(KeyError):
        reg2.get("local_only")

    # after saving, the change should persist
    reg.save()
    reg2.reload()
    assert reg2.get("local_only") == "temp"
    assert reg.get("local_only") == "temp"


def test_reload_detects_external_change(temp_yaml):
    reg = Registry(temp_yaml)

    # modify the file externally: change period_ns value
    text = temp_yaml.read_text(encoding="utf-8")
    new_text = text.replace("50_000", "12345")
    temp_yaml.write_text(new_text, encoding="utf-8")

    # Change should be detected and reloaded automatically
    assert reg.get("period_ns") == 12345


def test_undo_redo_for_set(temp_yaml):
    reg = Registry(temp_yaml)

    reg["period_ns"] = 12345
    assert reg.get("period_ns") == 12345

    assert reg.undo()
    assert reg.get("period_ns") == 50000
    assert Registry(temp_yaml).get("period_ns") == 50000

    assert reg.redo()
    assert reg.get("period_ns") == 12345
    assert Registry(temp_yaml).get("period_ns") == 12345


def test_undo_redo_for_manual_root_save(temp_yaml):
    reg = Registry(temp_yaml)

    reg.root["local_only"] = "temp"
    reg.save()
    assert reg.get("local_only") == "temp"

    assert reg.undo()
    with pytest.raises(KeyError):
        reg.get("local_only")
    with pytest.raises(KeyError):
        Registry(temp_yaml).get("local_only")

    assert reg.redo()
    assert reg.get("local_only") == "temp"
    assert Registry(temp_yaml).get("local_only") == "temp"


def test_undo_history_limit(temp_yaml):
    reg = Registry(temp_yaml, history_size=2)

    reg["period_ns"] = 1
    reg["period_ns"] = 2
    reg["period_ns"] = 3

    assert reg.undo()
    assert reg.get("period_ns") == 2
    assert reg.undo()
    assert reg.get("period_ns") == 1
    assert not reg.undo()


def test_save_to_other_path_does_not_record_history(temp_yaml, tmp_path):
    reg = Registry(temp_yaml)
    export_path = tmp_path / "export.yaml"

    reg.save(export_path)

    assert export_path.exists()
    assert not reg.undo()


def test_undo_restores_external_file_state_before_save(temp_yaml):
    reg = Registry(temp_yaml)
    text = temp_yaml.read_text(encoding="utf-8")
    temp_yaml.write_text(text.replace("50_000", "12345"), encoding="utf-8")

    reg.root["local_only"] = "temp"
    reg.save()

    assert reg.get("period_ns") == 50000
    assert reg.get("local_only") == "temp"

    assert reg.undo()
    assert reg.get("period_ns") == 12345
    with pytest.raises(KeyError):
        reg.get("local_only")


def test_create_false_raises(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        Registry(path, create=False)
