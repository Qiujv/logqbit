from pathlib import Path

import pytest
from ruamel.yaml.representer import RepresenterError

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


def test_failed_set_recovers_on_next_synchronized_operation(temp_yaml):
    class Unsupported:
        pass

    reg = Registry(temp_yaml)
    original_parser = reg._yaml.parser

    with pytest.raises(RepresenterError):
        reg["invalid"] = Unsupported()

    assert "_parser" not in reg._yaml.__dict__
    assert isinstance(reg.get_local("invalid"), Unsupported)
    assert not reg.undo()
    assert not list(temp_yaml.parent.glob(f"{temp_yaml.stem}*.tmp"))

    reg["valid"] = 1
    assert reg._yaml.parser is not original_parser
    assert Registry(temp_yaml)["valid"] == 1
    with pytest.raises(KeyError):
        Registry(temp_yaml)["invalid"]


def test_reload_discards_dirty_local_changes(temp_yaml):
    reg = Registry(temp_yaml)
    reg.set_local("local_only", "temp")

    assert reg.get_local("local_only") == "temp"

    reg.reload()

    with pytest.raises(KeyError):
        reg.get_local("local_only")
    assert not reg._root_dirty


def test_primary_save_clears_root_dirty(temp_yaml):
    reg = Registry(temp_yaml)
    reg.set_local("local_only", "saved")

    reg.save()

    assert not reg._root_dirty
    reg.reload()
    assert reg.get_local("local_only") == "saved"


def test_save_to_other_path_keeps_primary_root_dirty(temp_yaml, tmp_path):
    reg = Registry(temp_yaml)
    reg.set_local("local_only", "exported")

    export_path = tmp_path / "export.yaml"
    reg.save(export_path)

    assert reg._root_dirty
    assert Registry(export_path)["local_only"] == "exported"

    reg.reload()
    with pytest.raises(KeyError):
        reg.get_local("local_only")


def test_failed_print_local_does_not_poison_yaml(temp_yaml):
    class Unsupported:
        pass

    reg = Registry(temp_yaml)
    original_parser = reg._yaml.parser
    reg.set_local("invalid", Unsupported())

    with pytest.raises(RepresenterError):
        reg.print_local()

    assert "_parser" not in reg._yaml.__dict__
    reg.reload()
    reg["valid"] = 1
    assert reg._yaml.parser is not original_parser
    assert Registry(temp_yaml)["valid"] == 1


def test_failed_load_does_not_poison_yaml(temp_yaml):
    reg = Registry(temp_yaml)
    original_parser = reg._yaml.parser
    temp_yaml.write_text("invalid: [", encoding="utf-8")

    with pytest.raises(Exception):
        reg.load()

    assert "_parser" not in reg._yaml.__dict__
    temp_yaml.write_text("valid: 1\n", encoding="utf-8")
    assert reg.load()["valid"] == 1
    assert reg._yaml.parser is not original_parser
