import shutil
from pathlib import Path

import pytest

from logqbit.registry import Registry


@pytest.fixture
def yaml_copy(tmp_path: Path) -> Path:
    src = Path(__file__).parent / 'data' / 'test.yaml'
    dst = tmp_path / 'test.yaml'
    shutil.copy(src, dst)
    return dst

def test_get_unit_value(yaml_copy):
    reg = Registry(yaml_copy)
    # frr for Q1 is '5.856 GHz' in the test YAML
    val = reg.get('Device/Q1/frr')
    assert hasattr(val, 'unit')
    assert val.unit.name == 'GHz'
    assert abs(val._value - 5.856) < 1e-9


def test_set_and_persistence(yaml_copy):
    reg = Registry(yaml_copy)
    # set a new nested key and ensure it's written to file
    reg['new_section/answer'] = 42

    # creating a new Registry from the file should see the persisted value
    reg2 = Registry(yaml_copy)
    assert reg2.get('new_section/answer') == 42


def test_local_change_not_saved_until_save(yaml_copy):
    reg = Registry(yaml_copy)
    # make a local-only change
    reg.root['local_only'] = 'temp'

    # new Registry instance (reads file) should not see local_only
    reg2 = Registry(yaml_copy)
    with pytest.raises(Exception):
        reg2.get('local_only')

    # after saving, the change should persist
    reg.save()
    reg2.reload()
    assert reg2.get('local_only') == 'temp'

def test_reload_detects_external_change(yaml_copy):
    reg = Registry(yaml_copy)

    # modify the file externally: change period_ns value
    text = yaml_copy.read_text(encoding='utf-8')
    new_text = text.replace('50_000', '12345')
    yaml_copy.write_text(new_text, encoding='utf-8')

    # Change should be detected and reloaded automatically
    assert reg.get('period_ns') == 12345
