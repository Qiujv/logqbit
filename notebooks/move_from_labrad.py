# %%
import ast
import json
import os
import socket
from configparser import ConfigParser
from pathlib import Path
from typing import Literal

from labcodes.fileio import labrad
from logqbit.logfolder import yaml
from tqdm.notebook import tqdm

# %%
# folder_in = Path("//moli/data")
folder_in = Path("D:/data/crab.dir")
if ":" in folder_in.parts[0]:
    create_machine = socket.gethostname()
else:
    create_machine = folder_in.as_posix().strip("/").split("/")[0]

folder_out = Path(f"./logqbit_{create_machine}").resolve()

path_pairs: list[tuple[Path, Path, int]] = []
for _path, _, _file_names in folder_in.walk():
    if not _path.name.endswith(".dir"):
        continue
    n_files = len([f for f in _file_names if f.endswith(".csv")])
    if n_files == 0:
        continue
    out_name = _path.relative_to(folder_in).as_posix().replace(".dir", "")
    path_pairs.append((_path, folder_out / out_name, n_files))
    print(f"n_files={n_files:<5}, {out_name}")

# %%
# path_in, path_out, n_files = path_pairs[0]
for path_in, path_out, n_files in path_pairs:

    path_out.mkdir(parents=True, exist_ok=True)

    ini = ConfigParser()
    tag_info: dict[str, set[Literal["star", "trash"]]] = {}
    if ini.read(path_in / "session.ini"):
        tag_info = ast.literal_eval(ini["Tags"]["datasets"])
        tag_info = {int(k[:5]): v for k, v in tag_info.items()}  # Trancate keys to id only.

    start_from: int = max(
        (
            int(entry.name)
            for entry in os.scandir(path_out)
            if entry.is_dir() and entry.name.isdecimal()
        ),
        default=-1,
    )

    for lfi_path in tqdm(path_in.glob("*.csv"), total=n_files, desc=path_in.as_posix()):
        idx = int(lfi_path.name[:5])
        lfo_path = path_out / f"{idx}"
        if lfo_path.exists() and idx != start_from:
            continue
        lfo_path.mkdir(parents=True, exist_ok=True)

        lfi = labrad.read_logfile_labrad(lfi_path)
        lfi.df.to_feather(lfo_path / "data.feather", compression="zstd", compression_level=3)
        with open(lfo_path / "const.yaml", "w", encoding="utf-8") as f:
            yaml.dump(lfi.conf, f)
        _tags = tag_info.get(idx, set())
        idx = {
            "title": lfi.conf['general']['title'],
            "star": "star" in _tags,
            "trash": "trash" in _tags,
            "plot_axes": lfi.indeps,
            "create_time": ''.join(lfi.conf['general']['created'].split(',')),
            "create_machine": create_machine,
        }
        with open(lfo_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(idx, f)

# %%