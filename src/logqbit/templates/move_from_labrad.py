"""Template script for migrating data from LabRAD format to logqbit format.

This script needs to be customized for your specific data directory structure.
Copy this file to your working directory and modify the paths as needed.

Usage:
    1. Copy this template: python -m logqbit.cli copy-template move_from_labrad
    2. Edit the copied script to set your input/output paths
    3. Run: python move_from_labrad.py
"""

# %%
import ast
import json
import os
import socket
from configparser import ConfigParser
from pathlib import Path
from typing import Literal

from labcodes.fileio import labrad
from tqdm.notebook import tqdm

from logqbit.logfolder import yaml

# %%
# ============================================================================
# Configuration - EDIT THESE PATHS FOR YOUR DATA
# ============================================================================

# Input: Your LabRAD data directory
# Examples:
#   - Network path: Path("//moli/data")
#   - Local path: Path("D:/data/crab.dir")
folder_in = Path("D:/data/crab.dir")

# Auto-detect machine name from path
if ":" in folder_in.parts[0]:
    create_machine = socket.gethostname()
else:
    create_machine = folder_in.as_posix().strip("/").split("/")[0]

# Output: Where to save converted logqbit data
# Will be created if it doesn't exist
# folder_out will be auto-generated based on folder_in, or you can specify it directly
# folder_out = Path("./my_converted_data").resolve()

folder_out = Path(f"./logqbit_{create_machine}").resolve()

# ============================================================================
# End of Configuration
# ============================================================================

# %%

# Scan for all .dir folders containing .csv files
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
# Process each directory
for path_in, path_out, n_files in path_pairs:

    path_out.mkdir(parents=True, exist_ok=True)

    # Read LabRAD session tags (star/trash)
    ini = ConfigParser()
    tag_info: dict[str, set[Literal["star", "trash"]]] = {}
    if ini.read(path_in / "session.ini"):
        tag_info = ast.literal_eval(ini["Tags"]["datasets"])
        tag_info = {int(k[:5]): v for k, v in tag_info.items()}  # Truncate keys to id only

    # Find last converted file to support resuming
    start_from: int = max(
        (
            int(entry.name)
            for entry in os.scandir(path_out)
            if entry.is_dir() and entry.name.isdecimal()
        ),
        default=-1,
    )

    # Convert each .csv file
    for lfi_path in tqdm(path_in.glob("*.csv"), total=n_files, desc=path_in.as_posix()):
        idx = int(lfi_path.name[:5])
        lfo_path = path_out / f"{idx}"
        
        # Skip already converted files (except the last one, in case it was incomplete)
        if lfo_path.exists() and idx != start_from:
            continue
            
        lfo_path.mkdir(parents=True, exist_ok=True)

        # Read LabRAD format
        lfi = labrad.read_logfile_labrad(lfi_path)
        
        # Write data.feather
        lfi.df.to_feather(lfo_path / "data.feather", compression="zstd", compression_level=3)
        
        # Write const.yaml
        with open(lfo_path / "const.yaml", "w", encoding="utf-8") as f:
            yaml.dump(lfi.conf, f)
        
        # Write metadata.json
        _tags = tag_info.get(idx, set())
        metadata = {
            "title": lfi.conf['general']['title'],
            "star": "star" in _tags,
            "trash": "trash" in _tags,
            "plot_axes": lfi.indeps,
            "create_time": ''.join(lfi.conf['general']['created'].split(',')),
            "create_machine": create_machine,
        }
        with open(lfo_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

print("\n✓ Migration complete!")
print(f"Converted data saved to: {folder_out}")
