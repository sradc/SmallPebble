# Copyright 2022-2026 The SmallPebble Authors, Sidney Radcliffe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Minimalist data loader for SmallPebble.
Downloads pre-processed .npz files from GitHub Releases.
"""

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

DEFAULT_SAVEDIR = Path.home() / ".smallpebble"
DATASETS = {
    "mnist": {
        "url": "https://github.com/sradc/smallpebble/releases/download/datasets/mnist.npz",
        "sha256": "14f88124e6bae0c4bbe34adf973c4de4babb37dc2ea068318fbe3d97ee9f3c5e",
    },
    "cifar": {
        "url": "https://github.com/sradc/smallpebble/releases/download/datasets/cifar.npz",
        "sha256": "57c1d901bb9a722adb7e1fea732053e5ebe5bfebfd81fea010171be52b881146",
    },
}


def load_data(name: Literal["mnist", "cifar"], savedir: Path | str = None):
    """
    Load 'mnist' or 'cifar'. Downloads data if not present.
    Returns: X_train, y_train, X_test, y_test
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset must be one of {list(DATASETS.keys())}")
    savedir = Path(savedir) if savedir else DEFAULT_SAVEDIR
    savedir.mkdir(parents=True, exist_ok=True)
    filepath = savedir / f"{name}.npz"
    if not filepath.exists():
        info = DATASETS[name]
        _download(info["url"], filepath, info["sha256"])
    with np.load(filepath) as data:
        return (data["X_train"], data["y_train"], data["X_test"], data["y_test"])


def _download(url, filepath, expected_sha256):
    print(f"Downloading {filepath.name}...")
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Please install 'requests' and 'tqdm' to download datasets.")
    temp_filepath = filepath.with_suffix(".part")
    try:
        with requests.get(url, stream=True) as r:
            # Download
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            sha256_hasher = hashlib.sha256()
            with (
                open(temp_filepath, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True) as bar,
            ):
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    sha256_hasher.update(chunk)
                    bar.update(len(chunk))
            # Verify Hash
            calculated_hash = sha256_hasher.hexdigest()
            if calculated_hash != expected_sha256:
                raise ValueError(
                    f"Security check failed! Hash mismatch for {filepath.name}.\n"
                    f"Expected: {expected_sha256}\n"
                    f"Got:      {calculated_hash}"
                )
        temp_filepath.rename(filepath)
    finally:
        if temp_filepath.exists():
            print(f"Something went wrong, deleting '{temp_filepath}'")
            temp_filepath.unlink()
