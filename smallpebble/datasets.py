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

from pathlib import Path
from typing import Literal

import numpy as np

DEFAULT_SAVEDIR = Path.home() / ".smallpebble"
BASE_URL = "https://github.com/sradc/smallpebble/releases/download/datasets/"


def load_data(name: Literal["mnist", "cifar"], savedir: Path | str = None):
    """
    Load 'mnist' or 'cifar'. Downloads data if not present.
    Returns: X_train, y_train, X_test, y_test
    """
    if name not in ["mnist", "cifar"]:
        raise ValueError("Dataset must be 'mnist' or 'cifar'")

    savedir = Path(savedir) if savedir else DEFAULT_SAVEDIR
    savedir.mkdir(parents=True, exist_ok=True)

    filepath = savedir / f"{name}.npz"
    if not filepath.exists():
        _download(BASE_URL + f"{name}.npz", filepath)

    with np.load(filepath) as data:
        return (data["X_train"], data["y_train"], data["X_test"], data["y_test"])


def _download(url, filepath):
    print(f"Downloading {filepath.name}...")
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Please install 'requests' and 'tqdm' to download datasets.")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(filepath, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        if filepath.exists():
            filepath.unlink()  # Cleanup partial file
        raise e
