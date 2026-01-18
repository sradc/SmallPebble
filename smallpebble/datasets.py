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
Loads MNIST (from OpenML) and CIFAR-10 (from CS.Toronto).
"""

import pickle
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np

DEFAULT_SAVEDIR = Path.home() / ".smallpebble"
CHUNK_SIZE = 1024 * 1024


def _require_extras():
    """Lazy loader for optional dependencies."""
    try:
        import requests
        from tqdm import tqdm

        return requests, tqdm
    except ImportError:
        raise ImportError(
            "Fetching datasets requires 'requests' and 'tqdm'. "
            "Install them with: pip install 'smallpebble[examples]'"
        )


def load_data(name: Literal["mnist", "cifar"], savedir: Path | str = None):
    """
    Load 'mnist' or 'cifar'.
    Returns: X_train, y_train, X_test, y_test
    """
    savedir = Path(savedir) if savedir else DEFAULT_SAVEDIR
    savedir.mkdir(parents=True, exist_ok=True)

    if name == "mnist":
        return _load_mnist(savedir)
    elif name == "cifar":
        return _load_cifar(savedir)
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar'")


def _load_mnist(savedir):
    _, tqdm = _require_extras()

    filename = "mnist_784.arff"
    npy_filename = "mnist.npy"
    url = "https://www.openml.org/data/download/52667/mnist_784.arff"

    # Check cache
    if (savedir / npy_filename).exists():
        data = np.load(savedir / npy_filename)
    else:
        print("Downloading MNIST...")
        filepath = savedir / filename
        _download(url, filepath)

        print("Parsing MNIST...")
        # Basic ARFF parser for MNIST specifically
        data = []
        with open(filepath, "r") as f:
            for line in tqdm(f, total=70000 + 500):  # approx lines
                if line.startswith("@") or line == "\n":
                    continue
                # Parse csv line to integers
                data.append([int(x) for x in line.strip().split(",")])

        data = np.array(data, dtype=np.uint8)
        np.save(savedir / npy_filename, data)
        filepath.unlink()  # Delete ARFF to save space

    # Split
    X = data[:, :-1]
    y = data[:, -1]
    return X[:60000], y[:60000], X[60000:], y[60000:]


def _load_cifar(savedir):
    """
    Downloads the official python version from CS Toronto.
    Format: N x 3072 (stored as [R, G, B] flattened).
    We reshape to N x 32 x 32 x 3 (H, W, C).
    """
    filename = "cifar-10-python.tar.gz"
    npy_filename = "cifar.npy"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if (savedir / npy_filename).exists():
        data_dict = np.load(savedir / npy_filename, allow_pickle=True).item()
        return (
            data_dict["X_train"],
            data_dict["y_train"],
            data_dict["X_test"],
            data_dict["y_test"],
        )

    print("Downloading CIFAR-10...")
    filepath = savedir / filename
    _download(url, filepath)

    print("Extracting CIFAR-10...")
    X_train = []
    y_train = []
    X_test = None
    y_test = None
    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            if "data_batch" in member.name:
                batch = pickle.load(tar.extractfile(member), encoding="bytes")
                X_train.append(batch[b"data"])
                y_train.extend(batch[b"labels"])
            elif "test_batch" in member.name:
                batch = pickle.load(tar.extractfile(member), encoding="bytes")
                X_test = batch[b"data"]
                y_test = batch[b"labels"]

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Preprocess: Reshape to HWC (32, 32, 3)
    # Original is N x 3072 (Channel, Row, Col) -> (N, 3, 32, 32)
    def _reshape(data):
        # N, C, H, W -> N, H, W, C
        return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    X_train = _reshape(X_train)
    X_test = _reshape(X_test)

    # Save as compressed dictionary
    save_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    np.save(savedir / npy_filename, save_dict)

    filepath.unlink()  # Delete tar.gz
    return X_train, y_train, X_test, y_test


def _download(url, filepath):
    requests, tqdm = _require_extras()

    with open(filepath, "wb") as file:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filepath.name) as bar:
            for data in response.iter_content(chunk_size=CHUNK_SIZE):
                file.write(data)
                bar.update(len(data))
