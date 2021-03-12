"""Load (/download) the MNIST dataset."""
import hashlib
import pathlib
import numpy as np
import requests
from tqdm import tqdm

DEFAULT_SAVEDIR = pathlib.Path.home() / ".smallpebble"

URL = "https://www.openml.org/data/download/52667/mnist_784.arff"
ARFF_FILENAME = "mnist_784.arff"
NUMPY_FILENAME = "mnist_784.npy"
ARFF_SHA256 = "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"
N_IMAGES = 70_000
IMAGE_SIZE = 28 * 28


def load_mnist(savedir=None, delete_intermediate_files=True):
    """Load the MNIST dataset, either from disk or from openml.org.
    See [1] [2] for more information on MNIST.

    Notes:
    Caches in savedir, to avoid redownloading.
    Converts the data into NumPy's 'npy' format, 
    which is smaller and faster to load than 'arff'.
    
    [1] https://www.openml.org/d/554
    [2] http://yann.lecun.com/exdb/mnist
    """
    savedir = pathlib.Path(savedir) if savedir else DEFAULT_SAVEDIR

    if (savedir / NUMPY_FILENAME).is_file():
        images_and_labels = np.load(savedir / NUMPY_FILENAME)
    else:
        savedir.mkdir(exist_ok=True)

        print("Downloading from openml.org")
        download_mnist_784_arff(savedir)
        print("File successfully downloaded and validated.")

        print("Converting file...")
        images_and_labels = arff_to_numpy(savedir)
        print("Successfully converted file.")

        if delete_intermediate_files:
            (savedir / ARFF_FILENAME).unlink()

    images = images_and_labels[:, :IMAGE_SIZE]
    labels = images_and_labels[:, -1]

    train_slice = slice(0, 60_000)
    test_slice = slice(60_000, 70_000)

    X_train = images[train_slice, :]
    y_train = labels[train_slice]
    X_test = images[test_slice, :]
    y_test = labels[test_slice]

    return X_train, y_train, X_test, y_test


def download_mnist_784_arff(savedir):
    """Download file and check hash."""
    with open(savedir / ARFF_FILENAME, "wb") as file:
        response = requests.get(URL, stream=True)
        for data in tqdm(response.iter_content(chunk_size=4096), total=3856):
            file.write(data)
    with open(savedir / ARFF_FILENAME, "rb") as file:
        hashed = hashlib.sha256(file.read()).hexdigest()
    assert ARFF_SHA256 == hashed, "Unexpected file hash."


def arff_lines(savedir):
    "Yield lines from mnist_784.arff"
    with open(savedir / "mnist_784.arff", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            yield line


def arff_to_numpy(savedir):
    """Convert the arff file to NumPy."""
    images_and_labels = np.zeros([N_IMAGES, IMAGE_SIZE + 1], dtype=np.uint8)
    for i, line in tqdm(enumerate(arff_lines(savedir)), total=N_IMAGES):
        if i < 797:
            continue  # skip metadata
        idx = i - 797
        images_and_labels[idx, :] = line.split(",")
    np.save(savedir / NUMPY_FILENAME, images_and_labels)
    return images_and_labels


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist(delete_intermediate_files=False)

    for a in (X_train, y_train, X_test, y_test):
        print(a.shape)
