"""Download the TensorFlow flowers dataset used for training."""

import tarfile
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def download_data(dest: str = "data") -> Path:
    """Download and extract the dataset to *dest*.

    Paramenters
    -----------
    dest: str
        Directory where the extracted ``flower_photos`` folder will be placed.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    tar_path = dest_path / "flower_photos.tgz"
    if not tar_path.exists():
        print(f"Downloading {URL}...")
        urllib.request.urlretrieve(URL, tar_path)
    else:
        print("Archive already downloaded.")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_path)
    return dest_path / "flower_photos"


if __name__ == "__main__":
    download_data()
    print("Dataset ready in 'data/flower_photos'.")
