"""Download the TensorFlow flowers dataset used for training."""

import argparse
import tarfile
import tempfile
import urllib.request
import pathlib

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def download_dataset(dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        print(f"Downloading dataset to {tmp.name}...")
        urllib.request.urlretrieve(DATA_URL, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(dest)
    print(f"Dataset extracted to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the TensorFlow flowers dataset"
    )
    parser.add_argument("dest", nargs="?", default="data", help="Destination directory")
    args = parser.parse_args()
    download_dataset(pathlib.Path(args.dest))


if __name__ == "__main__":
    main()
