"""Download the TensorFlow flowers dataset used for training."""

import argparse
import tarfile
import tempfile
import urllib.request
import pathlib
import random
import shutil

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def _create_test_split(source: pathlib.Path, target: pathlib.Path, count: int) -> None:
    """Create a small test split by copying ``count`` images per class."""

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*"))
        sample = random.sample(images, min(count, len(images)))
        dest_dir = target / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in sample:
            shutil.copy(img, dest_dir / img.name)


def download_dataset(
    dest: pathlib.Path, include_test: bool = False, test_count: int = 10
) -> None:
    """Download and optionally create a small test set."""

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        print(f"Downloading dataset to {tmp.name}...")
        urllib.request.urlretrieve(DATA_URL, tmp.name)
        with tarfile.open(tmp.name, "r:gz") as tar:
            tar.extractall(dest)

    if include_test:
        source = dest / "flower_photos"
        target = dest / "flower_photos_test"
        _create_test_split(source, target, test_count)
    print(f"Dataset extracted to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the TensorFlow flowers dataset"
    )
    parser.add_argument("dest", nargs="?", default="data", help="Destination directory")
    parser.add_argument(
        "--with-test",
        action="store_true",
        help="Also create a small test split for evaluation",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=10,
        help="Number of images per class for the test split",
    )
    args = parser.parse_args()
    download_dataset(
        pathlib.Path(args.dest), include_test=args.with_test, test_count=args.test_count
    )


if __name__ == "__main__":
    main()
