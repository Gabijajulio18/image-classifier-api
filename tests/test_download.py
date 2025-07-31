import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import tarfile
import urllib.request

from src.download_data import download_dataset, DATA_URL


class DummyTar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def extractall(self, path):
        self.extracted = path


def test_download_dataset_calls(monkeypatch, tmp_path):
    calls = {}

    def fake_urlretrieve(url, filename):
        calls["url"] = url
        Path(filename).write_text("dummy")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    dummy_tar = DummyTar()
    monkeypatch.setattr(tarfile, "open", lambda name, mode: dummy_tar)

    download_dataset(tmp_path, include_test=False)

    assert calls["url"] == DATA_URL
    assert dummy_tar.extracted == tmp_path
