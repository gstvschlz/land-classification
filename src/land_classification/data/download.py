"""Download and unpack EuroSAT-RGB (Helber et al., 2019)."""
from __future__ import annotations
import hashlib
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

# Zenodo mirror of the original DFKI release: 27 000 RGB patches, 64x64, 10 classes.
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
DIRNAME = "EuroSAT_RGB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_eurosat(data_root: Path) -> Path:
    """Return ``data_root/EuroSAT_RGB``, downloading + verifying it if missing."""
    data_root = Path(data_root)
    target = data_root / DIRNAME
    if target.is_dir() and any(target.iterdir()):
        return target

    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / f"{DIRNAME}.zip"
    if not archive.exists():
        print(f"Downloading {URL} (~95 MB)")
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as r, open(archive, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if _sha256(archive) != SHA256:
        archive.unlink()
        raise RuntimeError(f"Checksum mismatch for {URL}; the corrupt download was deleted, rerun.")

    with zipfile.ZipFile(archive) as z:
        z.extractall(data_root)
    archive.unlink()
    return target
