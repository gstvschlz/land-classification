"""Sanity check: extractor output matches the layout the explainability code assumes."""
import sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from land_classification.data.features import extract_one
from land_classification.explain.traditional import feature_layout

names, groups = feature_layout()
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "x.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
    f = extract_one(str(p))
assert f.shape == (len(names),) == (87,), (f.shape, len(names))
assert np.isfinite(f).all()
assert len(set(groups)) == 6
print(f"OK: {len(names)} features, groups {sorted(set(groups))}")
