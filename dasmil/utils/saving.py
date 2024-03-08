import torch
from pathlib import Path
from typing import List, Dict, NamedTuple, Sequence, Optional
import numpy as np
import zarr

FEATURES_VERSION = "0.1"


class VersionMismatchError(Exception):
    pass


def check_version(path: Path, version: str = FEATURES_VERSION, throw: bool = True):
    f = zarr.open_group(str(path), mode="r")
    result = f.attrs.get("version", None) == version
    if throw and not result:
        raise VersionMismatchError(f"Version mismatch: expected {version}, got {f.attrs['version']}")
    return result


def ensure_numpy(x):
    return x.numpy() if isinstance(x, torch.Tensor) else x

class LoadedFeatures(NamedTuple):
    feats: np.ndarray
    coords: np.ndarray
    labels: Optional[np.ndarray]
    files: Optional[np.ndarray]


def load_features(
    path: Path,
    remove_classes: Sequence[str] = (),
    augmentations: Optional[Sequence[str]] = None,
    n: Optional[int] = None,
) -> LoadedFeatures:
    n = n or -1
    f = zarr.open_group(str(path), mode="r")
    classes = np.array(f.attrs["classes"]) if "classes" in f.attrs else None

    feats = f["feats"][:n]
    labels = classes[f["labels"][:n]] if classes is not None and "labels" in f else None
    files = f["files"][:n] if "files" in f else None
    coords = f["coords"][:n] if "coords" in f else None

    # Remove classes
    if len(remove_classes) > 0:
        remove_mask = np.isin(labels, remove_classes)
        feats = feats[~remove_mask]
        labels = labels[~remove_mask]
        files = files[~remove_mask]
    return LoadedFeatures(feats=feats,  coords=coords, labels=labels, files=files)
