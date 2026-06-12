"""Download and locate daily model artifacts for hosted apps."""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_MODEL_ROOT = Path('models/daily')
DEFAULT_MODEL_ARTIFACT_URL = (
    'https://github.com/dnperumov/Tennis-Match-Prediction/'
    'releases/download/daily-model-latest/daily-model-latest.tar.gz'
)


def latest_model_dir(model_root: str | Path = DEFAULT_MODEL_ROOT) -> Path | None:
    root = Path(model_root)
    if not root.exists():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (
            (path / 'stacked_model.joblib').exists()
            or (path / 'daily_ensemble_model.pkl').exists()
        )
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def ensure_latest_model_artifact(
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    artifact_url: str | None = None,
) -> Path:
    """Return a local model dir, downloading the latest release artifact if needed."""
    existing = latest_model_dir(model_root)
    if existing is not None:
        return existing

    url = artifact_url or os.getenv('MODEL_ARTIFACT_URL') or DEFAULT_MODEL_ARTIFACT_URL
    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / 'daily-model-latest.tar.gz'
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, 'r:gz') as archive:
            _safe_extract(archive, root)

    downloaded = latest_model_dir(root)
    if downloaded is None:
        raise FileNotFoundError(f'Model artifact downloaded from {url}, but no daily model files were found.')
    return downloaded


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f'Unsafe path in model archive: {member.name}')
    archive.extractall(destination, filter='data')
