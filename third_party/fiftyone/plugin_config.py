"""Shared FiftyOne plugin path configuration for mmplatform."""

from __future__ import annotations

import os
from pathlib import Path

import fiftyone as fo

PLUGINS_DIR = Path(__file__).resolve().parent / "plugins"
DINO_V1_MANIFEST = (
    PLUGINS_DIR / "mmplatform-cvat" / "models" / "dino-v1-manifest.json"
)


def _configure_model_zoo_manifests() -> None:
    """Register custom zoo models (DINO v1) for server subprocess too."""
    manifest = str(DINO_V1_MANIFEST.resolve())
    if not DINO_V1_MANIFEST.is_file():
        return

    existing = list(fo.config.model_zoo_manifest_paths or [])
    if manifest not in existing:
        existing.append(manifest)

    os.environ["FIFTYONE_MODEL_ZOO_MANIFEST_PATHS"] = os.pathsep.join(existing)
    fo.config.model_zoo_manifest_paths = existing


def configure_plugins() -> Path:
    """Point FiftyOne at mmplatform's local plugins directory.

    FiftyOne App の server は別 subprocess で起動するため、
    ``fo.config.plugins_dir`` だけでは server 側に反映されない。
    ``FIFTYONE_PLUGINS_DIR`` 環境変数も必ず設定する。
    """
    plugins_dir = str(PLUGINS_DIR)
    os.environ["FIFTYONE_PLUGINS_DIR"] = plugins_dir
    fo.config.plugins_dir = plugins_dir
    _configure_model_zoo_manifests()
    return PLUGINS_DIR


def make_cvat_panel_spaces():
    """Initial App layout: Samples + CVAT active-learning panel side by side."""
    from fiftyone.core.odm.workspace import Panel as WorkspacePanel
    from fiftyone.core.odm.workspace import Space

    samples = WorkspacePanel(type="Samples", pinned=True)
    cvat = WorkspacePanel(type="cvat_kcenter_panel")
    return Space(
        children=[samples, cvat],
        orientation="horizontal",
        sizes=[0.65, 0.35],
        active_child=cvat.component_id,
    )
