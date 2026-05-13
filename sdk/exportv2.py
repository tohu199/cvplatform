from pathlib import Path
from typing import Dict

from cvat_sdk import Client
from cvat_sdk.core.proxies.types import Location

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGIN_INFO_PATH = REPO_ROOT / "login_info.yaml"

DEFAULT_CVAT_HOST = "http://localhost:8080"
TASK_ID = 1
FORMAT_NAME = "COCO 1.0"
OUT = REPO_ROOT / "data" / "exports" / f"task_{TASK_ID}_coco.zip"


def _load_login_yaml(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


_creds = _load_login_yaml(_LOGIN_INFO_PATH)
USERNAME = _creds.get("USERNAME")
PASSWORD = _creds.get("PASSWORD")
if not USERNAME or not PASSWORD:
    raise SystemExit(
        f"USERNAME and PASSWORD must be set in {_LOGIN_INFO_PATH} (one KEY:value per line)."
    )
CVAT_HOST = (_creds.get("HOST") or "").strip() or DEFAULT_CVAT_HOST

OUT.parent.mkdir(parents=True, exist_ok=True)

with Client(CVAT_HOST) as client:
    client.login((USERNAME, PASSWORD))
    task = client.tasks.retrieve(TASK_ID)
    task.export_dataset(
        format_name=FORMAT_NAME,
        filename=OUT,
        include_images=True,
        location=Location.LOCAL,
    )

print(OUT)
