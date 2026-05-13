from pathlib import Path
from pprint import pprint

from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.api_client.models import *
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGIN_INFO_PATH = REPO_ROOT / "login_info.yaml"


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

# Set up an API client
# Read Configuration class docs for more info about parameters and authentication methods
configuration = Configuration(
    host="http://localhost:8080",
    username=USERNAME,
    password=PASSWORD,
)

with ApiClient(configuration) as api_client:
    format = "COCO 1.0" # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats
    id = 1 # int | A unique integer value identifying this task.
    cloud_storage_id = 1 # int | Storage id (optional)
    filename = "task_1_coco.zip" # str | Desired output file name (optional)
    location = "local" # str | Where need to save downloaded dataset (optional)
    save_images = True # bool | Include images or not (optional) if omitted the server will use the default value of False

    try:
        (data, response) = api_client.tasks_api.create_dataset_export(
            format,
            id,
            cloud_storage_id=cloud_storage_id,
            filename=filename,
            location=location,
            save_images=save_images,
        )
        pprint(data)
    except exceptions.ApiException as e:
        print("Exception when calling TasksApi.create_dataset_export(): %s\n" % e)
