import inspect
from pathlib import Path

# cvat-sdk の Git ツリー単体には生成物 cvat_sdk.api_client が無いため、sys.path に
# cvat-sdk を足さない。ローカル開発は公式の developer_guide どおり pip install する。
# https://docs.cvat.ai/docs/api_sdk/sdk/developer_guide
REPO_ROOT = Path(__file__).resolve().parent.parent

from cvat_sdk import Client, make_client
from cvat_sdk.core.proxies.types import Location

LOGIN_INFO = REPO_ROOT / "login_info"

CVAT_HOST = "http://localhost:8080"  # 実際の URL・ポートに合わせる
TASK_ID = 1
OUT = REPO_ROOT / "data" / "exports" / f"task_{TASK_ID}_coco.zip"


def _client_with_pat(host: str, token: str) -> Client:
    """Match cvat-sdk core/client.py: PAT via make_client(access_token=) or AccessTokenCredentials."""
    sig = inspect.signature(make_client)
    if "access_token" in sig.parameters:
        return make_client(host, access_token=token)
    try:
        from cvat_sdk.core.client import AccessTokenCredentials
    except ImportError:
        AccessTokenCredentials = None
    if AccessTokenCredentials is not None:
        client = Client(host)
        client.login(AccessTokenCredentials(token))
        return client
    client = Client(host)
    client.api_client.set_default_header("Authorization", f"Token {token}")
    return client


access_token = LOGIN_INFO.read_text(encoding="utf-8").strip()
if not access_token:
    raise SystemExit(f"Personal access token is empty: {LOGIN_INFO}")

OUT.parent.mkdir(parents=True, exist_ok=True)

with _client_with_pat(CVAT_HOST, access_token) as client:
    task = client.tasks.retrieve(TASK_ID)
    path = task.export_dataset(
        format_name="COCO 1.0",
        filename=OUT,
        include_images=True,
        location=Location.LOCAL,
    )
print(path)
