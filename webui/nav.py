"""Shared top navigation and page layout for all webui pages."""

from __future__ import annotations

PAGE_MAX_WIDTH = "62rem"

BASE_STYLES = f"""
    :root {{ font-family: system-ui, sans-serif; color: #1a1a1a; }}
    body {{
      max-width: {PAGE_MAX_WIDTH};
      margin: 1.5rem auto;
      padding: 0 1rem 2rem;
      line-height: 1.55;
    }}
"""

NAV_STYLES = """
    nav.webui-nav {
      margin-bottom: 1.25rem;
      font-size: 0.95rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem 0.75rem;
      align-items: center;
    }
    nav.webui-nav a { color: #06c; text-decoration: none; }
    nav.webui-nav a:hover { text-decoration: underline; }
"""

_NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("/hub", "統括", "hub"),
    ("/hub/label", "ラベル学習フロー", "label"),
    ("/hub/ppal", "PPAL学習フロー", "ppal_flow"),
    ("/", "データエクスポート", "export"),
    ("/train", "mmdet 学習", "train"),
    ("/ppal-train", "PPAL 学習", "ppal_train"),
    ("/fiftyone-upload", "FiftyOne 画像アップロード", "fiftyone_upload"),
    ("/nuclio-deploy", "モデルデプロイ", "deploy"),
)


def webui_nav_html(*, active: str = "") -> str:
    parts: list[str] = []
    for href, label, key in _NAV_ITEMS:
        if active == key:
            parts.append(f"<strong>{label}</strong>")
        else:
            parts.append(f'<a href="{href}">{label}</a>')
    joined = '\n    <span aria-hidden="true">·</span>\n    '.join(parts)
    return f"""  <nav class="webui-nav">
    {joined}
  </nav>"""
