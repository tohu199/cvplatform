"""HTML for /hub — schematic overview of CVAT ↔ MMDetection workflow."""

from __future__ import annotations

import html


def hub_page_html(cvat_ui_url: str) -> str:
    """cvat_ui_url: base URL of the CVAT web UI (e.g. http://localhost:8080)."""
    raw = (cvat_ui_url or "").strip() or "http://localhost:8080"
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    raw = raw.rstrip("/")
    href_cvat = html.escape(raw, quote=True)
    visible_cvat = html.escape(raw, quote=False)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>統括 — mmplatform</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; color: #1a1a1a; }}
    body {{ max-width: 58rem; margin: 1.5rem auto; padding: 0 1rem 2rem; line-height: 1.55; }}
    nav {{ margin-bottom: 1.25rem; font-size: 0.95rem; display: flex; flex-wrap: wrap; gap: 0.35rem 0.75rem; align-items: center; }}
    nav a {{ color: #06c; text-decoration: none; }}
    nav a:hover {{ text-decoration: underline; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.35rem; }}
    .lead {{ color: #444; font-size: 0.95rem; margin: 0 0 1.25rem; }}
    .muted {{ color: #666; font-size: 0.85rem; margin-top: 1.5rem; }}
    .diagram {{
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      justify-content: center;
      gap: 1rem 1.25rem;
      margin-top: 0.5rem;
    }}
    .node {{
      flex: 1 1 14rem;
      max-width: 20rem;
      border: 2px solid #2c3e50;
      border-radius: 12px;
      padding: 1rem 1.15rem 1.1rem;
      background: linear-gradient(165deg, #fafbfc 0%, #f0f2f5 100%);
      box-shadow: 0 3px 14px rgba(0,0,0,0.07);
      text-decoration: none;
      color: inherit;
      display: block;
      transition: border-color 0.15s, box-shadow 0.15s, transform 0.12s;
    }}
    .node:hover {{
      border-color: #0a58ca;
      box-shadow: 0 6px 22px rgba(10,88,202,0.12);
      transform: translateY(-1px);
    }}
    .node h2 {{ margin: 0 0 0.35rem; font-size: 1.1rem; letter-spacing: 0.02em; }}
    .node-sub {{ margin: 0 0 0.65rem; font-size: 0.82rem; color: #555; font-weight: 600; }}
    .node ul {{ margin: 0; padding-left: 1.15rem; font-size: 0.9rem; color: #333; }}
    .node li {{ margin: 0.25rem 0; }}
    .node .tap-hint {{ margin: 0.75rem 0 0; font-size: 0.78rem; color: #627; font-style: italic; }}
    .flows {{
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 1.35rem;
      min-width: 10rem;
      padding: 0.25rem 0.35rem;
    }}
    .flow {{
      display: block;
      text-align: center;
      padding: 0.55rem 0.65rem;
      border-radius: 10px;
      border: 2px dashed #3d7ea6;
      background: #e8f4fc;
      color: #0b4f7a;
      font-weight: 600;
      font-size: 0.88rem;
      text-decoration: none;
      line-height: 1.35;
      transition: background 0.15s, border-color 0.15s;
    }}
    .flow:hover {{ background: #d4ebfa; border-color: #0a58ca; color: #042e4a; }}
    .flow .glyph {{ font-size: 1.35rem; display: block; margin: 0.2rem 0; line-height: 1; }}
    @media (max-width: 44rem) {{
      .diagram {{ flex-direction: column; align-items: center; }}
      .flows {{ flex-direction: row; flex-wrap: wrap; justify-content: center; }}
      .node {{ max-width: 100%; width: 100%; }}
    }}
  </style>
</head>
<body>
  <nav>
    <a href="/hub"><strong>統括</strong></a>
    <span aria-hidden="true">·</span>
    <a href="/">CVAT export</a>
    <span aria-hidden="true">·</span>
    <a href="/fiftyone-upload">FiftyOne upload</a>
    <span aria-hidden="true">·</span>
    <a href="/train">YOLOX 学習</a>
    <span aria-hidden="true">·</span>
    <a href="/ppal-train">PPAL 学習</a>
    <span aria-hidden="true">·</span>
    <a href="/nuclio-deploy">Nuclio デプロイ</a>
  </nav>

  <h1>パイプライン統括</h1>
  <p class="lead">
    CVAT でのアノテーション作業と、この WebUI 経由の MMDetection 学習・デプロイの流れを模式図で示します。
    <strong>ブロック</strong>や<strong>矢印</strong>をクリックすると、該当する画面（または CVAT）へ移動します。
  </p>

  <section class="diagram" aria-label="CVAT と MMDetection のデータの流れ">
    <a class="node" href="{href_cvat}" target="_blank" rel="noopener noreferrer" title="CVAT を別タブで開く">
      <h2>CVAT</h2>
      <p class="node-sub">ブラウザ上のアノテーション</p>
      <ul>
        <li>モデル<strong>自動教示</strong></li>
        <li>人間の<strong>教示修正</strong></li>
      </ul>
      <p class="tap-hint">クリック → CVAT（{visible_cvat}）</p>
    </a>

    <div class="flows" role="presentation">
      <a class="flow" href="/" title="COCO 形式でエクスポート">
        データエクスポート
        <span class="glyph" aria-hidden="true">→</span>
        <span style="font-weight:500;font-size:0.78rem;color:#345">CVAT → MMDetection 用データ</span>
      </a>
      <a class="flow" href="/nuclio-deploy" title="Nuclio に学習済みモデルを載せる">
        <span style="font-weight:500;font-size:0.78rem;color:#345">学習済みモデル → CVAT</span>
        <span class="glyph" aria-hidden="true">←</span>
        モデルデプロイ
      </a>
    </div>

    <a class="node" href="/train" title="YOLOX 学習">
      <h2>MMDetection</h2>
      <p class="node-sub">この WebUI から実行</p>
      <ul>
        <li><strong>YOLOX 学習</strong>（CVAT 自動教示向け）</li>
      </ul>
      <p class="tap-hint">クリック → YOLOX 学習ページ</p>
    </a>

    <a class="node" href="/ppal-train" title="PPAL RetinaNet 学習">
      <h2>PPAL</h2>
      <p class="node-sub">能動学習用 RetinaNet</p>
      <ul>
        <li><strong>PPAL 学習</strong>（FiftyOne サンプリング向け）</li>
      </ul>
      <p class="tap-hint">クリック → PPAL 学習ページ</p>
    </a>
  </section>

  <p class="muted">
    CVAT の URL は環境変数 <code>MMPLATFORM_CVAT_UI_URL</code> で変更できます（現在: <code>{visible_cvat}</code>）。
    「データエクスポート」は <a href="/">COCO export</a>、「モデルデプロイ」は <a href="/nuclio-deploy">Nuclio デプロイ</a> です。
  </p>
</body>
</html>
"""
