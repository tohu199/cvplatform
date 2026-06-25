"""HTML for /hub — schematic overview of CVAT ↔ training workflows."""

from __future__ import annotations

import html

from webui.nav import BASE_STYLES, NAV_STYLES, webui_nav_html


def _normalize_url(raw: str, default: str) -> tuple[str, str]:
    """Return (href, visible) with scheme and no trailing slash."""
    value = (raw or "").strip() or default
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    value = value.rstrip("/")
    return html.escape(value, quote=True), html.escape(value, quote=False)


_BASE_STYLES = (
    BASE_STYLES
    + NAV_STYLES
    + """
    h1 { font-size: 1.35rem; margin: 0 0 0.35rem; }
    .lead { color: #444; font-size: 0.95rem; margin: 0 0 1.25rem; }
    .muted { color: #666; font-size: 0.85rem; margin-top: 1.5rem; }
"""
)

_INDEX_STYLES = """
    .flow-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
      gap: 1.1rem;
      margin-top: 0.75rem;
    }
    .flow-card {
      border: 2px solid #2c3e50;
      border-radius: 12px;
      padding: 1.1rem 1.2rem 1.15rem;
      background: linear-gradient(165deg, #fafbfc 0%, #f0f2f5 100%);
      box-shadow: 0 3px 14px rgba(0,0,0,0.07);
      text-decoration: none;
      color: inherit;
      display: block;
      transition: border-color 0.15s, box-shadow 0.15s, transform 0.12s;
    }
    .flow-card:hover {
      border-color: #0a58ca;
      box-shadow: 0 6px 22px rgba(10,88,202,0.12);
      transform: translateY(-1px);
    }
    .flow-card h2 { margin: 0 0 0.4rem; font-size: 1.08rem; }
    .flow-card p { margin: 0; font-size: 0.9rem; color: #444; }
    .flow-card .tap-hint { margin: 0.75rem 0 0; font-size: 0.78rem; color: #627; font-style: italic; }
"""

_DIAGRAM_STYLES = """
    .diagram {
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      justify-content: center;
      gap: 1rem 1.25rem;
      margin-top: 0.5rem;
    }
    .node {
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
    }
    .node:hover {
      border-color: #0a58ca;
      box-shadow: 0 6px 22px rgba(10,88,202,0.12);
      transform: translateY(-1px);
    }
    .node h2 { margin: 0 0 0.35rem; font-size: 1.1rem; letter-spacing: 0.02em; }
    .node-sub { margin: 0 0 0.65rem; font-size: 0.82rem; color: #555; font-weight: 600; }
    .node ul { margin: 0; padding-left: 1.15rem; font-size: 0.9rem; color: #333; }
    .node li { margin: 0.25rem 0; }
    .node .tap-hint { margin: 0.75rem 0 0; font-size: 0.78rem; color: #627; font-style: italic; }
    .flows {
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 1.35rem;
      min-width: 10rem;
      padding: 0.25rem 0.35rem;
    }
    .flow {
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
    }
    .flow:hover { background: #d4ebfa; border-color: #0a58ca; color: #042e4a; }
    .flow .glyph { font-size: 1.35rem; display: block; margin: 0.2rem 0; line-height: 1; }
    .diagram-ppal {
      display: grid;
      flex-wrap: unset;
      align-items: start;
      grid-template-columns: 15rem 12rem 13rem;
      grid-template-rows: auto auto auto;
      gap: 1rem 1.35rem;
      justify-content: center;
      width: fit-content;
      max-width: 100%;
      margin: 0.5rem auto 0;
      padding: 0.25rem;
    }
    .diagram-ppal .node,
    .diagram-ppal .flow {
      flex: unset;
      max-width: 100%;
      min-width: 0;
      box-sizing: border-box;
    }
    .diagram-ppal .flows {
      gap: 1rem;
      min-width: 0;
      width: 100%;
      padding: 0;
      justify-content: flex-start;
    }
    .diagram-ppal .flow {
      padding: 0.5rem 0.55rem;
      font-size: 0.84rem;
    }
    .diagram-ppal .slot-cvat { grid-column: 1; grid-row: 1; }
    .diagram-ppal .slot-flows { grid-column: 2; grid-row: 1; }
    .diagram-ppal .slot-ppal { grid-column: 3; grid-row: 1; }
    .diagram-ppal .slot-fiftyone { grid-column: 1 / 3; grid-row: 2; }
    .diagram-ppal .slot-model { grid-column: 3; grid-row: 2; }
    .diagram-ppal .slot-upload { grid-column: 2; grid-row: 3; }
    .spur-node {
      padding: 0.45rem 0.6rem 0.5rem;
      border: 2px solid #4a6b4a;
      border-radius: 12px;
      background: linear-gradient(165deg, #f8fbf8 0%, #eef4ee 100%);
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      text-decoration: none;
      color: inherit;
      display: block;
      transition: border-color 0.15s, box-shadow 0.15s, transform 0.12s;
    }
    .spur-node:hover { border-color: #3d7a3d; box-shadow: 0 4px 14px rgba(61,122,61,0.1); transform: translateY(-1px); }
    .spur-node h2 { margin: 0; font-size: 0.82rem; line-height: 1.3; }
    .spur-node .tap-hint { margin: 0.3rem 0 0; font-size: 0.68rem; color: #627; font-style: italic; }
    @media (max-width: 44rem) {
      .diagram { flex-direction: column; align-items: center; }
      .flows { flex-direction: row; flex-wrap: wrap; justify-content: center; }
      .node { max-width: 100%; width: 100%; }
      .diagram-ppal {
        grid-template-columns: 1fr;
        width: 100%;
      }
      .diagram-ppal .slot-cvat,
      .diagram-ppal .slot-flows,
      .diagram-ppal .slot-ppal,
      .diagram-ppal .slot-fiftyone,
      .diagram-ppal .slot-model,
      .diagram-ppal .slot-upload {
        grid-column: 1;
        grid-row: auto;
        max-width: 100%;
      }
    }
"""


def hub_index_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>統括 — mmplatform</title>
  <style>{_BASE_STYLES}{_INDEX_STYLES}</style>
</head>
<body>
{webui_nav_html(active="hub")}

  <h1>パイプライン統括</h1>
  <p class="lead">
    mmplatform では、用途に応じて<strong>2 つの学習フロー</strong>を用意しています。
    それぞれの模式図ページで、CVAT と WebUI の各画面のつながりを確認できます。
  </p>

  <section class="flow-cards" aria-label="学習フローの選択">
    <a class="flow-card" href="/hub/label" title="通常のラベル学習フロー">
      <h2>通常のラベル学習フロー</h2>
      <p>
        CVAT でのアノテーション → データエクスポート → MMDetection (v3.x) 学習 →
        モデルデプロイ → CVAT 自動教示
      </p>
      <p class="tap-hint">クリック → フロー模式図を開く</p>
    </a>

    <a class="flow-card" href="/hub/ppal" title="PPAL 能動学習フロー">
      <h2>PPAL 学習フロー</h2>
      <p>
        CVAT でのアノテーション → データエクスポート → PPAL 学習 → FiftyOne での能動学習サンプリング。
        未ラベル画像の追加は FiftyOne 画像アップロードから。
      </p>
      <p class="tap-hint">クリック → フロー模式図を開く</p>
    </a>
  </section>

  <p class="muted">
    各ブロックや矢印をクリックすると、該当する画面（または CVAT / FiftyOne）へ移動します。
  </p>
</body>
</html>
"""


def hub_label_flow_html(cvat_ui_url: str) -> str:
    """Normal label learning flow — ring layout (CVAT ↔ export/train/deploy)."""
    href_cvat, visible_cvat = _normalize_url(cvat_ui_url, "http://localhost:8080")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ラベル学習フロー — mmplatform</title>
  <style>{_BASE_STYLES}{_DIAGRAM_STYLES}</style>
</head>
<body>
{webui_nav_html(active="label")}

  <h1>通常のラベル学習フロー</h1>
  <p class="lead">
    CVAT でのアノテーション作業と、この WebUI 経由の MMDetection 学習・デプロイの流れを模式図で示します。
    <strong>ブロック</strong>や<strong>矢印</strong>をクリックすると、該当する画面（または CVAT）へ移動します。
  </p>

  <section class="diagram" aria-label="CVAT と MMDetection のデータの流れ">
    <a class="node" href="{href_cvat}" target="_blank" rel="noopener noreferrer" title="CVAT を別タブで開く">
      <h2>CVAT</h2>
      <p class="node-sub">ブラウザ上のアノテーション（共通）</p>
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
        <span style="font-weight:500;font-size:0.78rem;color:#345">CVAT → 学習用データ</span>
      </a>
      <a class="flow" href="/nuclio-deploy" title="Nuclio に学習済みモデルを載せる">
        <span style="font-weight:500;font-size:0.78rem;color:#345">学習済みモデル → CVAT</span>
        <span class="glyph" aria-hidden="true">←</span>
        モデルデプロイ
      </a>
    </div>

    <a class="node" href="/train" title="MMDetection 学習">
      <h2>mmdet 学習</h2>
      <p class="node-sub">MMDetection v3.x（この WebUI から実行）</p>
      <ul>
        <li><strong>YOLOX 学習</strong>（CVAT 自動教示向け）</li>
      </ul>
      <p class="tap-hint">クリック → mmdet 学習ページ</p>
    </a>
  </section>

  <p class="muted">
    CVAT の URL は環境変数 <code>MMPLATFORM_CVAT_UI_URL</code> で変更できます（現在: <code>{visible_cvat}</code>）。
    「データエクスポート」は <a href="/">COCO export</a>、「モデルデプロイ」は <a href="/nuclio-deploy">Nuclio デプロイ</a> です。
    <a href="/hub">統括トップ</a>に戻る。
  </p>
</body>
</html>
"""


def hub_ppal_flow_html(cvat_ui_url: str, fiftyone_app_url: str) -> str:
    """PPAL active learning flow — ring layout with upload spur outside FiftyOne."""
    href_cvat, visible_cvat = _normalize_url(cvat_ui_url, "http://localhost:8080")
    href_fo, visible_fo = _normalize_url(fiftyone_app_url, "http://localhost:5151")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PPAL 学習フロー — mmplatform</title>
  <style>{_BASE_STYLES}{_DIAGRAM_STYLES}</style>
</head>
<body>
{webui_nav_html(active="ppal_flow")}

  <h1>PPAL 学習フロー</h1>
  <p class="lead">
    CVAT でのアノテーションと、PPAL 能動学習・FiftyOne サンプリングの流れを模式図で示します。
    <strong>ブロック</strong>や<strong>矢印</strong>をクリックすると、該当する画面（または CVAT / FiftyOne）へ移動します。
  </p>

  <section class="diagram diagram-ppal" aria-label="CVAT と PPAL / FiftyOne のデータの流れ">
    <a class="node slot-cvat" href="{href_cvat}" target="_blank" rel="noopener noreferrer" title="CVAT を別タブで開く">
      <h2>CVAT</h2>
      <p class="node-sub">ブラウザ上のアノテーション（共通）</p>
      <ul>
        <li>ラベル付きデータの作成</li>
        <li>教示の<strong>修正・確認</strong></li>
      </ul>
      <p class="tap-hint">クリック → CVAT（{visible_cvat}）</p>
    </a>

    <div class="flows slot-flows" role="presentation">
      <a class="flow" href="/" title="COCO 形式でエクスポート">
        データエクスポート
        <span class="glyph" aria-hidden="true">→</span>
        <span style="font-weight:500;font-size:0.78rem;color:#345">CVAT → PPAL 用データ</span>
      </a>
      <a class="flow" href="{href_cvat}" target="_blank" rel="noopener noreferrer" title="選定画像を CVAT でアノテーション">
        <span style="font-weight:500;font-size:0.78rem;color:#345">選定画像 → CVAT</span>
        <span class="glyph" aria-hidden="true">←</span>
        アノテーション対象
      </a>
    </div>

    <a class="node slot-ppal" href="/ppal-train" title="PPAL RetinaNet 学習">
      <h2>PPAL 学習</h2>
      <p class="node-sub">能動学習用 RetinaNet</p>
      <ul>
        <li><strong>PPAL 学習</strong>（この WebUI から実行）</li>
      </ul>
      <p class="tap-hint">クリック → PPAL 学習ページ</p>
    </a>

    <a class="node slot-fiftyone" href="{href_fo}" target="_blank" rel="noopener noreferrer" title="FiftyOne App を別タブで開く">
      <h2>FiftyOne</h2>
      <p class="node-sub">能動学習サンプリング</p>
      <ul>
        <li>PPAL プラグインで<strong>選定</strong></li>
        <li>未ラベル画像の優先付け</li>
      </ul>
      <p class="tap-hint">クリック → FiftyOne（{visible_fo}）</p>
    </a>

    <a class="flow slot-model" href="{href_fo}" target="_blank" rel="noopener noreferrer" title="FiftyOne で PPAL サンプリング">
      学習モデル
      <span class="glyph" aria-hidden="true">←</span>
      <span style="font-weight:500;font-size:0.78rem;color:#345">PPAL → FiftyOne</span>
    </a>

    <a class="node spur-node slot-upload" href="/fiftyone-upload" title="FiftyOne へ画像をアップロード">
      <h2>FiftyOne 画像アップロード</h2>
      <p class="tap-hint">クリック → 画像アップロード</p>
    </a>
  </section>

  <p class="muted">
    CVAT: <code>MMPLATFORM_CVAT_UI_URL</code>（現在: <code>{visible_cvat}</code>） /
    FiftyOne: <code>MMPLATFORM_FIFTYONE_APP_URL</code>（現在: <code>{visible_fo}</code>）。
    「データエクスポート」は <a href="/">COCO export</a> です。
    <a href="/hub">統括トップ</a>に戻る。
  </p>
</body>
</html>
"""


def hub_page_html(cvat_ui_url: str) -> str:
    """Backward-compatible alias for the hub index page."""
    return hub_index_html()
