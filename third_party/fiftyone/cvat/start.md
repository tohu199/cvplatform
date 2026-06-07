FiftyOne と CVAT の連携は、**「FiftyOne から CVAT にアノテーションタスクを投げて、完了後に結果を戻す」**形で行います。  
基本は以下の流れです。

## 1) 事前準備

- CVAT サーバーを用意（ローカル or リモート）
- FiftyOne をインストール
- CVAT にログインできるユーザー/API トークンを確認

```bash
pip install fiftyone
```

## 2) FiftyOne 側で CVAT 接続情報を設定

環境変数または設定ファイルで CVAT 接続を定義します（代表例）。

```bash
export FIFTYONE_CVAT_URL="http://localhost:8080"
export FIFTYONE_CVAT_USERNAME="your_user"
export FIFTYONE_CVAT_PASSWORD="your_pass"
# または token を使う構成
```

> 実運用ではパスワード直書きより、`.env` やシークレット管理を推奨です。

## 3) データセットを読み込み、CVAT にアノテーション依頼を作成

FiftyOne では `annotate()` を使って外部アノテーションバックエンド（CVAT）にジョブを作れます。

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

# 例: 検出タスクをCVATへ
anno_key = "cvat_run_001"

dataset.annotate(
    anno_key,
    label_field="ground_truth",
    backend="cvat",
    label_type="detections",   # tasksに応じて変更
    classes=["person", "car", "dog"],
)
```

## 4) CVAT 上で作業 → FiftyOne に取り込み

CVAT でアノテーションを終えたら、FiftyOne に結果をロードします。

```python
dataset.load_annotations(anno_key)
```

必要ならクリーンアップ：

```python
dataset.delete_annotation_run(anno_key)
```

## 5) よくあるハマりどころ

- CVAT URL が誤っている（`http/https`、ポート）
- CVAT 側権限不足（プロジェクト/タスク作成権限）
- `label_type` と `label_field` の不一致
- Docker ネットワーク越しに URL 到達不可（`localhost` の向き）

---

## 6) mmplatform カスタム Plugin（k-center + CVAT 送信）

FiftyOne App 内で **k-center greedy 順に並べ替え → 1 枚以上を選んで CVAT 送信** する最小 Plugin です。

### 重要: パネルの場所

**右サイドバー（フィールド表示）ではありません。**

| 開き方 | 操作 |
|--------|------|
| **自動（推奨）** | `launch_app.py` / `quickstart.py` 起動時に Samples の横へ表示 |
| **+ ボタン** | 画面上部 **Samples タブ横の「+」** → **「CVAT: k-center 選定」** |
| **Operator browser** | `/` キーまたはブラウザアイコン → **「Open CVAT k-center panel」** |

### インストール

```bash
# 方法 A（推奨）: プラグイン付き起動スクリプト
python third_party/fiftyone/launch_app.py --dataset quickstart

# 方法 B: quickstart.py（プラグイン + パネル自動表示を同梱）
python third_party/fiftyone/quickstart.py

# 方法 C: 環境変数で plugins ディレクトリを指定
export FIFTYONE_PLUGINS_DIR="/path/to/mmplatform/third_party/fiftyone/plugins"
python third_party/fiftyone/quickstart.py

# 方法 D: デフォルト plugins ディレクトリへ symlink
bash third_party/fiftyone/plugins/install_plugin.sh
```

起動後、プラグイン読み込み確認:

```bash
FIFTYONE_PLUGINS_DIR=/path/to/mmplatform/third_party/fiftyone/plugins fiftyone plugins list
# @mmplatform/cvat-kcenter が表示されれば OK
```

### CVAT 接続（必須）

```bash
export FIFTYONE_CVAT_URL="http://localhost:8080"
export FIFTYONE_CVAT_USERNAME="your_user"
export FIFTYONE_CVAT_PASSWORD="your_pass"
```

### App での使い方

1. **k-center で並べ替え** — embedding 計算後、多様性順に View を更新し先頭 10 枚を選択
2. グリッドで **1 枚以上** を選択
3. **CVAT に送信** — `annotate(backend="cvat")` でタスク作成

### パネルが見えない / "no longer exists!" と出るとき

1. **起動方法** — `launch_app.py` または更新済み `quickstart.py` を使う（`fo.config` だけでは不十分）
2. **古い server が残っている** — 5151 番ポートで以前起動した FiftyOne を Ctrl+C で止めてから再起動
3. **環境変数** — server subprocess 向けに必須:
   ```bash
   export FIFTYONE_PLUGINS_DIR="/path/to/mmplatform/third_party/fiftyone/plugins"
   python third_party/fiftyone/launch_app.py
   ```
4. **右サイドバーではない** — Samples タブ横の **+** または Operator browser を確認

**原因メモ:** App server は別プロセスで動くため、Python 内の `fo.config.plugins_dir = ...` だけでは plugin が server に載らず、パネル登録（`register_panel`）が走らない。その結果 `Panel "cvat_kcenter_panel" no longer exists!` になる。

### ファイル構成

```
third_party/fiftyone/plugins/mmplatform-cvat/
  fiftyone.yml
  __init__.py      # Panel + Operator
  kcenter.py       # k-center greedy
third_party/fiftyone/plugin_config.py
third_party/fiftyone/launch_app.py
```