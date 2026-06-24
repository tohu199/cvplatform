# mmplatform
## 概要(工事中)
- つなぎ部分
  - CVAT で作ったアノテーションをExportして、MMDetection側に渡す

- CVAT
  - 既存物体検出モデルによる自動アノテーション
  - 人間による手動修正

- MMDetection
  - モデルの自動学習

## インストール
```bash
pip install cvat-sdk
```

## 認証
- CVATのアカウント作成
- Personal Access Token (PAT) を作成

## 立ち上げ
メインサーバ
  ```bash
  python webui/server.py

  ```

CVAT の立ち上げ
- third_party/cvat/SETUP.md　を参照


FiftyOne の立ち上げ
- [text](third_party/fiftyone/launch_app.py) を実行
