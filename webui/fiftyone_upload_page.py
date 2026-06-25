"""HTML for /fiftyone-upload."""

from __future__ import annotations

import html

from webui.nav import BASE_STYLES, NAV_STYLES, webui_nav_html


def fiftyone_upload_page_html(
    *,
    default_dataset: str,
    app_url: str,
    upload_root: str,
) -> str:
    ds_visible = html.escape(default_dataset, quote=False)
    app_href = html.escape(app_url.rstrip("/"), quote=True)
    app_visible = html.escape(app_url.rstrip("/"), quote=False)
    upload_visible = html.escape(upload_root, quote=False)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FiftyOne 画像アップロード</title>
  <style>
    {BASE_STYLES}
    {NAV_STYLES}
    label {{ display: block; font-weight: 600; margin-top: 0.85rem; }}
    input[type="text"], select {{ width: 100%; max-width: 28rem; padding: 0.4rem 0.5rem; font-size: 1rem; }}
    input[type="file"] {{ margin-top: 0.35rem; font-size: 0.95rem; }}
    button {{ margin-top: 1rem; margin-right: 0.5rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }}
    .panel {{ margin-top: 0.75rem; padding: 0.85rem 1rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }}
    .muted {{ color: #555; font-size: 0.88rem; }}
    .ok {{ color: #0a6; font-weight: 600; }}
    .err {{ color: #a20; font-weight: 600; }}
    #result {{ margin-top: 1rem; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85rem; }}
    #progress {{ margin-top: 0.75rem; font-weight: 600; }}
    .dropzone {{
      margin-top: 0.5rem; padding: 1.25rem; border: 2px dashed #8aa; border-radius: 8px;
      background: #f7fbff; text-align: center; color: #345;
    }}
    .dropzone.dragover {{ border-color: #06c; background: #eef6ff; }}
  </style>
</head>
<body>
{webui_nav_html(active="fiftyone_upload")}

  <h1>FiftyOne 画像アップロード</h1>
  <p class="muted">
    ブラウザから画像を一括アップロードし、FiftyOne データセットにサンプルとして登録します。
    保存先: <code>{upload_visible}</code> / &lt;dataset&gt; /
  </p>

  <div class="panel">
    <label for="dataset">データセット名</label>
    <input id="dataset" type="text" value="{ds_visible}" autocomplete="off" spellcheck="false" />
    <p class="muted">存在しない名前の場合は新規データセットを作成します。既存データセット一覧は下の select から選べます。</p>

    <label for="dataset_pick">既存データセット（任意）</label>
    <select id="dataset_pick">
      <option value="">— 読み込み中 —</option>
    </select>

    <label for="tags">Tags（任意・カンマ区切り）</label>
    <input id="tags" type="text" placeholder="batch_2026, web_upload" autocomplete="off" />

    <label for="files">画像ファイル（複数選択可）</label>
    <div id="dropzone" class="dropzone">
      ここにドラッグ＆ドロップ、または下のボタンで選択<br />
      <span class="muted">jpg / png / webp / bmp / gif / tif</span>
    </div>
    <input id="files" type="file" accept="image/*" multiple />

    <div id="file_summary" class="muted" style="margin-top:0.5rem"></div>
    <div id="progress"></div>
    <button type="button" id="upload_btn">アップロードして FiftyOne に登録</button>
    <div id="result"></div>
  </div>

  <p class="muted">
    FiftyOne App: <a href="{app_href}" target="_blank" rel="noopener noreferrer">{app_visible}</a>
    （別途 <code>third_party/fiftyone/launch_app.py</code> で起動）
  </p>

  <script>
    const datasetInput = document.getElementById('dataset');
    const datasetPick = document.getElementById('dataset_pick');
    const tagsInput = document.getElementById('tags');
    const filesInput = document.getElementById('files');
    const dropzone = document.getElementById('dropzone');
    const fileSummary = document.getElementById('file_summary');
    const progressEl = document.getElementById('progress');
    const resultEl = document.getElementById('result');
    const uploadBtn = document.getElementById('upload_btn');

    let selectedFiles = [];

    function setFiles(fileList) {{
      selectedFiles = Array.from(fileList || []);
      const n = selectedFiles.length;
      if (!n) {{
        fileSummary.textContent = '';
        return;
      }}
      const names = selectedFiles.slice(0, 5).map(f => f.name).join(', ');
      fileSummary.textContent = n + ' ファイル: ' + names + (n > 5 ? ' …' : '');
    }}

    filesInput.addEventListener('change', () => setFiles(filesInput.files));

    ['dragenter', 'dragover'].forEach(ev => {{
      dropzone.addEventListener(ev, e => {{
        e.preventDefault();
        dropzone.classList.add('dragover');
      }});
    }});
    ['dragleave', 'drop'].forEach(ev => {{
      dropzone.addEventListener(ev, e => {{
        e.preventDefault();
        dropzone.classList.remove('dragover');
      }});
    }});
    dropzone.addEventListener('drop', e => {{
      if (e.dataTransfer && e.dataTransfer.files) {{
        setFiles(e.dataTransfer.files);
        filesInput.files = e.dataTransfer.files;
      }}
    }});

    async function loadDatasets() {{
      try {{
        const r = await fetch('/api/fiftyone/datasets');
        const j = await r.json();
        datasetPick.innerHTML = '<option value="">— 既存から選ぶ —</option>';
        for (const name of (j.datasets || [])) {{
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = name;
          datasetPick.appendChild(opt);
        }}
        const current = datasetInput.value.trim();
        if (current && (j.datasets || []).includes(current)) {{
          datasetPick.value = current;
        }}
      }} catch (err) {{
        datasetPick.innerHTML = '<option value="">（一覧取得失敗）</option>';
      }}
    }}

    datasetPick.addEventListener('change', () => {{
      if (datasetPick.value) datasetInput.value = datasetPick.value;
    }});

    uploadBtn.addEventListener('click', async () => {{
      resultEl.textContent = '';
      resultEl.className = '';
      progressEl.textContent = '';
      const dataset = datasetInput.value.trim();
      if (!dataset) {{
        resultEl.className = 'err';
        resultEl.textContent = 'データセット名を入力してください';
        return;
      }}
      if (!selectedFiles.length) {{
        resultEl.className = 'err';
        resultEl.textContent = '画像ファイルを選択してください';
        return;
      }}

      const form = new FormData();
      form.append('dataset', dataset);
      form.append('tags', tagsInput.value.trim());
      for (const file of selectedFiles) {{
        form.append('files', file, file.name);
      }}

      uploadBtn.disabled = true;
      progressEl.textContent = 'アップロード中… (' + selectedFiles.length + ' ファイル)';
      try {{
        const r = await fetch('/api/fiftyone/upload', {{ method: 'POST', body: form }});
        const j = await r.json();
        if (!r.ok) {{
          resultEl.className = 'err';
          resultEl.textContent = j.detail || JSON.stringify(j);
          return;
        }}
        resultEl.className = 'ok';
        let msg = '完了\\n';
        msg += 'dataset: ' + j.dataset + '\\n';
        msg += 'added: ' + j.added + '\\n';
        msg += 'total samples: ' + j.sample_count + '\\n';
        msg += 'upload_dir: ' + j.upload_dir;
        if (j.skipped && j.skipped.length) {{
          msg += '\\nskipped: ' + j.skipped.join(', ');
        }}
        resultEl.textContent = msg;
        progressEl.textContent = '';
        await loadDatasets();
      }} catch (err) {{
        resultEl.className = 'err';
        resultEl.textContent = String(err);
      }} finally {{
        uploadBtn.disabled = false;
      }}
    }});

    loadDatasets();
  </script>
</body>
</html>
"""
