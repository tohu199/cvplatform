# HTML for /nuclio-deploy (separate from server.py for readability).

NUCLIO_DEPLOY_PAGE_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CVAT Nuclio デプロイ</title>
  <style>
    :root { font-family: system-ui, sans-serif; }
    body { max-width: 48rem; margin: 1.5rem auto; padding: 0 1rem; line-height: 1.5; }
    nav { margin-bottom: 1rem; font-size: 0.95rem; }
    nav a { margin-right: 1rem; }
    label { display: block; font-weight: 600; margin-top: 0.75rem; }
    select { width: 100%; max-width: 36rem; padding: 0.4rem 0.5rem; font-size: 1rem; }
    button { margin-top: 0.75rem; padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
    #log { margin-top: 1rem; width: 100%; min-height: 14rem; font-family: ui-monospace, monospace; font-size: 0.78rem; white-space: pre-wrap; background: #111; color: #e8e8e8; padding: 0.6rem; border-radius: 4px; overflow: auto; }
    .muted { color: #555; font-size: 0.88rem; }
    .ok { color: #0a6; font-weight: 600; }
    .err { color: #a20; font-weight: 600; }
    #meta { margin-top: 0.75rem; font-size: 0.9rem; }
  </style>
</head>
<body>
  <nav>
    <a href="/">CVAT export</a>
    <a href="/train">YOLOX 学習</a>
    <a href="/nuclio-deploy">Nuclio デプロイ</a>
  </nav>
  <h1>CVAT 自動アノテ用 Nuclio デプロイ</h1>
  <p class="muted">
    <code>work_dirs/web_train_*</code> を選び、<code>nuctl deploy</code> まで実行します。
    このマシンに <strong>nuctl</strong> と <strong>Docker</strong> があり、CVAT の serverless 用ネットワーク設定が
    <code>third_party/cvat/serverless/deploy_gpu.sh</code> と同じ前提である必要があります。
  </p>

  <label for="wd">work_dir（フォルダ名）</label>
  <select id="wd"></select>
  <div><button type="button" id="btnRefresh">一覧を再読込</button></div>

  <div>
    <button type="button" id="btnDeploy">選択した work_dir をデプロイ</button>
  </div>
  <div id="status"></div>
  <div id="meta"></div>
  <div id="log"></div>

  <script>
    const wd = document.getElementById('wd');
    const log = document.getElementById('log');
    const status = document.getElementById('status');
    const meta = document.getElementById('meta');

    /** Read body as text, then JSON if possible (avoids SyntaxError on HTML/plain 5xx). */
    async function readResponse(r) {
      const raw = await r.text();
      try {
        return { ok: r.ok, raw, json: raw ? JSON.parse(raw) : {} };
      } catch (_) {
        return { ok: r.ok, raw, json: null };
      }
    }

    function formatHttpError(r, raw, json) {
      if (json && json.detail !== undefined) return typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail);
      if (raw && raw.trim()) return raw.trim().slice(0, 8000);
      return 'HTTP ' + r.status;
    }

    async function loadDirs() {
      wd.innerHTML = '';
      const r = await fetch('/api/nuclio/work-dirs', { headers: { Accept: 'application/json' } });
      const { ok, raw, json } = await readResponse(r);
      if (!ok) {
        status.className = 'err';
        status.textContent = formatHttpError(r, raw, json);
        return;
      }
      status.textContent = '';
      const j = json || {};
      const rows = j.work_dirs || [];
      if (!rows.length) {
        const o = document.createElement('option');
        o.value = '';
        o.textContent = '（デプロイ可能な work_dir がありません）';
        wd.appendChild(o);
        return;
      }
      for (const row of rows) {
        const o = document.createElement('option');
        o.value = row.id;
        o.textContent = row.id + '  —  ckpt: ' + row.checkpoint + '  —  nuclio: ' + row.nuclio_function;
        wd.appendChild(o);
      }
    }

    document.getElementById('btnRefresh').addEventListener('click', loadDirs);
    document.getElementById('btnDeploy').addEventListener('click', async () => {
      const id = wd.value;
      if (!id) return;
      log.textContent = '';
      meta.textContent = '';
      status.className = '';
      status.textContent = 'デプロイ中…（Docker ビルドで数分かかることがあります）';
      const btn = document.getElementById('btnDeploy');
      btn.disabled = true;
      try {
        const r = await fetch('/api/nuclio/deploy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ work_dir: id }),
        });
        const { ok: httpOk, raw, json } = await readResponse(r);
        if (!httpOk) {
          status.className = 'err';
          status.textContent = formatHttpError(r, raw, json);
          log.textContent = raw || '';
          return;
        }
        const j = json || {};
        status.className = j.ok ? 'ok' : 'err';
        status.textContent = j.ok ? 'デプロイ完了' : 'デプロイ失敗（終了コード ' + j.exit_code + '）';
        meta.textContent =
          'nuclio: ' + (j.nuclio_function || '') +
          '  |  image: ' + (j.docker_image || '') +
          '  |  classes: ' + JSON.stringify(j.classes || []);
        log.textContent = j.log || '';
        loadDirs();
      } catch (e) {
        status.className = 'err';
        status.textContent = String(e);
      } finally {
        btn.disabled = false;
      }
    });

    loadDirs();
  </script>
</body>
</html>
"""
