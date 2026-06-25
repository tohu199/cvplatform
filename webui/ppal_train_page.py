# HTML for /ppal-train (PPAL RetinaNet active-learning training).

from webui.nav import BASE_STYLES, NAV_STYLES, webui_nav_html

PPAL_TRAIN_PAGE_HTML = (
"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PPAL 学習</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
"""
    + BASE_STYLES
    + NAV_STYLES
    + """
    label { display: block; font-weight: 600; margin-top: 0.75rem; }
    input, select { max-width: 100%; padding: 0.35rem 0.5rem; font-size: 1rem; }
    input[type="number"] { max-width: 10rem; }
    .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end; margin-top: 0.5rem; }
    .row > div { flex: 1 1 12rem; }
    button { margin-top: 1rem; margin-right: 0.5rem; padding: 0.45rem 0.9rem; font-size: 1rem; cursor: pointer; }
    #log { width: 100%; height: 12rem; font-family: ui-monospace, monospace; font-size: 0.78rem; overflow: auto; background: #111; color: #e8e8e8; padding: 0.5rem; border-radius: 4px; white-space: pre-wrap; }
    .chart-panel { margin-top: 1rem; max-width: 100%; height: 220px; position: relative; }
    .muted { color: #555; font-size: 0.88rem; }
    .err { color: #a22; }
    .ok { color: #080; }
    .status { margin-top: 0.5rem; font-weight: 600; }
    .data-panel { margin-top: 0.5rem; padding: 0.75rem; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
    .work-dir-name-row { display: flex; align-items: stretch; flex-wrap: wrap; max-width: 36rem; margin-top: 0.25rem; }
    .work-dir-prefix {
      display: flex; align-items: center; font-family: ui-monospace, monospace; font-size: 1rem;
      padding: 0.35rem 0.55rem; background: #f0f0f0; border: 1px solid #ccc; border-right: none;
      border-radius: 4px 0 0 4px; color: #333;
    }
    #work_dir_suffix { flex: 1 1 12rem; min-width: 10rem; border-radius: 0 4px 4px 0; }
    #categoryChecks { margin-top: 0.5rem; padding: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; max-height: 9rem; overflow: auto; }
    #categoryChecks label { display: inline-block; margin: 0 0.75rem 0.35rem 0; font-weight: 400; font-size: 0.88rem; }
    #labeled_sources, #val_sources { width: 100%; font-family: ui-monospace, monospace; font-size: 0.82rem; }
    .source-select { min-height: 7rem; }
    #metricChecks { margin-top: 0.5rem; padding: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; max-height: 10rem; overflow: auto; }
    #metricChecks label { display: inline-block; margin: 0 0.75rem 0.35rem 0; font-weight: 400; font-size: 0.88rem; }
  </style>
</head>
<body>
"""
    + webui_nav_html(active="ppal_train")
    + """
  <h1>PPAL RetinaNet 学習</h1>
  <p class="muted">
    CVAT でアノテーション済みの <strong>labeled / val データ</strong>で PPAL 用 RetinaNet を学習します。
    アクティブラーニングでは <strong>過去ラウンド + 今回</strong>の labeled export を複数選択してください（YOLOX 学習と同様に ConcatDataset で結合）。
    ハイパーパラメータの既定値は PPAL 本家（<code>retinanet_26e.py</code> / <code>ppal_retinanet_coco.py</code>）に合わせています。
    出力は <code>third_party/PPAL/work_dirs/web_ppal_*</code> に保存され、FiftyOne の PPAL サンプリングで利用できます。
  </p>

  <label for="work_dir_suffix">出力 work_dir 名</label>
  <div class="work-dir-name-row">
    <span class="work-dir-prefix" id="work_dir_prefix">web_ppal_</span>
    <input id="work_dir_suffix" type="text" autocomplete="off" spellcheck="false" />
  </div>
  <p class="muted">空欄のときは自動生成。接尾辞は英数字と <code>_</code> <code>-</code> のみ。</p>

  <section class="data-panel">
    <h2 style="margin:0 0 0.5rem;font-size:1.05rem">Labeled データ</h2>
    <p class="muted">学習用アノテーション（CVAT エクスポート COCO）。Ctrl / Cmd + クリックで複数選択。</p>
    <select id="labeled_sources" class="source-select" multiple aria-label="Labeled データ"></select>
  </section>

  <section class="data-panel">
    <h2 style="margin:0 0 0.5rem;font-size:1.05rem">Val データ</h2>
    <p class="muted">検証用アノテーション。複数選択可（各 export の画像パスを維持して ConcatDataset で評価）。</p>
    <select id="val_sources" class="source-select" multiple aria-label="Val データ"></select>
  </section>

  <section class="data-panel">
    <h2 style="margin:0 0 0.5rem;font-size:1.05rem">カテゴリ</h2>
    <div id="categoryChecks"><span class="muted">（Labeled / Val を選択してください）</span></div>
  </section>

  <label for="pretrained">事前学習（PPAL work_dirs）</label>
  <select id="pretrained"></select>
  <p class="muted">未選択時は ResNet 事前学習から開始。選択時は前回の PPAL checkpoint から継続学習します。</p>

  <p class="muted">config: <code id="config_hint">configs/coco_active_learning/al_train/retinanet_26e.py</code>（固定）</p>

  <div class="row">
    <div><label for="max_epochs">max_epochs</label><input id="max_epochs" type="number" min="1" value="26" /></div>
    <div><label for="lr">学習率 (lr)</label><input id="lr" type="number" step="any" value="0.01" /></div>
    <div><label for="batch_size">batch_size (samples_per_gpu)</label><input id="batch_size" type="number" min="1" value="1" /></div>
  </div>

  <div>
    <button type="button" id="btnStart">学習開始</button>
    <button type="button" id="btnStop">停止</button>
  </div>
  <div id="status" class="status"></div>
  <div id="workdir" class="muted"></div>
  <div id="logjsonpath" class="muted"></div>

  <h2>メトリック（折れ線）</h2>
  <p class="muted">初期表示: <strong>loss</strong> / <strong>memory</strong> / <strong>coco/bbox_mAP</strong>。PPAL の <code>*.log.json</code> から読み取ります。</p>
  <div id="metricChecks"></div>

  <h3>loss 系（loss, loss_cls, …）</h3>
  <div class="chart-panel"><canvas id="cLoss"></canvas></div>
  <h3>COCO bbox mAP（validation）</h3>
  <div class="chart-panel"><canvas id="cMap"></canvas></div>
  <h3>memory / lr / time など</h3>
  <div class="chart-panel"><canvas id="cSys"></canvas></div>

  <h2>ログ</h2>
  <div id="log"></div>

  <script>
    const STORAGE_KEY = 'mmplatform_ppal_train_selected_metrics';
    let datasets = [];
    let lastState = {};
    let defaultMetrics = ['loss', 'memory', 'coco/bbox_mAP'];
    const chartRefs = { loss: null, map: null, sys: null };
    let prevAvailKey = '';
    let prevWorkDirForCharts = undefined;
    let suggestedCategoryNames = [];
    let categoryFetchTimer = null;

    function buildSourceOptions(dsList) {
      const opts = [];
      for (const d of dsList) {
        const prefixes = d.image_prefixes && d.image_prefixes.length ? d.image_prefixes : ['images/'];
        for (const ann of d.annotations || []) {
          for (const prefix of prefixes) {
            const item = { data_root_rel: d.data_root_rel, ann_file: ann, img_prefix: prefix };
            opts.push({
              value: JSON.stringify(item),
              label: d.id + '  |  ' + ann + '  |  ' + prefix,
            });
          }
        }
      }
      return opts;
    }

    function getSelectedSources(selectId) {
      const el = document.getElementById(selectId);
      const out = [];
      for (const opt of el.selectedOptions) {
        if (!opt.value) continue;
        try { out.push(JSON.parse(opt.value)); } catch (e) {}
      }
      return out;
    }

    function getSelectedLabeledSources() {
      return getSelectedSources('labeled_sources');
    }

    function getSelectedValSources() {
      return getSelectedSources('val_sources');
    }

    function getSelectedCategories() {
      const box = document.getElementById('categoryChecks');
      return [...box.querySelectorAll('input[type=checkbox][data-category]:checked')].map(i => i.dataset.category);
    }

    function renderCategoryChecks(payload) {
      const box = document.getElementById('categoryChecks');
      const rows = (payload && payload.categories) || [];
      suggestedCategoryNames = (payload && payload.names) || [];
      if (!rows.length) {
        box.innerHTML = '<span class="muted">（Labeled / Val を選択してください）</span>';
        return;
      }
      const prev = new Set(getSelectedCategories());
      const defaultSel = new Set((payload.default_selected || suggestedCategoryNames));
      box.innerHTML = '';
      for (const row of rows) {
        const lab = document.createElement('label');
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.dataset.category = row.name;
        const tags = [];
        if (row.in_train) tags.push('labeled');
        if (row.in_val) tags.push('val');
        inp.checked = prev.size ? prev.has(row.name) : defaultSel.has(row.name);
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(' ' + row.name + (tags.length ? ' (' + tags.join(', ') + ')' : '')));
        box.appendChild(lab);
      }
    }

    async function refreshSuggestedCategories() {
      const labeled_sources = getSelectedLabeledSources();
      const val_sources = getSelectedValSources();
      if (!labeled_sources.length && !val_sources.length) {
        renderCategoryChecks({ categories: [], names: [], default_selected: [] });
        return;
      }
      try {
        const r = await fetch('/api/ppal-train/suggest-categories', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ labeled_sources, val_sources }),
        });
        const j = await r.json();
        if (!r.ok) {
          const detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
          document.getElementById('categoryChecks').innerHTML = '<span class="err">' + detail + '</span>';
          return;
        }
        renderCategoryChecks(j);
      } catch (e) {
        document.getElementById('categoryChecks').innerHTML = '<span class="err">カテゴリ取得に失敗しました</span>';
      }
    }

    function scheduleCategoryRefresh() {
      if (categoryFetchTimer) clearTimeout(categoryFetchTimer);
      categoryFetchTimer = setTimeout(refreshSuggestedCategories, 200);
    }

    function fillSourceSelect(el, sourceOptions) {
      el.innerHTML = '';
      if (!sourceOptions.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.disabled = true;
        opt.textContent = '(data/exports に COCO がありません)';
        el.appendChild(opt);
        return;
      }
      for (const o of sourceOptions) {
        const opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        el.appendChild(opt);
      }
    }

    function populateSourceSelects() {
      const opts = buildSourceOptions(datasets);
      fillSourceSelect(document.getElementById('labeled_sources'), opts);
      fillSourceSelect(document.getElementById('val_sources'), opts);
      scheduleCategoryRefresh();
    }

    async function loadDefaults() {
      try {
        const r = await fetch('/api/ppal-train/defaults');
        const j = await r.json();
        if (!r.ok) return;
        if (j.max_epochs != null) document.getElementById('max_epochs').value = j.max_epochs;
        if (j.lr != null) document.getElementById('lr').value = j.lr;
        if (j.batch_size != null) document.getElementById('batch_size').value = j.batch_size;
        if (j.config_path) document.getElementById('config_hint').textContent = j.config_path;
      } catch (e) {}
    }
    async function loadDefaultWorkDirName() {
      try {
        const r = await fetch('/api/ppal-train/default-work-dir');
        const j = await r.json();
        if (r.ok) {
          if (j.work_dir_prefix) document.getElementById('work_dir_prefix').textContent = j.work_dir_prefix;
          const example = j.work_dir_suffix_example || '';
          if (example) document.getElementById('work_dir_suffix').placeholder = example + '（空欄で自動）';
        }
      } catch (e) {}
    }

    async function loadPretrainedModels() {
      const sel = document.getElementById('pretrained');
      sel.innerHTML = '';
      const def = document.createElement('option');
      def.value = '';
      def.textContent = '（なし: ResNet 事前学習から開始）';
      sel.appendChild(def);
      const r = await fetch('/api/ppal-train/pretrained-models');
      const j = await r.json();
      for (const row of (j.pretrained_models || [])) {
        const opt = document.createElement('option');
        opt.value = row.id;
        const tag = row.ppal_ready ? 'PPAL OK' : 'checkpoint';
        opt.textContent = row.id + '  —  ' + row.checkpoint + ' (' + tag + ')';
        sel.appendChild(opt);
      }
    }

    async function loadDatasets() {
      const r = await fetch('/api/ppal-train/datasets');
      const j = await r.json();
      datasets = j.datasets || [];
      populateSourceSelects();
    }

    function chartBucket(key) {
      if (key === 'loss' || key.startsWith('loss_')) return 'loss';
      if (key.indexOf('bbox_mAP') !== -1 || key.indexOf('coco/bbox') === 0) return 'map';
      return 'sys';
    }

    function buildChartPayload(metricSeries, selectedKeys, bucket) {
      const steps = new Set();
      for (const k of selectedKeys) {
        if (chartBucket(k) !== bucket) continue;
        (metricSeries[k] || []).forEach(p => steps.add(p.step));
      }
      const labels = Array.from(steps).sort((a, b) => a - b);
      const palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
      const datasetsOut = [];
      let ci = 0;
      for (const k of selectedKeys) {
        if (chartBucket(k) !== bucket) continue;
        const pts = metricSeries[k] || [];
        const byStep = {};
        pts.forEach(p => { byStep[String(p.step)] = p.value; });
        const data = labels.map(s => (Object.prototype.hasOwnProperty.call(byStep, String(s)) ? byStep[String(s)] : null));
        datasetsOut.push({
          label: k,
          data: data,
          borderColor: palette[ci++ % palette.length],
          backgroundColor: 'transparent',
          tension: 0.12,
          pointRadius: 0,
          spanGaps: false
        });
      }
      return { labels, datasets: datasetsOut };
    }

    function chartOptions(yTitle, showLegend) {
      return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: showLegend, position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'step' } },
          y: { title: { display: true, text: yTitle } }
        }
      };
    }

    function renderOneChart(refKey, canvasId, metricSeries, selectedKeys, yTitle) {
      const el = document.getElementById(canvasId);
      if (!el) return;
      const { labels, datasets } = buildChartPayload(metricSeries, selectedKeys, refKey);
      const ctx = el.getContext('2d');
      const ch = chartRefs[refKey];
      if (!ch) {
        chartRefs[refKey] = new Chart(ctx, {
          type: 'line',
          data: { labels, datasets },
          options: chartOptions(yTitle, datasets.length > 0)
        });
        return;
      }
      ch.data.labels = labels;
      ch.data.datasets = datasets;
      ch.options.plugins.legend.display = datasets.length > 0;
      if (ch.options.scales && ch.options.scales.y && ch.options.scales.y.title) {
        ch.options.scales.y.title.text = yTitle;
      }
      ch.update('none');
    }

    function getSelectedMetrics() {
      const box = document.getElementById('metricChecks');
      if (!box) return [];
      return [...box.querySelectorAll('input[type=checkbox]:checked')].map(i => i.dataset.metric);
    }

    function destroyChart(refKey) {
      if (chartRefs[refKey]) {
        chartRefs[refKey].destroy();
        chartRefs[refKey] = null;
      }
    }

    function renderChartsFromState(j) {
      const wd = j.work_dir != null ? j.work_dir : null;
      if (wd !== prevWorkDirForCharts) {
        prevWorkDirForCharts = wd;
        ['loss', 'map', 'sys'].forEach(destroyChart);
      }
      const ms = j.metric_series || {};
      const sel = getSelectedMetrics();
      renderOneChart('loss', 'cLoss', ms, sel, 'loss 系');
      renderOneChart('map', 'cMap', ms, sel, 'mAP');
      renderOneChart('sys', 'cSys', ms, sel, 'memory / lr / time …');
    }

    function rebuildCheckboxes(available) {
      const box = document.getElementById('metricChecks');
      if (!available || !available.length) {
        box.innerHTML = '<span class="muted">（log.json 未生成）</span>';
        return;
      }
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch (e) {}
      const fromSaved = Array.isArray(saved) ? saved.filter(m => available.includes(m)) : [];
      let selected = new Set(fromSaved);
      if (selected.size === 0) {
        defaultMetrics.forEach(m => { if (available.includes(m)) selected.add(m); });
      }
      box.innerHTML = '';
      for (const m of available) {
        const safeId = 'mc_' + m.replace(/[^a-zA-Z0-9]+/g, '_');
        const lab = document.createElement('label');
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.dataset.metric = m;
        inp.id = safeId;
        inp.checked = selected.has(m);
        inp.addEventListener('change', () => {
          const all = [...box.querySelectorAll('input[type=checkbox]:checked')].map(i => i.dataset.metric);
          localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
          renderChartsFromState(lastState);
        });
        lab.appendChild(inp);
        lab.appendChild(document.createTextNode(' ' + m));
        box.appendChild(lab);
      }
    }

    async function poll() {
      const r = await fetch('/api/ppal-train/state');
      const j = await r.json();
      lastState = j;
      if (Array.isArray(j.default_metrics) && j.default_metrics.length) defaultMetrics = j.default_metrics;
      const st = j.status || 'idle';
      const statusEl = document.getElementById('status');
      statusEl.textContent = '状態: ' + st + (j.error ? ' — ' + j.error : '');
      statusEl.className = 'status ' + (st === 'failed' ? 'err' : (st === 'completed' ? 'ok' : ''));
      document.getElementById('workdir').textContent = j.work_dir ? ('work_dir: ' + j.work_dir) : '';
      document.getElementById('logjsonpath').textContent = j.log_json_path ? ('log.json: ' + j.log_json_path) : '';
      document.getElementById('log').textContent = (j.lines_tail || []).join('\\n');
      const logEl = document.getElementById('log');
      logEl.scrollTop = logEl.scrollHeight;

      const avail = j.available_metrics || [];
      const key = avail.join('|');
      if (key !== prevAvailKey) {
        prevAvailKey = key;
        rebuildCheckboxes(avail);
      }
      renderChartsFromState(j);
    }

    document.getElementById('labeled_sources').addEventListener('change', scheduleCategoryRefresh);
    document.getElementById('val_sources').addEventListener('change', scheduleCategoryRefresh);

    document.getElementById('btnStart').addEventListener('click', async () => {
      const labeled_sources = getSelectedLabeledSources();
      const val_sources = getSelectedValSources();
      if (!labeled_sources.length) { alert('Labeled データを 1 件以上選択してください'); return; }
      if (!val_sources.length) { alert('Val データを 1 件以上選択してください'); return; }
      const classes = getSelectedCategories();
      if (!classes.length) { alert('カテゴリを 1 つ以上選択してください'); return; }
      const body = {
        labeled_sources,
        val_sources,
        classes,
        max_epochs: parseInt(document.getElementById('max_epochs').value, 10),
        lr: parseFloat(document.getElementById('lr').value),
        batch_size: parseInt(document.getElementById('batch_size').value, 10),
      };
      const pw = document.getElementById('pretrained').value;
      if (pw) body.pretrained_work_dir = pw;
      const wdn = document.getElementById('work_dir_suffix').value.trim();
      if (wdn) body.work_dir_name = wdn;
      const r = await fetch('/api/ppal-train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok) {
        alert(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail));
        return;
      }
      prevAvailKey = '';
      poll();
    });

    document.getElementById('btnStop').addEventListener('click', async () => {
      await fetch('/api/ppal-train/stop', { method: 'POST' });
      poll();
    });

    Promise.all([loadDatasets(), loadPretrainedModels(), loadDefaults(), loadDefaultWorkDirName()]).then(() => {
      poll();
      setInterval(poll, 1200);
    });
  </script>
</body>
</html>
"""
)
