/* Utilitaires communs : thème des graphiques, tableaux, formats, requêtes. */
const NIDS = (() => {
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  // Palette catégorielle sombre validée (ordre fixe, jamais cyclée au-delà de 8)
  const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767'];
  const STATUS = { normal: '#0ca30c', attack: '#d03b3b', warning: '#fab219' };
  const SEQ = [[0, '#151a21'], [0.15, '#0d366b'], [0.4, '#184f95'], [0.7, '#256abf'], [1, '#3987e5']];
  const DIV = [[0, '#2a78d6'], [0.5, '#383835'], [1, '#e34948']];
  const CATS = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'];

  function layout(extra = {}) {
    const base = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'system-ui, -apple-system, Segoe UI, sans-serif', size: 12, color: '#c3c2b7' },
      margin: { l: 56, r: 18, t: 18, b: 48 },
      colorway: SERIES,
      xaxis: { gridcolor: '#2c2c2a', zerolinecolor: '#383835', linecolor: '#383835', tickcolor: '#383835', automargin: true },
      yaxis: { gridcolor: '#2c2c2a', zerolinecolor: '#383835', linecolor: '#383835', tickcolor: '#383835', automargin: true },
      legend: { orientation: 'h', y: 1.1, x: 0, font: { size: 12 }, bgcolor: 'rgba(0,0,0,0)' },
      hoverlabel: { bgcolor: '#1c232d', bordercolor: '#3a4554', font: { color: '#f0efec', size: 12 } },
      bargap: 0.25,
    };
    return deepMerge(base, extra);
  }
  function deepMerge(a, b) {
    const out = { ...a };
    for (const k in b) {
      if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') out[k] = deepMerge(a[k], b[k]);
      else out[k] = b[k];
    }
    return out;
  }
  const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };
  function plot(id, data, lay = {}) {
    const el = typeof id === 'string' ? document.getElementById(id) : id;
    if (!el) return;
    // étiquettes de valeur en bout de barre : jamais rognées par la zone de tracé
    data = data.map((t) => (t.type === 'bar' && t.textposition === 'outside' && t.cliponaxis === undefined) ? { ...t, cliponaxis: false } : t);
    Plotly.react(el, data, layout(lay), config);
  }

  // ------------------------------------------------------------ formats
  function fmt(x, d = 4) {
    if (x === null || x === undefined || x === '') return '—';
    if (typeof x !== 'number') return x;
    if (!isFinite(x)) return '∞';
    if (Number.isInteger(x) && Math.abs(x) < 1e15) return x.toLocaleString('fr-FR');
    const a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e7)) return x.toExponential(2);
    return x.toLocaleString('fr-FR', { maximumFractionDigits: d });
  }
  const pct = (x, d = 1) => (x === null || x === undefined) ? '—' : (100 * x).toLocaleString('fr-FR', { maximumFractionDigits: d }) + ' %';
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ------------------------------------------------------------ tableaux
  /* cols : [{key, label, fmt?: fn|'pct'|digits, num?: bool, html?: fn(row)}] */
  function table(el, rows, cols, opts = {}) {
    el = typeof el === 'string' ? document.getElementById(el) : el;
    if (!el) return;
    if (!rows || !rows.length) { el.innerHTML = '<div class="loading">Aucune donnée.</div>'; return; }
    cols = cols || Object.keys(rows[0]).map((k) => ({ key: k, label: k }));
    let sortKey = opts.sortKey || null, asc = opts.asc ?? false;
    const render = () => {
      let data = rows.slice();
      if (sortKey) {
        data.sort((a, b) => {
          const x = a[sortKey], y = b[sortKey];
          if (x === y) return 0;
          if (x === null || x === undefined) return 1;
          if (y === null || y === undefined) return -1;
          return (x > y ? 1 : -1) * (asc ? 1 : -1);
        });
      }
      if (opts.limit) data = data.slice(0, opts.limit);
      const head = cols.map((c) => `<th class="${c.num ? 'num' : ''}" data-k="${c.key}">${esc(c.label ?? c.key)}${sortKey === c.key ? (asc ? ' ▲' : ' ▼') : ''}</th>`).join('');
      const body = data.map((r) => {
        const cls = opts.rowClass ? opts.rowClass(r) : '';
        const attrs = opts.rowAttr ? opts.rowAttr(r) : '';
        return `<tr class="${cls}" ${attrs}>` + cols.map((c) => {
          let val = r[c.key];
          let out;
          if (c.html) out = c.html(r);
          else if (c.fmt === 'pct') out = pct(val);
          else if (typeof c.fmt === 'function') out = c.fmt(val, r);
          else if (typeof val === 'number') out = fmt(val, c.fmt ?? 4);
          else out = esc(val ?? '—');
          return `<td class="${c.num || typeof val === 'number' ? 'num' : ''}">${out}</td>`;
        }).join('') + '</tr>';
      }).join('');
      el.innerHTML = `<div class="table-wrap" style="${opts.maxHeight ? 'max-height:' + opts.maxHeight : ''}"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
      el.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
        const k = th.dataset.k;
        if (sortKey === k) asc = !asc; else { sortKey = k; asc = false; }
        render();
      }));
      if (opts.onRow) el.querySelectorAll('tbody tr').forEach((tr, i) => tr.addEventListener('click', () => opts.onRow(data[i])));
    };
    render();
  }

  async function get(url) {
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  async function post(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  function seg(el, onChange) {
    el = typeof el === 'string' ? document.getElementById(el) : el;
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      el.querySelectorAll('button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      onChange(b.dataset.v);
    }));
    return () => el.querySelector('button.on')?.dataset.v;
  }
  function showError(el, err) {
    el = typeof el === 'string' ? document.getElementById(el) : el;
    if (el) el.innerHTML = `<div class="err">⚠ ${esc(err.message || err)}</div>`;
  }
  function confusionHeatmap(id, matrix, labels, title = '') {
    const z = matrix.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((x) => x / s); });
    const text = matrix.map((row, i) => row.map((x, j) => `${x}<br>${(100 * z[i][j]).toFixed(1)} %`));
    plot(id, [{
      type: 'heatmap', z, x: labels, y: labels, text, texttemplate: '%{text}', textfont: { size: labels.length > 4 ? 10 : 13 },
      colorscale: SEQ, zmin: 0, zmax: 1, showscale: false, xgap: 2, ygap: 2,
      hovertemplate: 'Réel %{y} → prédit %{x}<br>%{text}<extra></extra>',
    }], { xaxis: { title: 'Classe prédite', side: 'bottom', gridcolor: 'rgba(0,0,0,0)' }, yaxis: { title: 'Classe réelle', autorange: 'reversed', gridcolor: 'rgba(0,0,0,0)' }, margin: { l: 110, b: 90, t: title ? 30 : 10 }, title: title ? { text: title, font: { size: 13 } } : undefined });
  }
  return { SERIES, STATUS, SEQ, DIV, CATS, layout, plot, fmt, pct, esc, table, get, post, seg, showError, confusionHeatmap };
})();
