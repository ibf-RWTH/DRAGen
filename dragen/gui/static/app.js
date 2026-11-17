'use strict';

/* DRAGen Studio: single-page front end for dragen.gui.server. No dependencies. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c);
  return node;
};
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : (typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(d)) : String(v));
const pct = v => (v === null || v === undefined) ? '–' : (100 * v).toFixed(1) + ' %';
const dur = s => {
  if (s === null || s === undefined) return '';
  if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + ' s';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m < 60 ? `${m} min ${r} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
};
const ago = t => {
  const s = Date.now() / 1000 - t;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};

async function api(path, options = {}) {
  const init = { method: options.method || 'GET', headers: {} };
  if (options.body !== undefined) {
    init.body = JSON.stringify(options.body);
    init.headers['Content-Type'] = 'application/json';
  }
  const res = await fetch('/api/' + path, init);
  const data = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(message, bad = false) {
  const t = el('div', { class: 'toast' + (bad ? ' bad' : '') }, message);
  $('#toasts').append(t);
  setTimeout(() => t.remove(), bad ? 6000 : 3500);
}

const PHASE_COLORS = {
  1: '#5b8def', 2: '#e0445b', 3: '#f0a030', 4: '#9b6dfc', 5: '#22c3a6', 6: '#3b3f4f', 7: '#ff7aa8',
};
const PHASE_NAMES = { 1: 'ferrite', 2: 'martensite', 3: 'pearlite', 4: 'bainite', 5: 'austenite', 6: 'inclusions', 7: 'bands' };

const state = {
  info: null, inputs: [], config: null, jobs: [], selected: null, view: 'new',
  filter: 'all', search: '', detail: null, logLines: [], tab: 'overview',
  slice: { axis: 2, index: null, color: 'grain', bounds: true, data: null },
  bandsStash: 1,
};

/* theme --------------------------------------------------------------------------------- */
function initTheme() {
  const saved = localStorage.getItem('dragen-theme');
  const dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('dragen-theme', next);
    if (state.slice.data) drawSlice();
    View3D.requestRender();
    if (state.view === 'run' && state.tab === 'sizes') loadHistogram();
  });
}

/* configuration form ------------------------------------------------------------------- */
const getPath = (obj, path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
function setPath(obj, path, value) {
  const keys = path.split('.');
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k];
  o[keys[keys.length - 1]] = value;
}

function readInput(input) {
  if (input.type === 'checkbox') return input.checked;
  if (input.dataset.type === 'int') return parseInt(input.value, 10);
  if (input.type === 'number') {
    if (input.value === '') return null;
    const v = Number(input.value);
    return input.step === '1' ? Math.round(v) : v;
  }
  return input.value === '' ? null : input.value;
}

function writeForm() {
  for (const input of $$('[data-bind]')) {
    const v = getPath(state.config, input.dataset.bind);
    if (input.type === 'checkbox') input.checked = !!v;
    else input.value = v === null || v === undefined ? '' : v;
  }
  $('#bands-on').checked = state.config.bands.count > 0;
  $('#incl-on').checked = state.config.inclusions.fraction > 0;
  syncFeatureCards();
  renderPhases();
}

function syncFeatureCards() {
  $('#bands-body').classList.toggle('off', !$('#bands-on').checked);
  $('#incl-body').classList.toggle('off', !$('#incl-on').checked);
}

function renderPhases() {
  const host = $('#phase-rows');
  host.innerHTML = '';
  state.config.phases.forEach((p, i) => {
    const select = el('select', {}, ...state.info.phases.map(name => el('option', { value: name, selected: name === p.phase }, name)));
    select.addEventListener('change', () => { p.phase = select.value; onChange(); });
    const source = el('input', { type: 'text', list: 'input-list', value: p.source || '', placeholder: 'path to .csv or GAN .pkl' });
    source.addEventListener('input', () => {
      p.source = source.value || null;
      const hit = state.inputs.find(f => f.path === source.value);
      if (hit && hit.phase && state.info.phases.includes(hit.phase)) { p.phase = hit.phase; select.value = hit.phase; }
      onChange();
    });
    const frac = el('input', { type: 'number', step: 'any', min: '0', max: '1', value: p.fraction });
    frac.addEventListener('input', () => { p.fraction = frac.value === '' ? 0 : Number(frac.value); onChange(); });
    const remove = el('button', { class: 'remove', title: 'Remove phase', disabled: state.config.phases.length === 1 }, '×');
    remove.addEventListener('click', () => { state.config.phases.splice(i, 1); renderPhases(); onChange(); });
    host.append(el('div', { class: 'phase-row' },
      el('label', {}, 'Phase', select), el('label', {}, 'Grain source', source),
      el('label', {}, 'Fraction', frac), remove));
  });
  updateFractionSum();
}

function updateFractionSum() {
  const sum = state.config.phases.reduce((s, p) => s + (Number(p.fraction) || 0), 0);
  const tag = $('#fraction-sum');
  tag.textContent = `Σ = ${sum.toFixed(3)}`;
  tag.className = 'fraction-sum ' + (Math.abs(sum - 1) < 1e-6 ? 'ok' : 'bad');
}

function payload() {
  const c = JSON.parse(JSON.stringify(state.config));
  if (!$('#bands-on').checked) c.bands.count = 0;
  if (!$('#incl-on').checked) c.inclusions.fraction = 0;
  for (const p of c.phases) if (p.source === '') p.source = null;
  if (!c.bands.source) c.bands.source = null;
  if (!c.inclusions.source) c.inclusions.source = null;
  return c;
}

let validateTimer = null;
function onChange() {
  updateFractionSum();
  clearTimeout(validateTimer);
  validateTimer = setTimeout(validate, 250);
}

async function validate() {
  const box = $('#validation');
  try {
    const res = await api('validate', { method: 'POST', body: { config: payload() } });
    const lines = [];
    if (res.ok) lines.push(el('div', { class: 'ok' }, `✓ Ready — ${res.voxels.toLocaleString()} voxels (${res.shape.join(' × ')})`));
    for (const e of res.errors) lines.push(el('div', { class: 'err' }, '✕ ' + e));
    for (const w of res.warnings || []) lines.push(el('div', { class: 'warn' }, '! ' + w));
    box.replaceChildren(...lines);
    $('#start-run').disabled = !res.ok;
    if (res.shape) {
      const sp = res.spacing.map(s => s.toFixed(3)).join(' / ');
      $('#box-stats').textContent = `${res.shape.join(' × ')} voxels · ${res.voxels.toLocaleString()} total · voxel edge ${sp} µm`;
    }
    return res;
  } catch (err) {
    box.replaceChildren(el('div', { class: 'err' }, '✕ ' + err.message));
    $('#start-run').disabled = true;
  }
}

function initForm() {
  for (const input of $$('[data-bind]')) {
    const ev = input.tagName === 'SELECT' || input.type === 'checkbox' ? 'change' : 'input';
    input.addEventListener(ev, () => { setPath(state.config, input.dataset.bind, readInput(input)); onChange(); });
  }
  $('#bands-on').addEventListener('change', e => {
    if (e.target.checked) state.config.bands.count = Math.max(1, state.bandsStash || 1);
    else state.bandsStash = state.config.bands.count || 1;
    writeForm(); onChange();
  });
  $('#incl-on').addEventListener('change', e => {
    if (e.target.checked && !(state.config.inclusions.fraction > 0)) state.config.inclusions.fraction = 0.01;
    writeForm(); onChange();
  });
  $('#add-phase').addEventListener('click', () => {
    const used = new Set(state.config.phases.map(p => p.phase));
    const free = state.info.phases.find(p => !used.has(p)) || state.info.phases[0];
    state.config.phases.push({ phase: free, source: null, fraction: 0 });
    renderPhases(); onChange();
  });
  $('#band-phase').replaceChildren(...state.info.band_phases.map(p => el('option', { value: p }, p)));
  $('#unsupported').replaceChildren(...state.info.not_supported.map(f =>
    el('li', {}, el('b', {}, f.feature), el('span', { class: 'tag' }, 'planned ' + f.planned), el('div', {}, f.note))));
  $('#input-list').replaceChildren(...state.inputs.map(f => el('option', { value: f.path }, `${f.kind} · ${f.name}`)));
  $('#start-run').addEventListener('click', startRun);
  $('#preset-save').addEventListener('click', savePreset);
  $('#preset-delete').addEventListener('click', deletePreset);
  $('#preset-select').addEventListener('change', loadPreset);
}

async function startRun() {
  const button = $('#start-run');
  button.disabled = true;
  try {
    const res = await api('jobs', { method: 'POST', body: { name: $('#run-name').value, config: payload() } });
    toast(res.jobs.length === 1 ? `Started “${res.jobs[0].name}”` : `Queued ${res.jobs.length} RVEs`);
    await refreshJobs();
    selectJob(res.jobs[0].id);
  } catch (err) {
    toast(err.message, true);
  } finally {
    button.disabled = false;
  }
}

async function refreshPresets(select) {
  const res = await api('presets');
  $('#preset-select').replaceChildren(el('option', { value: '' }, 'Load preset…'),
    ...res.presets.map(n => el('option', { value: n, selected: n === select }, n)));
}

async function savePreset() {
  const current = $('#preset-select').value;
  const name = prompt('Preset name', current || $('#run-name').value || 'my-preset');
  if (!name) return;
  try {
    const res = await api('presets', { method: 'POST', body: { name, config: payload() } });
    await refreshPresets(res.name);
    toast(`Preset “${res.name}” saved`);
  } catch (err) { toast(err.message, true); }
}

async function loadPreset() {
  const name = $('#preset-select').value;
  if (!name) return;
  try {
    const res = await api('presets/' + encodeURIComponent(name));
    state.config = mergeDefaults(res.config);
    writeForm(); validate();
    toast(`Preset “${name}” loaded`);
  } catch (err) { toast(err.message, true); }
}

async function deletePreset() {
  const name = $('#preset-select').value;
  if (!name || !confirm(`Delete preset “${name}”?`)) return;
  try {
    await api('presets/' + encodeURIComponent(name), { method: 'DELETE' });
    await refreshPresets();
    toast(`Preset “${name}” deleted`);
  } catch (err) { toast(err.message, true); }
}

function mergeDefaults(cfg) {
  const d = JSON.parse(JSON.stringify(state.info.defaults));
  const out = { ...d, ...cfg };
  for (const k of ['box', 'rsa', 'tessellation', 'bands', 'inclusions']) out[k] = { ...d[k], ...(cfg[k] || {}) };
  if (!out.box.periodic) out.box.periodic = out.box.size.map(() => true);
  return out;
}

/* job list ------------------------------------------------------------------------------ */
async function refreshJobs() {
  try {
    const res = await api('jobs');
    state.jobs = res.jobs;
    renderJobs();
    renderQueue();
  } catch (err) { /* server gone: keep the last state */ }
}

function renderQueue() {
  const running = state.jobs.filter(j => j.status === 'running').length;
  const queued = state.jobs.filter(j => j.status === 'queued').length;
  const host = $('#queue-summary');
  host.replaceChildren();
  if (running) host.append(el('span', { class: 'status running' }, `${running} running`));
  if (queued) host.append(el('span', { class: 'status queued' }, `${queued} queued`));
}

function renderJobs() {
  const list = $('#run-list');
  const term = state.search.toLowerCase();
  const visible = state.jobs.filter(j => {
    if (state.filter === 'running' && !['running', 'queued'].includes(j.status)) return false;
    if (state.filter === 'done' && j.status !== 'done') return false;
    if (state.filter === 'failed' && !['failed', 'cancelled', 'interrupted'].includes(j.status)) return false;
    return !term || j.name.toLowerCase().includes(term) || j.id.toLowerCase().includes(term);
  });
  // meshing jobs right below their run
  const byParent = {};
  for (const j of visible) if (j.parent) (byParent[j.parent] ||= []).push(j);
  const ordered = [];
  for (const j of visible) {
    if (j.parent && visible.some(p => p.id === j.parent)) continue;
    ordered.push(j);
    for (const c of byParent[j.id] || []) ordered.push(c);
  }
  if (!ordered.length) {
    list.replaceChildren(el('li', { class: 'empty' }, state.jobs.length ? 'No run matches the filter.' : 'No runs yet. Configure one and press “Start generation”.'));
    return;
  }
  list.replaceChildren(...ordered.map(j => {
    const item = el('li', { class: 'run-item' + (j.id === state.selected ? ' selected' : '') + (j.parent ? ' child' : '') },
      el('span', { class: 'name', title: j.name }, ({ mesh: '⬡ ', export: '⇩ ' }[j.kind] || '') + j.name),
      el('span', { class: 'status ' + j.status }, j.status),
      el('span', { class: 'meta' },
        el('span', {}, j.imported ? 'imported' : ago(j.created)),
        j.elapsed !== null ? el('span', {}, '· ' + dur(j.elapsed)) : null));
    item.addEventListener('click', () => selectJob(j.id));
    return item;
  }));
}

/* run detail ---------------------------------------------------------------------------- */
function showView(view) {
  state.view = view;
  $('#view-new').classList.toggle('hidden', view !== 'new');
  $('#view-run').classList.toggle('hidden', view !== 'run');
  if (view === 'new') { state.selected = null; renderJobs(); }
}

async function selectJob(id) {
  const changed = state.selected !== id;
  state.selected = id;
  if (changed) {
    state.logLines = [];
    state.slice.data = null;
    state.slice.index = null;
    state.detail = null;
    $('#log').textContent = '';
    View3D.forget();
  }
  showView('run');
  renderJobs();
  await loadDetail();
  if (changed && state.detail) {
    const { kind, status } = state.detail.job;
    setTab({ mesh: 'mesh', export: 'export' }[kind] || (status === 'done' ? 'overview' : 'log'));
  }
}

async function loadDetail() {
  if (!state.selected) return;
  try {
    const d = await api('jobs/' + encodeURIComponent(state.selected));
    const before = state.detail?.job.status;
    state.detail = d;
    renderDetail();
    if (before && before !== d.job.status && d.job.status === 'done' && d.job.kind === 'generate') {
      setTab('overview');
    } else if (state.tab === '3d' && d.job.status === 'done') {
      View3D.open(state.selected, d);        // e.g. a mesh finished: offer its surface
    }
    await loadLog();
  } catch (err) {
    toast(err.message, true);
    showView('new');
  }
}

function renderDetail() {
  const { job, result } = state.detail;
  $('#run-kind').textContent = { mesh: 'Mesh (experimental)', export: 'Export' }[job.kind] || (job.imported ? 'Imported run' : 'Generation run');
  $('#run-title').textContent = job.name;
  const sub = [`<span class="status ${job.status}">${job.status}</span>`, `<span class="mono">${esc(job.id)}</span>`];
  if (job.elapsed !== null) sub.push(dur(job.elapsed));
  if (result) sub.push(`${result.shape.join(' × ')} voxels`, `${result.n_grains.toLocaleString()} grains`);
  if (job.error) sub.push(`<span style="color:var(--bad)">${esc(job.error)}</span>`);
  $('#run-sub').innerHTML = sub.join(' · ');

  const actions = $('#run-actions');
  actions.replaceChildren();
  if (['queued', 'running'].includes(job.status)) {
    actions.append(el('button', { class: 'btn danger', onclick: () => jobAction('cancel') }, 'Cancel'));
  }
  if (job.kind === 'generate' && job.status === 'done' && !job.imported) {
    const disabled = !!state.info.mesh_unavailable;
    actions.append(el('button', { class: 'btn', disabled, title: disabled ? state.info.mesh_unavailable : 'C3D10 mesh with smoothed, remeshed grain boundaries (experimental)', onclick: () => jobAction('mesh') }, '⬡ Mesh'));
  }
  if (job.kind === 'generate' && job.status === 'done') {
    actions.append(el('button', { class: 'btn', title: 'Solver files: DAMASK, Abaqus, Exodus (MOOSE), VTU, MSH, grain table', onclick: () => setTab('export') }, '⇩ Export'));
  }
  if (job.kind === 'generate' && state.detail.config) {
    actions.append(el('button', { class: 'btn ghost', onclick: () => reuseConfig() }, 'Reuse config'));
  }
  const files = { mesh: ['mesh.json', 'mesh.npz', 'log.txt'], export: ['export.zip', 'export.json', 'log.txt'] }[job.kind]
    || ['config.yaml', 'result.json', 'grains.csv', 'labels.npz'];
  if (job.status === 'done') {
    const menu = el('select', { title: 'Download' }, el('option', { value: '' }, 'Download…'), ...files.map(f => el('option', { value: f }, f)));
    menu.addEventListener('change', () => {
      if (menu.value) window.location = `/api/jobs/${encodeURIComponent(job.id)}/files/${menu.value}`;
      menu.value = '';
    });
    actions.append(menu);
  }
  if (!['queued', 'running'].includes(job.status) && !job.imported) {
    actions.append(el('button', { class: 'btn ghost danger', onclick: () => jobAction('delete') }, 'Delete'));
  }

  $$('#run-tabs button').forEach(b => {
    const t = b.dataset.tab;
    const isMesh = job.kind === 'mesh', isExport = job.kind === 'export';
    const needsResult = ['overview', '3d', 'slices', 'sizes', 'export'].includes(t);
    b.classList.toggle('hidden', (isMesh && (needsResult || t === 'config')) ||
      (isExport && !['export', 'log'].includes(t)));
    b.disabled = job.kind === 'generate' && needsResult && job.status !== 'done';
  });
  $('#config-view').textContent = state.detail.config ? toYaml(state.detail.config) : '';
  renderOverview();
  renderMesh();
  renderExport();
}

function reuseConfig() {
  state.config = mergeDefaults(state.detail.config);
  writeForm(); validate();
  showView('new');
  toast('Configuration copied into the form');
}

async function jobAction(action) {
  const job = state.detail.job;
  try {
    if (action === 'cancel') await api(`jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' });
    if (action === 'mesh') {
      const res = await api(`jobs/${encodeURIComponent(job.id)}/mesh`, { method: 'POST' });
      toast('Meshing queued');
      await refreshJobs();
      return selectJob(res.job.id);
    }
    if (action === 'export') {
      const res = await api(`jobs/${encodeURIComponent(job.id)}/export`, { method: 'POST', body: readExportForm() });
      toast('Export queued');
      await refreshJobs();
      return selectJob(res.job.id);
    }
    if (action === 'delete') {
      if (!confirm(`Delete “${job.name}” and its files?`)) return;
      await api(`jobs/${encodeURIComponent(job.id)}`, { method: 'DELETE' });
      toast('Run deleted');
      showView('new');
      return refreshJobs();
    }
    await refreshJobs();
    await loadDetail();
  } catch (err) { toast(err.message, true); }
}

function setTab(tab) {
  state.tab = tab;
  $$('#run-tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $$('#view-run .tab').forEach(t => t.classList.toggle('hidden', t.dataset.tab !== tab));
  if (tab === 'slices') loadSlice();
  if (tab === '3d' && state.detail) View3D.open(state.selected, state.detail);
  if (tab === 'sizes') loadHistogram();
  if (tab === 'log') { const log = $('#log'); log.scrollTop = log.scrollHeight; }
}

function kpi(k, v, s = '', cls = '') {
  return el('div', { class: 'kpi ' + cls }, el('div', { class: 'k' }, k), el('div', { class: 'v' }, v), s ? el('div', { class: 's' }, s) : null);
}

function renderOverview() {
  const host = $('#overview');
  const { job, result } = state.detail;
  if (job.kind !== 'generate') { host.replaceChildren(); return; }
  if (!result) {
    host.replaceChildren(el('div', { class: 'card' }, job.status === 'running'
      ? el('p', { class: 'muted' }, el('span', { class: 'spinner' }), '  Generating… see the log for progress.')
      : el('p', { class: 'muted' }, job.status === 'queued' ? 'Waiting for a free worker.' : 'No result.')));
    return;
  }
  const r = result.report;
  const viol = result.acceptance_violations || [];
  const verdict = viol.length
    ? el('div', { class: 'verdict bad' }, el('div', {}, el('b', {}, `${viol.length} acceptance criteria violated`), el('ul', {}, ...viol.map(v => el('li', {}, v)))))
    : el('div', { class: 'verdict ok' }, el('div', {}, el('b', {}, 'Accepted'), 'All acceptance criteria are met (dragen.analysis.acceptance).'));
  const kpis = el('div', { class: 'kpis' },
    kpi('Grains', fmt(r.n_grains), `${fmt(r.n_voxels)} voxels`),
    kpi('Fill ratio', pct(r.fill_ratio), '', r.fill_ratio >= 1 ? 'good' : 'bad'),
    kpi('Split grains', fmt(r.fragmented_grains), 'several bodies', r.fragmented_grains ? 'bad' : 'good'),
    kpi('Volume ratio', fmt(r.volume_ratio_median, 3), `p10 ${fmt(r.volume_ratio_p10, 2)} · p90 ${fmt(r.volume_ratio_p90, 2)}`),
    kpi('Main body p10', fmt(r.main_body_ratio_p10, 3), 'limit ≥ 0.60', r.main_body_ratio_p10 >= 0.6 ? 'good' : 'bad'),
    kpi('Volume KS', fmt(r.volume_ks, 3), 'limit ≤ 0.12', r.volume_ks <= 0.12 ? 'good' : 'bad'),
    kpi('Generation', dur(result.seconds.total), `RSA ${dur(result.seconds.rsa)} · tess. ${dur(result.seconds.tessellation)}`),
  );
  // phase fractions, target vs actual
  const bars = el('div', { class: 'bars' });
  const targets = result.target_phase_ratio || {};
  const phases = new Set([...Object.keys(targets), ...Object.keys(r.phase_fractions || {})]);
  for (const p of [...phases].sort()) {
    const actual = r.phase_fractions[p] || 0, target = targets[p] || 0;
    const color = PHASE_COLORS[p] || '#888';
    bars.append(el('div', { class: 'bar-row' },
      el('span', {}, PHASE_NAMES[p] || 'phase ' + p),
      el('div', { class: 'bar' }, el('i', { style: `width:${100 * actual}%;background:${color}` }), el('span', { class: 'target', style: `left:${100 * target}%` })),
      el('span', { class: 'mono' }, `${pct(actual)} / ${pct(target)}`)));
  }
  const t = result.tessellation || {};
  const extra = el('table', { class: 'kv' },
    ...[['Epochs', t.epochs], ['Filled after RSA', pct(t.fraction_after_rsa)], ['From regular growth', pct(t.fraction_from_regular_growth)],
      ['From reset phase', pct(t.fraction_from_reset_phase)], ['From dilation', pct(t.fraction_from_dilation)],
      ['RSA attempts', result.rsa_attempts], ['Repaired voxels', result.repaired_voxels]]
      .map(([k, v]) => el('tr', {}, el('td', {}, k), el('td', {}, fmt(v)))));
  const cards = [el('div', { class: 'card' }, el('h2', {}, 'Phase fractions'), bars, el('p', { class: 'hint' }, 'Bar: RVE · line: target')),
    el('div', { class: 'card' }, el('h2', {}, 'Tessellation'), extra)];
  if ((result.bands || []).length) {
    cards.push(el('div', { class: 'card' }, el('h2', {}, 'Bands'), el('table', { class: 'kv' },
      ...result.bands.map(b => el('tr', {}, el('td', {}, `Band ${b.band + 1} · ${b.grains} grains · ${fmt(b.thickness, 2)} µm`),
        el('td', {}, `filling ${pct(b.filling)} · outside ${pct(b.outside_fraction)}`))))));
  }
  const inc = result.inclusions;
  if (inc && inc.inclusions) {
    cards.push(el('div', { class: 'card' }, el('h2', {}, 'Inclusions'), el('table', { class: 'kv' },
      ...[['Placed', `${inc.placed} of ${inc.inclusions}`], ['Volume fraction', pct(inc.volume_fraction)],
        ['Expected (placed)', pct(inc.expected_placed_fraction)], ['Touching a boundary', inc.touching_boundary]]
        .map(([k, v]) => el('tr', {}, el('td', {}, k), el('td', {}, fmt(v)))))));
  }
  host.replaceChildren(verdict, kpis, el('div', { class: 'grid' }, ...cards));
}

const mb = n => n < 1024 * 1024 ? (n / 1024).toFixed(1) + ' kB' : (n / 1024 / 1024).toFixed(1) + ' MB';

function renderExport() {
  const host = $('#export-panel');
  const { job, export: exp, exports, meshes } = state.detail;
  if (job.kind === 'export') {
    if (!exp) {
      host.replaceChildren(el('div', { class: 'card' }, el('p', { class: 'muted' },
        job.status === 'running' ? [el('span', { class: 'spinner' }), '  Exporting… see the log.'] : (job.status === 'queued' ? 'Waiting for a free worker.' : 'No export result.'))));
      return;
    }
    const o = exp.options || {};
    host.replaceChildren(
      el('div', { class: 'kpis' },
        kpi('Files', fmt(exp.files.length)), kpi('Mesh', exp.mesh, exp.cell_type),
        kpi('Archive', mb(exp.zip_bytes), 'export.zip'),
        exp.substructure ? kpi('Substructure', fmt(exp.substructure.blocks) + ' blocks', `${fmt(exp.substructure.packets)} packets`) : null,
        kpi('Time', dur(Object.values(exp.seconds || {}).reduce((a, b) => a + b, 0)))),
      el('div', { class: 'card' }, el('h2', {}, 'Files'),
        el('table', { class: 'kv' }, ...exp.files.map(f => el('tr', {}, el('td', { class: 'mono' }, f.name),
          el('td', {}, f.format), el('td', {}, mb(f.bytes))))),
        el('p', { class: 'hint' }, `Abaqus: lengths in ${o.length_unit || 'm'}, materials per ${o.material_per || 'grain'}` +
          (o.element_type ? `, element type ${o.element_type}` : '') +
          '. The .inp holds mesh, sets, sections and periodic equations; materials and steps are yours (*Include).'),
        el('a', { class: 'btn primary', href: `/api/jobs/${encodeURIComponent(job.id)}/files/export.zip` }, '⇩ Download export.zip')));
    return;
  }
  if (job.kind !== 'generate' || job.status !== 'done') { host.replaceChildren(); return; }
  const formats = state.info.export_formats || {};
  const unavailable = state.info.export_unavailable || {};
  const checks = Object.entries(formats).map(([key, label]) => el('label', { class: 'check', title: unavailable[key] || '' },
    el('input', { type: 'checkbox', name: 'export-format', value: key, checked: !unavailable[key] && key !== 'gmsh',
      disabled: !!unavailable[key] }), el('span', {}, label + (unavailable[key] ? ' – not installed' : ''))));
  const doneMeshes = (meshes || []).filter(m => m.status === 'done');
  const meshSel = el('select', { id: 'export-mesh' }, el('option', { value: 'voxel' }, 'Voxels (one hexahedron per voxel)'),
    ...doneMeshes.map(m => el('option', { value: m.id }, `Tetrahedra: ${m.name}`)));
  if (doneMeshes.length) meshSel.value = doneMeshes[0].id;
  const rows = (exports || []).map(x => el('tr', {}, el('td', {}, el('a', { href: '#', onclick: e => { e.preventDefault(); selectJob(x.id); } }, x.name)),
    el('td', {}, el('span', { class: 'status ' + x.status }, x.status)),
    el('td', {}, x.status === 'done' ? el('a', { href: `/api/jobs/${encodeURIComponent(x.id)}/files/export.zip` }, 'export.zip') : '')));
  host.replaceChildren(el('div', { class: 'grid' },
    el('div', { class: 'card' }, el('h2', {}, 'Export this RVE'),
      el('div', { class: 'field' }, el('span', { class: 'label' }, 'Formats'), el('div', { class: 'checks' }, ...checks)),
      el('div', { class: 'row two' },
        el('label', {}, 'Mesh (Abaqus, Exodus, VTU, MSH)', meshSel),
        el('label', {}, 'Abaqus length unit', el('select', { id: 'export-unit' },
          ...[['m', 'm (as the library)'], ['mm', 'mm'], ['um', 'µm (legacy files)']].map(([v, t]) => el('option', { value: v }, t))))),
      el('div', { class: 'row two' },
        el('label', {}, 'Abaqus materials', el('select', { id: 'export-per' },
          el('option', { value: 'grain' }, 'one per grain (Ferrite_12, …)'), el('option', { value: 'phase' }, 'one per phase (Ferrite, …)'))),
        el('label', {}, 'Abaqus element type', el('input', { type: 'text', id: 'export-element', placeholder: 'default: C3D8 / C3D10' }))),
      el('div', { class: 'row two' },
        el('label', { class: 'check inline' }, el('input', { type: 'checkbox', id: 'export-subs' }),
          el('span', {}, 'Martensite substructure (packets, blocks; martensite, pearlite, bainite)')),
        el('label', {}, 'Block thickness [µm]', el('input', { type: 'number', id: 'export-block', min: '0', step: 'any', value: '1' }))),
      el('p', { class: 'hint' }, doneMeshes.length ? 'DAMASK always uses the voxels.' :
        'Without a finished mesh the voxels are exported; “⬡ Mesh” builds a smoothed tetrahedral mesh first. DAMASK always uses the voxels.'),
      el('button', { class: 'btn primary', onclick: () => jobAction('export') }, '⇩ Start export')),
    el('div', { class: 'card' }, el('h2', {}, 'Exports of this run'),
      rows.length ? el('table', { class: 'kv' }, ...rows) : el('p', { class: 'muted' }, 'No export yet.'))));
}

function readExportForm() {
  return {
    formats: $$('input[name=export-format]:checked').map(i => i.value),
    mesh: $('#export-mesh').value,
    length_unit: $('#export-unit').value,
    material_per: $('#export-per').value,
    element_type: $('#export-element').value.trim() || null,
    substructure: $('#export-subs').checked ? { block_thickness: Number($('#export-block').value) } : null,
  };
}

function renderMesh() {
  const host = $('#mesh-panel');
  const { job, mesh, meshes } = state.detail;
  if (job.kind === 'generate') {
    const rows = (meshes || []).map(m => el('tr', {}, el('td', {}, el('a', { href: '#', onclick: e => { e.preventDefault(); selectJob(m.id); } }, m.name)),
      el('td', {}, el('span', { class: 'status ' + m.status }, m.status))));
    host.replaceChildren(el('div', { class: 'card' }, el('h2', {}, 'Meshes of this run'),
      rows.length ? el('table', { class: 'kv' }, ...rows) : el('p', { class: 'muted' }, state.info.mesh_unavailable
        ? 'Meshing is not available: ' + state.info.mesh_unavailable
        : 'No mesh yet. “⬡ Mesh” builds a C3D10 mesh with smoothed and adaptively remeshed grain boundaries (experimental, can take minutes).')));
    return;
  }
  if (!mesh) {
    host.replaceChildren(el('div', { class: 'card' }, el('p', { class: 'muted' },
      job.status === 'running' ? [el('span', { class: 'spinner' }), '  Meshing… see the log.'] : (job.status === 'queued' ? 'Waiting for a free worker.' : 'No mesh result.'))));
    return;
  }
  const q = mesh.quality || {};
  const dih = q.dihedral_min_deg || {};
  const fb = mesh.feedback || [];
  const errors = ['inverted_elements', 'open_inner_faces', 'overused_faces', 'midside_mismatch', 'interface_faces_missing',
    'interface_faces_extra', 'periodic_unmatched_nodes', 'periodic_unmatched_facets'];
  const bad = errors.filter(k => q[k]);
  host.replaceChildren(
    el('div', { class: 'verdict ' + (bad.length ? 'bad' : 'ok') }, el('div', {}, el('b', {}, bad.length ? 'Mesh checks failed' : 'All mesh checks passed'),
      bad.length ? el('ul', {}, ...bad.map(k => el('li', {}, `${k}: ${q[k]}`))) : 'Conforming, no inverted elements, exact periodic node pairs.')),
    el('div', { class: 'kpis' },
      kpi('Elements', fmt(mesh.elements), mesh.cell_type),
      kpi('Nodes', fmt(mesh.nodes)),
      kpi('Surface triangles', fmt(mesh.surface_triangles), `staircase ${fmt(mesh.staircase_triangles)}`),
      kpi('Min dihedral', fmt(dih.min, 2) + '°', `p1 ${fmt(dih.p1, 1)}°`, dih.min >= 5 ? 'good' : 'bad'),
      kpi('Volume error', pct(q.volume_deviation?.max_abs), 'max per grain vs voxels'),
      kpi('Time', dur(mesh.seconds?.total))),
    el('div', { class: 'card' }, el('h2', {}, 'Feedback rounds'), el('table', { class: 'kv' },
      ...fb.map((f, i) => el('tr', {}, el('td', {}, `Round ${i}: ${fmt(f.triangles)} triangles, ${fmt(f.elements)} elements`),
        el('td', {}, `${f.below_min_dihedral} below 5° · min ${fmt(f.min_dihedral, 2)}°`))))));
}

async function loadLog() {
  if (!state.selected) return;
  try {
    const res = await api(`jobs/${encodeURIComponent(state.selected)}/log?offset=${state.logLines.length}`);
    if (res.lines.length) {
      const log = $('#log');
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
      state.logLines.push(...res.lines);
      log.textContent = state.logLines.join('\n');
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
  } catch (err) { /* ignore */ }
}

/* slice viewer -------------------------------------------------------------------------- */
const AXES = ['x', 'y', 'z'];
const SLICE_AXES = { 0: ['y', 'z'], 1: ['x', 'z'], 2: ['x', 'y'] };   // rows, cols of a slice

async function loadSlice() {
  if (!state.detail || state.detail.job.status !== 'done' || state.detail.job.kind !== 'generate') return;
  const s = state.slice;
  const index = s.index === null ? -1 : s.index;
  try {
    const res = await api(`jobs/${encodeURIComponent(state.selected)}/slice?axis=${s.axis}&index=${index}`);
    if (s.index === null) {
      s.index = Math.floor(res.count / 2);
      if (res.index !== s.index) return loadSlice();
    }
    const raw = atob(res.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    s.data = { ...res, labels: new Int32Array(bytes.buffer) };
    const slider = $('#slice-slider');
    slider.max = res.count - 1;
    slider.value = res.index;
    $('#slice-label').textContent = `${AXES[s.axis]} = ${res.index + 1} / ${res.count}`;
    drawSlice();
  } catch (err) { toast(err.message, true); }
}

function grainColor(id) {
  // golden-angle hues, two lightness levels, so neighbouring IDs differ clearly
  const h = (id * 137.508) % 360;
  const l = id % 2 ? 62 : 48;
  const s = 58 + (id % 3) * 10;
  return hslToRgb(h, s, l);
}

function hslToRgb(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function drawSlice() {
  const s = state.slice;
  if (!s.data) return;
  const { rows, cols, labels } = s.data;
  const canvas = $('#slice-canvas');
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  const meta = state.detail.grains || { phase: [] };
  const cache = new Map();
  const colorOf = id => {
    if (cache.has(id)) return cache.get(id);
    let c;
    if (id <= 0) c = [40, 40, 48];
    else if (s.color === 'phase') {
      const base = hexToRgb(PHASE_COLORS[meta.phase[id]] || '#888888');
      const shade = 0.82 + 0.18 * ((id * 0.618) % 1);
      c = base.map(v => Math.round(v * shade));
    } else c = grainColor(id);
    cache.set(id, c);
    return c;
  };
  for (let r = 0; r < rows; r++) {
    for (let q = 0; q < cols; q++) {
      const i = r * cols + q;
      const id = labels[i];
      let [R, G, B] = colorOf(id);
      if (s.bounds && ((q + 1 < cols && labels[i + 1] !== id) || (r + 1 < rows && labels[i + cols] !== id))) {
        R *= 0.35; G *= 0.35; B *= 0.35;
      }
      const o = 4 * i;
      img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // display size: fit the container, integer zoom when possible
  const wrap = $('#canvas-wrap');
  const avail = Math.min(wrap.clientWidth - 24, window.innerHeight * 0.68);
  const zoom = Math.max(1, Math.floor(avail / Math.max(rows, cols))) || 1;
  canvas.style.width = cols * zoom + 'px';
  canvas.style.height = rows * zoom + 'px';
  const [ra, ca] = SLICE_AXES[s.axis];
  canvas.title = `rows: ${ra}, columns: ${ca}`;
  const present = new Set(meta.phase.filter((p, i) => i > 0 && p));
  $('#phase-legend').replaceChildren(
    el('span', {}, `rows ↓ ${ra}, columns → ${ca}`),
    ...(s.color === 'phase' ? [...present].sort().map(p => el('span', {}, el('i', { style: `background:${PHASE_COLORS[p] || '#888'}` }), PHASE_NAMES[p] || p)) : []));
}

function initViewer() {
  $$('#axis-seg button').forEach(b => b.addEventListener('click', () => {
    $$('#axis-seg button').forEach(x => x.classList.toggle('active', x === b));
    state.slice.axis = Number(b.dataset.axis);
    state.slice.index = null;
    loadSlice();
  }));
  $$('#color-seg button').forEach(b => b.addEventListener('click', () => {
    $$('#color-seg button').forEach(x => x.classList.toggle('active', x === b));
    state.slice.color = b.dataset.color;
    drawSlice();
  }));
  $('#show-bounds').addEventListener('change', e => { state.slice.bounds = e.target.checked; drawSlice(); });
  let sliderTimer = null;
  $('#slice-slider').addEventListener('input', e => {
    state.slice.index = Number(e.target.value);
    $('#slice-label').textContent = `${AXES[state.slice.axis]} = ${state.slice.index + 1} / ${Number(e.target.max) + 1}`;
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(loadSlice, 40);
  });
  const canvas = $('#slice-canvas'), tip = $('#slice-tip');
  canvas.addEventListener('mousemove', e => {
    const s = state.slice;
    if (!s.data) return;
    const rect = canvas.getBoundingClientRect();
    const q = Math.floor((e.clientX - rect.left) / rect.width * s.data.cols);
    const r = Math.floor((e.clientY - rect.top) / rect.height * s.data.rows);
    if (q < 0 || r < 0 || q >= s.data.cols || r >= s.data.rows) return tip.classList.add('hidden');
    const id = s.data.labels[r * s.data.cols + q];
    const meta = state.detail.grains || { phase: [], volume: [] };
    const phase = PHASE_NAMES[meta.phase[id]] || '';
    const [ra, ca] = SLICE_AXES[s.axis];
    tip.textContent = `grain ${id}${phase ? ' · ' + phase : ''}${meta.volume[id] ? ' · target ' + meta.volume[id].toFixed(1) + ' µm³' : ''} · ${ra}=${r + 1} ${ca}=${q + 1}`;
    const wrap = $('#canvas-wrap').getBoundingClientRect();
    tip.style.left = (e.clientX - wrap.left) + 'px';
    tip.style.top = (e.clientY - wrap.top) + 'px';
    tip.classList.remove('hidden');
  });
  canvas.addEventListener('mouseleave', () => tip.classList.add('hidden'));
  window.addEventListener('resize', () => { if (state.tab === 'slices') drawSlice(); });
}

/* histogram ----------------------------------------------------------------------------- */
async function loadHistogram() {
  try {
    const h = await api(`jobs/${encodeURIComponent(state.selected)}/histogram`);
    drawHistogram(h);
  } catch (err) { $('#histogram').textContent = err.message; }
}

function drawHistogram(h) {
  const host = $('#histogram');
  if (!h.edges.length) { host.textContent = 'No grains.'; return; }
  const W = Math.max(320, host.clientWidth || 640), H = 300, m = { l: 44, r: 12, t: 14, b: 40 };
  const nb = h.actual.length;
  const sumA = h.actual.reduce((a, b) => a + b, 0) || 1;
  const sumT = h.target.reduce((a, b) => a + b, 0) || 1;
  const a = h.actual.map(v => v / sumA), t = h.target.map(v => v / sumT);
  const ymax = Math.max(...a, ...t, 1e-9) * 1.1;
  const x = i => m.l + (W - m.l - m.r) * i / nb;
  const y = v => H - m.b - (H - m.t - m.b) * v / ymax;
  const parts = [`<svg class="hist" viewBox="0 0 ${W} ${H}" width="100%" height="${H}">`];
  parts.push(`<line class="axis" x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}"/>`);
  for (let i = 0; i < nb; i++) {
    const bw = Math.max(1, x(i + 1) - x(i) - 2);
    parts.push(`<rect class="rve" x="${x(i) + 1}" y="${y(a[i])}" width="${bw}" height="${H - m.b - y(a[i])}" rx="2"><title>${h.edges[i].toFixed(2)}–${h.edges[i + 1].toFixed(2)} µm: ${h.actual[i]} grains</title></rect>`);
  }
  if (t.length) {
    const pts = t.map((v, i) => `${(x(i) + x(i + 1)) / 2},${y(v)}`).join(' ');
    parts.push(`<polyline class="target" points="${pts}"/>`);
  }
  for (let i = 0; i <= nb; i += Math.ceil(nb / 6)) {
    parts.push(`<text x="${x(i)}" y="${H - m.b + 16}" text-anchor="middle">${h.edges[i].toFixed(1)}</text>`);
  }
  parts.push(`<text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">equivalent diameter [µm]</text>`);
  parts.push(`<text x="12" y="${(H - m.b) / 2}" transform="rotate(-90 12 ${(H - m.b) / 2})" text-anchor="middle">share of grains</text>`);
  parts.push(`<rect class="rve" x="${W - 190}" y="${m.t}" width="12" height="12" rx="3"/><text x="${W - 172}" y="${m.t + 10}">RVE</text>`);
  parts.push(`<line class="target" x1="${W - 120}" y1="${m.t + 6}" x2="${W - 104}" y2="${m.t + 6}"/><text x="${W - 98}" y="${m.t + 10}">Target</text>`);
  parts.push('</svg>');
  host.innerHTML = parts.join('');
}

/* small YAML printer for the config tab ----------------------------------------------------- */
function toYaml(v, indent = '') {
  if (Array.isArray(v)) {
    if (v.every(x => typeof x !== 'object' || x === null)) return '[' + v.map(x => JSON.stringify(x)).join(', ') + ']';
    return v.map(x => `\n${indent}- ` + toYaml(x, indent + '  ').replace(/^\n\s*/, '')).join('');
  }
  if (v && typeof v === 'object') {
    return Object.entries(v).map(([k, x]) => {
      const inner = toYaml(x, indent + '  ');
      return `\n${indent}${k}:` + (inner.startsWith('\n') ? inner : ' ' + inner);
    }).join('');
  }
  return v === null ? 'null' : (typeof v === 'string' ? v : JSON.stringify(v));
}

/* polling ----------------------------------------------------------------------------- */
async function tick() {
  await refreshJobs();
  if (state.view === 'run' && state.selected) {
    const job = state.jobs.find(j => j.id === state.selected);
    const live = state.detail && ['queued', 'running'].includes(state.detail.job.status);
    const meshLive = state.detail && [...(state.detail.meshes || []), ...(state.detail.exports || [])].some(m => ['queued', 'running'].includes(m.status));
    if (job && (live || meshLive || job.status !== state.detail?.job.status)) await loadDetail();
  }
}

/* start ----------------------------------------------------------------------------- */
async function main() {
  initTheme();
  try {
    [state.info, { inputs: state.inputs }] = await Promise.all([api('info'), api('inputs')]);
  } catch (err) {
    document.body.innerHTML = `<p style="padding:40px">Cannot reach the DRAGen server: ${esc(err.message)}</p>`;
    return;
  }
  $('#workdir').textContent = state.info.workdir;
  $('#workdir').title = 'Work directory: ' + state.info.workdir;
  state.config = mergeDefaults(state.info.defaults);
  initForm();
  initViewer();
  writeForm();
  validate();
  refreshPresets();
  $('#new-run').addEventListener('click', () => showView('new'));
  $('#run-filter').addEventListener('input', e => { state.search = e.target.value; renderJobs(); });
  $$('#status-filters .chip').forEach(c => c.addEventListener('click', () => {
    $$('#status-filters .chip').forEach(x => x.classList.toggle('active', x === c));
    state.filter = c.dataset.status;
    renderJobs();
  }));
  $$('#run-tabs button').forEach(b => b.addEventListener('click', () => !b.disabled && setTab(b.dataset.tab)));
  await refreshJobs();
  setInterval(tick, 1500);
}

main();
