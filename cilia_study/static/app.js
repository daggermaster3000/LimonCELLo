'use strict';

// ── helpers ───────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const CLASSES = ['cilia', 'not', 'uncertain'];
const CLABEL = { cilia: 'Cilia', not: 'Not', uncertain: 'Uncertain' };
let user = localStorage.getItem('cc_user') || '';
let isAdmin = localStorage.getItem('cc_admin') === '1';

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('show'));
  $(id).classList.add('show');
  window.scrollTo(0, 0);
}
let toastT;
function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 1600);
}

// ── login ─────────────────────────────────────────────────
async function initLogin() {
  const members = await api('/api/members');
  $('player').innerHTML = members
    .map(m => `<option value="${m.name}">${m.name} · ${m.rank}</option>`).join('');
  if (user) $('player').value = user;
}
$('startBtn').onclick = async () => {
  const name = $('player').value;
  const res = await api('/api/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) { toast(res.error || 'login failed'); return; }
  user = res.user; isAdmin = !!res.is_admin;
  localStorage.setItem('cc_user', user);
  localStorage.setItem('cc_admin', isAdmin ? '1' : '0');
  route();
};
function switchUser() {
  localStorage.removeItem('cc_user'); localStorage.removeItem('cc_admin');
  user = ''; isAdmin = false; show('login');
}
$('logout').onclick = switchUser;

// admins land on the dashboard; everyone else goes straight to rating
function route() { if (isAdmin) enterHome(); else startLabeling(); }

// ── dashboard ─────────────────────────────────────────────
function enterHome() {
  $('whoName').textContent = user;
  show('home');
  loadDashboard();
}
$('refresh').onclick = () => loadDashboard();
$('goLabel').onclick = () => startLabeling();
$('goCurate').onclick = () => startCurate();
// back button on the rating screen: admins return to dashboard, others switch user
$('backHome').onclick = () => { if (isAdmin) enterHome(); else switchUser(); };
$('thanksReview').onclick = () => startLabeling();
$('thanksSwitch').onclick = switchUser;

const pct = (x) => (x == null ? '–' : Math.round(x * 100) + '%');

async function loadDashboard() {
  const d = await api('/api/results?user=' + encodeURIComponent(user));
  if (d && d.error) { toast('stats are restricted'); return; }
  window._res = d;
  const mine = (d.per_user.find(u => u.user === user) || {}).n_labeled || 0;
  $('myprog').textContent = `you: ${mine} / ${d.n_rois} labeled`;

  const anyVotes = d.rois.some(r => r.n > 0);
  $('empty').hidden = anyVotes;
  $('goCurate').hidden = !isAdmin;

  // summary cards
  $('summary').innerHTML = [
    ['Raters', d.n_raters],
    ["Fleiss' κ", d.fleiss_kappa == null ? '–' : d.fleiss_kappa],
    ['Mean pairwise agreement', pct(d.mean_pairwise_agreement)],
    ['Model ↔ human corr (R²)', d.correlation.r2 == null ? '–' : d.correlation.r2],
  ].map(([lbl, v]) => `<div class="stat"><div class="big">${v}</div><div class="lbl">${lbl}</div></div>`).join('');

  renderResetPanel(d);
  renderDisagree(d);
  renderConfusionPicker(d);
  renderAgreement(d);
  renderScatter(d);
}

function voteBar(v) {
  const n = v.cilia + v.not + v.uncertain || 1;
  const w = (x) => (100 * x / n).toFixed(1) + '%';
  return `<div class="votebar"><i class="vc" style="width:${w(v.cilia)}"></i>`
    + `<i class="vn" style="width:${w(v.not)}"></i>`
    + `<i class="vu" style="width:${w(v.uncertain)}"></i></div>`;
}

function renderResetPanel(d) {
  const bar = $('adminReset');
  if (!isAdmin) { bar.hidden = true; return; }
  bar.hidden = false;
  const sel = $('resetUser');
  const opts = ['<option value="*">Everyone</option>']
    .concat(d.raters.map(u => `<option value="${u}">${u}</option>`));
  sel.innerHTML = opts.join('');
}
$('resetBtn').onclick = async () => {
  const target = $('resetUser').value;
  const who = target === '*' ? 'ALL raters' : target;
  if (!confirm(`Reset decisions for ${who}? This cannot be undone.`)) return;
  const res = await api('/api/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ admin: user, target }),
  });
  if (res && res.ok) { toast(`Reset ${who} (${res.removed} labels)`); loadDashboard(); }
  else toast((res && res.error) || 'reset failed');
};

function renderDisagree(d) {
  const top = d.top_disagree.slice(0, 8);
  if (!top.length) { $('disagree').innerHTML = ''; return; }
  $('disagree').innerHTML = top.map(r => `
    <div class="roicell">
      <img src="${r.img}" alt="${r.id}" loading="lazy" />
      <div class="body">
        ${voteBar(r.votes)}
        <div class="meta">
          <span title="normalised entropy">split ${Math.round(r.entropy * 100)}%</span>
          <span class="chip ${r.model_label}">model: ${CLABEL[r.model_label]}</span>
        </div>
      </div>
    </div>`).join('');
}

function renderConfusionPicker(d) {
  const sel = $('cmUser');
  const cur = sel.value && d.raters.includes(sel.value) ? sel.value
    : (d.raters.includes(user) ? user : d.raters[0]);
  sel.innerHTML = d.raters.map(u => `<option value="${u}">${u}</option>`).join('');
  if (cur) sel.value = cur;
  sel.onchange = () => drawMatrices(d, sel.value);
  if (cur) drawMatrices(d, cur);
  else { $('cmModel').innerHTML = $('cmCons').innerHTML = '<p class="muted">no data yet</p>'; }
}

function cmTable(m) {
  const mx = Math.max(1, ...m.flat());
  const cell = (val, diag) => {
    const a = 0.08 + 0.72 * (val / mx);
    const bg = `rgba(37,99,176,${val ? a : 0.03})`;
    return `<td class="${diag ? 'diag' : ''}" style="background:${bg}">${val}</td>`;
  };
  let h = '<table class="cm"><tr><th></th>' + CLASSES.map(c => `<th>${CLABEL[c]}</th>`).join('') + '</tr>';
  m.forEach((row, i) => {
    h += `<tr><td class="rowh">${CLABEL[CLASSES[i]]}</td>`
      + row.map((v, j) => cell(v, i === j)).join('') + '</tr>';
  });
  return h + '</table>';
}

function drawMatrices(d, u) {
  const vm = d.confusion.vs_model[u], vc = d.confusion.vs_consensus[u];
  $('cmModel').innerHTML = vm ? cmTable(vm.matrix) : '<p class="muted">no data</p>';
  $('cmCons').innerHTML = vc && vc.n ? cmTable(vc.matrix) : '<p class="muted">no consensus yet</p>';
  $('accModel').textContent = vm && vm.acc != null ? pct(vm.acc) : '';
  $('accCons').textContent = vc && vc.acc != null ? pct(vc.acc) : '';
}

function heatColor(v) {
  if (v == null) return '#c9ced6';
  // low agreement -> red, high -> green
  const r = Math.round(209 + (31 - 209) * v);
  const g = Math.round(73 + (157 - 73) * v);
  const b = Math.round(91 + (118 - 91) * v);
  return `rgb(${r},${g},${b})`;
}
function renderAgreement(d) {
  const us = d.agreement_matrix.users, m = d.agreement_matrix.matrix;
  if (us.length < 2) { $('agree').innerHTML = '<p class="muted">need at least 2 raters.</p>'; return; }
  const shortName = (n) => n.split(' ')[0];
  let h = '<table class="heat"><tr><th></th>' + us.map(u => `<th>${shortName(u)}</th>`).join('') + '</tr>';
  m.forEach((row, i) => {
    h += `<tr><th>${shortName(us[i])}</th>`;
    row.forEach((v, j) => {
      if (i === j) h += `<td class="self">–</td>`;
      else h += `<td style="background:${heatColor(v)}" title="${us[i]} vs ${us[j]}">${v == null ? '' : Math.round(v * 100)}</td>`;
    });
    h += '</tr>';
  });
  $('agree').innerHTML = h + '</table>';
}

function renderScatter(d) {
  const pts = d.correlation.points;
  const W = 480, H = 320, pad = 44;
  const sx = (x) => pad + x * (W - 2 * pad);
  const sy = (y) => H - pad - y * (H - 2 * pad);
  const col = { cilia: '#1f9d76', not: '#d1495b', uncertain: '#c98a1a' };
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="model vs human scatter">`;
  // frame + grid
  svg += `<rect x="${pad}" y="${pad}" width="${W - 2 * pad}" height="${H - 2 * pad}" fill="none" stroke="#e3e6ea"/>`;
  [0.25, 0.5, 0.75].forEach(g => {
    svg += `<line x1="${sx(g)}" y1="${pad}" x2="${sx(g)}" y2="${H - pad}" stroke="#eef0f3"/>`;
    svg += `<line x1="${pad}" y1="${sy(g)}" x2="${W - pad}" y2="${sy(g)}" stroke="#eef0f3"/>`;
  });
  // 1:1 reference
  svg += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="#c9ced6" stroke-dasharray="4 4"/>`;
  // points
  pts.forEach(p => {
    svg += `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="6" fill="${col[p.model_label]}" fill-opacity="0.72" stroke="#fff"><title>${p.id} · model ${p.x} · human ${Math.round(p.y * 100)}%</title></circle>`;
  });
  // axes labels
  svg += `<text x="${W / 2}" y="${H - 10}" text-anchor="middle" font-size="12" fill="#6b7684">model P(cilia)</text>`;
  svg += `<text x="14" y="${H / 2}" text-anchor="middle" font-size="12" fill="#6b7684" transform="rotate(-90 14 ${H / 2})">humans calling "cilia"</text>`;
  const r2 = d.correlation.r2, r = d.correlation.r;
  if (r2 != null) svg += `<text x="${W - pad}" y="${pad + 4}" text-anchor="end" font-size="13" fill="#1c2430" font-weight="600">r = ${r} · R² = ${r2}</text>`;
  else if (!pts.length) svg += `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" font-size="13" fill="#6b7684">no votes yet</text>`;
  svg += '</svg>';
  $('scatter').innerHTML = svg;
}

// ── curate (admin) ────────────────────────────────────────
let pool = [], picked = new Set();

function scoreColor(p) {                       // green=cilia, red=not, amber=~0.5
  if (p >= 0.66) return '#1f9d76';
  if (p <= 0.33) return '#d1495b';
  return '#c98a1a';
}
async function startCurate() {
  const d = await api('/api/pool?user=' + encodeURIComponent(user));
  if (d && d.error) { toast('admins only'); return; }
  pool = d.pool;
  picked = new Set(pool.filter(p => p.selected).map(p => p.id));
  show('curate');
  renderPool();
}
function renderPool() {
  $('curCount').textContent = `${picked.size} selected`;
  $('pool').innerHTML = pool.map(p => `
    <div class="pooltile ${picked.has(p.id) ? 'on' : ''}" data-id="${p.id}">
      <img src="${p.img}" alt="${p.id}" loading="lazy" />
      <span class="sc"><span class="dot" style="background:${scoreColor(p.model_score)}"></span>${p.model_score.toFixed(2)}</span>
      <span class="tick">✓</span>
    </div>`).join('');
  $('pool').querySelectorAll('.pooltile').forEach(t =>
    t.onclick = () => togglePick(t.dataset.id));
}
function togglePick(id) {
  if (picked.has(id)) picked.delete(id); else picked.add(id);
  const t = $('pool').querySelector(`[data-id="${id}"]`);
  if (t) t.classList.toggle('on', picked.has(id));
  $('curCount').textContent = `${picked.size} selected`;
}
$('curBack').onclick = () => enterHome();
$('curNone').onclick = () => { picked.clear(); renderPool(); };
$('curAmb').onclick = () => {                   // auto-pick the 15 nearest 0.5
  picked = new Set(pool.slice().sort((a, b) =>
    Math.abs(a.model_score - 0.5) - Math.abs(b.model_score - 0.5))
    .slice(0, 15).map(p => p.id));
  renderPool();
};
$('curSave').onclick = async () => {
  if (!picked.size) { toast('pick at least one ROI'); return; }
  const res = await api('/api/select', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ admin: user, ids: [...picked] }),
  });
  if (res && res.ok) { toast(`Published ${res.selected_count} ROIs`); enterHome(); }
  else toast((res && res.error) || 'save failed');
};

// ── labeling ──────────────────────────────────────────────
let items = [], mine = {}, cur = 0;

async function startLabeling() {
  const ds = await api('/api/dataset');
  items = ds;
  const m = await api('/api/mine?user=' + encodeURIComponent(user));
  mine = (m && m.labels) || {};
  // resume at first unlabeled ROI
  cur = items.findIndex(it => !mine[it.id]);
  if (cur < 0) cur = 0;
  show('label');
  renderCard();
}

function renderCard() {
  const it = items[cur];
  $('roiImg').src = it.img;
  $('labelProg').textContent = `${Object.keys(mine).length} / ${items.length}`;
  $('pbar').style.width = (100 * Object.keys(mine).length / items.length) + '%';
  document.querySelectorAll('.btn.choice').forEach(b =>
    b.classList.toggle('sel', mine[it.id] === b.dataset.label));
  const done = Object.keys(mine).length;
  $('prevPick').textContent = mine[it.id]
    ? `your call: ${CLABEL[mine[it.id]]} (tap another to change)`
    : `ROI ${cur + 1} of ${items.length}`;
  $('prevBtn').disabled = cur === 0;
  $('nextBtn').textContent = (done >= items.length) ? 'Done ›' : 'Next ›';
}

async function pick(label) {
  const it = items[cur];
  mine[it.id] = label;
  document.querySelectorAll('.btn.choice').forEach(b =>
    b.classList.toggle('sel', label === b.dataset.label));
  await api('/api/label', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user, id: it.id, label }),
  });
  // auto-advance shortly
  setTimeout(() => {
    if (cur < items.length - 1) { cur += 1; renderCard(); }
    else if (Object.keys(mine).length >= items.length) { finishLabeling(); }
    else { renderCard(); }
  }, 260);
}

function finishLabeling() {
  if (isAdmin) { toast('All done'); enterHome(); }
  else show('thanks');
}

document.querySelectorAll('.btn.choice').forEach(b =>
  b.onclick = () => pick(b.dataset.label));
$('prevBtn').onclick = () => { if (cur > 0) { cur -= 1; renderCard(); } };
$('nextBtn').onclick = () => {
  if (cur < items.length - 1) { cur += 1; renderCard(); }
  else { finishLabeling(); }
};
window.addEventListener('keydown', (e) => {
  if (!$('label').classList.contains('show')) return;
  if (e.key === '1') pick('cilia');
  else if (e.key === '2') pick('not');
  else if (e.key === '3') pick('uncertain');
  else if (e.key === 'ArrowLeft') $('prevBtn').click();
  else if (e.key === 'ArrowRight') $('nextBtn').click();
});

// ── boot ──────────────────────────────────────────────────
initLogin().then(() => { if (user) route(); });
if ('serviceWorker' in navigator)
  navigator.serviceWorker.register('/sw.js').catch(() => {});
