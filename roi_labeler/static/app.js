/* CILIA QUEST — front-end game logic (vanilla JS). */
const $ = (id) => document.getElementById(id);
const api = (p, opt) => fetch(p, opt).then(r => r.json());

let user = localStorage.getItem('cq_user') || '';
let score = +(localStorage.getItem('cq_score') || 0);
let combo = 1;
let current = null;          // current ROI {id, filename, cilia_id, img, answered, total}
let busy = false;

/* ── tiny WebAudio chiptune SFX ── */
let actx;
function beep(freq, dur = 0.08, type = 'square', vol = 0.15) {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), g = actx.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.value = vol; o.connect(g); g.connect(actx.destination);
    o.start();
    g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + dur);
    o.stop(actx.currentTime + dur);
  } catch (e) {}
}
const sfxKeep = () => { beep(660, .07); setTimeout(() => beep(990, .09), 70); };
const sfxReject = () => { beep(200, .12, 'sawtooth'); };
const sfxCombo = () => { beep(880, .05); setTimeout(() => beep(1320, .06), 60); setTimeout(() => beep(1760, .08), 120); };
const sfxStart = () => { [523, 659, 784, 1046].forEach((f, i) => setTimeout(() => beep(f, .1), i * 90)); };

/* ── screen switching ── */
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('show'));
  $(id).classList.add('show');
}
function toast(msg) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('show'), 1400);
}

/* ── login ── */
async function initLogin() {
  const members = await api('/api/members');
  const sel = $('player');
  sel.innerHTML = members.map(m => `<option value="${m.name}">${m.name}  ·  ${m.rank}</option>`).join('');
  if (user) sel.value = user;
}
$('startBtn').onclick = async () => {
  const name = $('player').value;
  const res = await api('/api/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  if (!res.ok) { toast(res.error || 'login failed'); return; }
  user = res.user; localStorage.setItem('cq_user', user);
  sfxStart();
  startGame();
};

function startGame() {
  $('hudPlayer').textContent = user.split(' ')[0].toUpperCase();
  updateScore(0);
  show('game');
  nextCard();
}

/* ── scoring ── */
function updateScore(delta) {
  score = Math.max(0, score + delta);
  localStorage.setItem('cq_score', score);
  $('hudScore').textContent = score;
  $('hudCombo').textContent = 'x' + combo;
}

/* ── cards ── */
async function nextCard() {
  busy = true;
  const d = await api(`/api/next?user=${encodeURIComponent(user)}`);
  if (d.error === 'login') { show('login'); return; }
  if (d.done) {
    const w = d.winner;
    const champ = w ? `🍦 ${w.user.split(' ')[0].toUpperCase()} WINS THE ICE CREAM!<br>(${w.count} labelled)` : '';
    $('doneStats').innerHTML = `TEAM DONE ${d.answered}/${d.total}<br>YOU ${d.mine || 0}<br><br>${champ}`;
    show('done'); return;
  }
  current = d;
  const img = $('roi');
  img.style.transform = 'translateX(0) rotate(0)'; img.style.opacity = 1;
  img.src = d.img;
  $('metaId').textContent = `#${d.cilia_id} · ${trim(d.filename)}`;
  // GLOBAL progress — the whole team shares one pool of ROIs.
  $('progBar').style.width = (100 * d.answered / Math.max(1, d.total)) + '%';
  $('progText').textContent = `TEAM ${d.answered}/${d.total}  ·  YOU ${d.mine || 0}`;
  busy = false;
}
const trim = (s) => s.length > 26 ? '…' + s.slice(-24) : s;

async function answer(keep) {
  if (busy || !current) return;
  busy = true;
  flyOut(keep);
  showStamp(keep);
  if (keep) { sfxKeep(); } else { sfxReject(); }
  // Combo grows every 5 answers (keep the labeller in flow), capped at x8.
  streakCount += 1;
  combo = Math.min(8, 1 + Math.floor(streakCount / 5));
  if (combo > lastCombo) { sfxCombo(); toast('COMBO x' + combo + '!'); }
  lastCombo = combo;
  updateScore(10 * combo);
  const id = current.id;
  await api('/api/label', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user, id, keep })
  });
  setTimeout(nextCard, 230);
}
let streakCount = 0, lastCombo = 1;

function showStamp(keep) {
  const s = $('stamp');
  s.textContent = keep ? '✓ KEEP' : '✘ JUNK';
  s.className = 'stamp ' + (keep ? 'keep' : 'reject');
  void s.offsetWidth; s.classList.add('show');
}
function flyOut(keep) {
  const img = $('roi');
  img.style.transition = 'transform .22s, opacity .22s';
  img.style.transform = `translateX(${keep ? 120 : -120}%) rotate(${keep ? 18 : -18}deg)`;
  img.style.opacity = 0;
  setTimeout(() => { img.style.transition = 'transform .03s'; }, 230);
}

$('keepBtn').onclick = () => answer(true);
$('rejectBtn').onclick = () => answer(false);
document.addEventListener('keydown', (e) => {
  if (!$('game').classList.contains('show')) return;
  if (e.key === 'ArrowRight') answer(true);
  if (e.key === 'ArrowLeft') answer(false);
});

/* ── swipe ── */
const card = $('card');
let sx = 0, sy = 0, dragging = false;
card.addEventListener('pointerdown', (e) => { sx = e.clientX; sy = e.clientY; dragging = true; });
card.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - sx;
  $('roi').style.transform = `translateX(${dx}px) rotate(${dx / 18}deg)`;
  card.classList.toggle('swipe-keep', dx > 40);
  card.classList.toggle('swipe-reject', dx < -40);
});
card.addEventListener('pointerup', (e) => {
  if (!dragging) return; dragging = false;
  const dx = e.clientX - sx;
  card.classList.remove('swipe-keep', 'swipe-reject');
  if (Math.abs(dx) > 70) answer(dx > 0);
  else $('roi').style.transform = 'translateX(0) rotate(0)';
});

/* ── gallery (batch labelling) ── */
const GAL_N = 24;
let galItems = [];           // [{id,img,cilia_id,filename,guess,keep}]
let galBusy = false;

function enterGallery() {
  $('galPlayer').textContent = user.split(' ')[0].toUpperCase();
  show('gallery');
  loadGallery();
}

async function loadGallery() {
  if (galBusy) return;
  galBusy = true;
  $('grid').innerHTML = '<div class="prog-text" style="position:static">LOADING…</div>';
  const d = await api(`/api/batch?user=${encodeURIComponent(user)}&n=${GAL_N}`);
  galBusy = false;
  if (d.error === 'login') { show('login'); return; }
  setGalProgress(d);
  if (d.done || !d.items.length) {
    const board = await api('/api/leaderboard');
    const w = board[0];
    const champ = w ? `🍦 ${w.user.split(' ')[0].toUpperCase()} WINS THE ICE CREAM!<br>(${w.count} labelled)` : '';
    $('doneStats').innerHTML = `TEAM DONE ${d.answered}/${d.total}<br>YOU ${d.mine || 0}<br><br>${champ}`;
    show('done'); return;
  }
  // Pre-select from the model's guess (1 = cilium). Unknown → junk + ★ flag.
  galItems = d.items.map(it => ({ ...it, keep: it.guess === 1 }));
  renderGrid();
}

function renderGrid() {
  $('grid').innerHTML = galItems.map((it, i) =>
    `<div class="tile ${it.keep ? 'keep' : ''}" data-i="${i}">
       <img src="${it.img}" loading="lazy" alt="roi">
       ${it.guess == null ? '<span class="unsure">★</span>' : ''}
       <span class="mark"></span>
     </div>`).join('');
  updatePicked();
}

function updatePicked() {
  $('galPicked').textContent = galItems.filter(it => it.keep).length;
}

function setGalProgress(d) {
  $('galMine').textContent = d.mine || 0;
  $('galProgBar').style.width = (100 * d.answered / Math.max(1, d.total)) + '%';
  $('galProgText').textContent = `TEAM ${d.answered}/${d.total}  ·  YOU ${d.mine || 0}`;
}

$('grid').addEventListener('click', (e) => {
  const tile = e.target.closest('.tile');
  if (!tile) return;
  const i = +tile.dataset.i;
  galItems[i].keep = !galItems[i].keep;
  tile.classList.toggle('keep', galItems[i].keep);
  galItems[i].keep ? beep(660, .05) : beep(220, .06, 'sawtooth');
  updatePicked();
});

$('galKeepAll').onclick = () => { galItems.forEach(it => it.keep = true); renderGrid(); sfxCombo(); };
$('galJunkAll').onclick = () => { galItems.forEach(it => it.keep = false); renderGrid(); sfxReject(); };

$('galSubmit').onclick = async () => {
  if (galBusy || !galItems.length) return;
  galBusy = true;
  const items = galItems.map(it => ({ id: it.id, keep: it.keep }));
  const res = await api('/api/label_batch', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user, items })
  });
  galBusy = false;
  if (res && res.ok) {
    updateScore(10 * res.n);
    sfxKeep();
    toast(`+${res.n} LABELLED`);
    setGalProgress(res);
    loadGallery();
  } else {
    toast((res && res.error) || 'submit failed');
  }
};

$('galleryBtn').onclick = enterGallery;
$('galSwipe').onclick = () => { show('game'); nextCard(); };

/* ── leaderboard ── */
async function showBoard() {
  const data = await api('/api/leaderboard');
  $('boardList').innerHTML = data.map((d, i) =>
    `<li class="${d.user === user ? 'me' : ''}"><span class="rk">${medal(i)}</span>
     <span>${d.user.split(' ')[0]}</span><span>${d.count}</span></li>`).join('')
    || '<li>no scores yet</li>';
  show('board');
}
const medal = (i) => ['🍦', '🥈', '🥉'][i] || (i + 1) + '.';   // #1 = ice cream
$('boardBtn').onclick = showBoard;
$('boardBtn2').onclick = showBoard;
$('backBtn').onclick = () => show(current ? 'game' : 'login');

/* ── boot ── */
initLogin().then(() => { if (user) startGame(); });
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
