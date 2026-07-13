# Cilia Consensus — a "what counts as a cilium?" study PWA

A tiny, self-contained progressive web app to show, live in a talk, that people
**disagree on what a cilium is**. Every group member logs in on their phone,
judges the **same 20 cilia ROIs** as **cilia / not / uncertain**, and a dashboard
shows how much they (dis)agree and how they compare to the `cilialpha-3` CNN.

- **Standard library only** — no pip installs to run the server.
- `build_dataset.py` scores the model's training ROIs and builds a **pool of 50**
  candidates spanning the whole score range. In the app, **Quillan** curates which
  subset the group actually sees (**Curate ROIs** screen) — e.g. the ambiguous
  ones (score ≈ 0.5). A default set of the 15 most-ambiguous is pre-selected.
- The model's scores are shown only in the admin's Curate/dashboard views —
  labelers never see them (ROIs are shown under opaque ids).

## 1. Build the dataset (once)

Scores the tutorial ROIs with `models/cilialpha-3.pt` and copies the chosen 20
into `static/rois/` plus a `dataset.json` manifest. Needs `torch` + `pillow`
(already used by LimonCELLo's AI validator):

```powershell
python cilia_study/build_dataset.py
```

## 2. Run the server

```powershell
python cilia_study/server.py --host 0.0.0.0 --port 8000
```

## 3. Have people join

On the presenting laptop find your LAN IP (`ipconfig` → IPv4). Everyone on the
**same wifi** opens `http://<that-IP>:8000` on their phone (installable via
"Add to Home Screen"). Pick your name → **Start labeling** → tap Cilia / Not /
Uncertain for each of the 20 (keys 1/2/3 on desktop). Answers can be changed;
progress is saved server-side.

## 4. Present the dashboard

**Only `Quillan Favey` sees the statistics** (enforced server-side — everyone
else goes straight to rating and gets a "thank you" when done; the `/api/results`
endpoint returns 403 for any other user). Change the admin by editing `ADMIN`
near the top of `server.py`.

As the admin you also get a **Curate ROIs** button: a grid of all 50 pool
candidates (each tagged with the model's score) where you tap to pick which ones
are published to the group, then **Save**. "Auto-pick ambiguous" selects the 15
nearest 0.5. The dashboard stats cover only the currently-published ROIs.

Logged in as the admin, the homepage updates as votes come in (tap **Refresh**):

- **Summary** — #raters, Fleiss' κ, mean pairwise agreement, model↔human R².
- **Where do we disagree most?** — ROIs ranked by vote entropy, with the model's
  own call for contrast.
- **You vs. model & group** — per-rater 3×3 confusion matrices (against the model
  and against the group majority) with accuracy.
- **Who agrees with whom?** — user × user agreement heatmap.
- **Model vs. humans** — scatter of model P(cilia) vs. fraction of people calling
  each ROI a cilium, with r / R².

## Files

- `build_dataset.py` — one-off dataset builder (reuses `limoncello.ml.roi_validator`).
- `server.py` — stdlib HTTP server + all analytics.
- `members.json` — login roster (Bachmann group).
- `dataset.json` — generated manifest (`id`, model score/label, source).
- `labels.csv` — generated vote log (`ts,user,roi_id,label`, latest wins).
- `static/` — PWA (`index.html`, `app.js`, `style.css`, `manifest.webmanifest`,
  `sw.js`, icons, `rois/`).

To reset all votes, delete `cilia_study/labels.csv`.
