# Cilia Quest — ROI labeling PWA

A phone-friendly, retro-arcade Progressive Web App for crowd-labeling per-cilium
ROI thumbnails. Group members log in, swipe **keep / junk** on each ROI, and every
answer is recorded — both per-user and as a pipeline-compatible consensus.

- **Login** is restricted to the Bachmann group roster (`members.json`).
- Each answer → appended to `<run>/csv/labels.csv` (`ts, user, filename, cilia_id, keep`).
- A majority-vote **`<run>/csv/human_validation.csv`** (`filename, cilia_id, human_validated`)
  is regenerated on every submit — the exact format the pipeline and the data app
  already read, so labels flow straight back into the analysis.
- Game feel: score, combo multiplier, chiptune SFX, CRT scanlines, high-score board,
  swipe gestures (← junk / keep →), installable to the home screen.

Stdlib only — no FastAPI/Flask. Uses `pandas` + `Pillow` (already in the repo) to
read the ROI list and serve thumbnails.

## Run

```bash
# default dataset = the bundled tutorial run (544 ROIs)
python roi_labeler/server.py

# or point it at any finished run with figures/cilia_rois/ + csv/all_cilia_features.xlsx
python roi_labeler/server.py --run "C:/path/to/lc-analysis-YYYY-..." --port 8000
```

On start it prints a LAN URL like `http://192.168.x.x:8000`. Open that on any phone
on the **same wifi**, then "Add to Home Screen" to install it as an app.

## How a run gets its ROIs

The labeler serves the `figures/cilia_rois/*.png` thumbnails a batch run produces
(Capture screenshots + per-cilium ROIs enabled). To label the new training set,
first run the pipeline over `Thomaso/training_dataset/files.txt` with ROI export on,
then point `--server.py --run` at that output folder.

## Files

| File | Role |
|------|------|
| `server.py` | stdlib HTTP server: members, login, next-ROI, image, label, leaderboard |
| `members.json` | login roster (name + rank) |
| `static/index.html` `style.css` `app.js` | the retro PWA front-end |
| `static/sw.js` `manifest.webmanifest` `icon-*.png` | PWA install/offline shell |

## API (for reference)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/members` | login roster |
| `POST /api/login {name}` | validate against roster |
| `GET /api/next?user=` | next ROI this user hasn't labeled (`{id,filename,cilia_id,img,answered,total}`) |
| `GET /api/img?id=` | ROI thumbnail PNG |
| `POST /api/label {user,id,keep}` | record an answer → labels.csv + consensus |
| `GET /api/leaderboard` | per-user answer counts |

## Notes

- Each user gets a stable, per-user shuffle of the queue, so labelers don't all see
  the same order (reduces position bias); consensus needs ≥2 labelers per ROI to
  outvote a single mistake.
- Login is roster-only (no password) — it's an internal LAN tool. Add a per-user
  PIN if you ever expose it beyond the lab network.
