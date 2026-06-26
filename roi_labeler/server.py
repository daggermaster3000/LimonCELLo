"""
Cilia ROI Labeler — retro PWA backend (stdlib only, no FastAPI needed).

A tiny threaded HTTP server that serves a phone-friendly, arcade-styled PWA where
logged-in lab members swipe/keep-reject per-cilium ROI thumbnails. Each answer is
appended to ``<run>/csv/labels.csv`` (per user, with timestamp) and a consensus
``<run>/csv/human_validation.csv`` (the format the pipeline + data app already
read) is regenerated on every submission — so multiple labelers' answers land in
``human_validation.csv`` as a majority vote while every individual answer is kept.

Run:
    python roi_labeler/server.py --run "<lc-analysis run dir>" --port 8000

Then open http://<this-machine-ip>:8000 on any phone on the same network.
Defaults to the bundled tutorial run if --run is omitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
_DEFAULT_RUN = (HERE.parent / "tutorial" / "output"
                / "lc-analysis-2026-06-17_14-33-07")

_LOCK = threading.Lock()


# ── ROI dataset ──────────────────────────────────────────────────────────────
def _img_stem(filename: str) -> str:
    s = str(filename)
    for ext in (".ims", ".tif", ".tiff", ".czi", ".nd2", ".lif", ".png"):
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s


def load_rois(run_dir: Path) -> list[dict]:
    """Build the labeling queue: every cilium with an ROI PNG on disk.

    Prefers the canonical (filename, cilia_id) pairs from ``all_cilia_features``
    so the consensus CSV matches the pipeline; falls back to parsing the ROI PNG
    filenames when the workbook is absent.
    """
    roi_dir = run_dir / "figures" / "cilia_rois"
    pngs = {p.name for p in roi_dir.glob("*.png")} if roi_dir.is_dir() else set()
    out: list[dict] = []
    xls = run_dir / "csv" / "all_cilia_features.xlsx"
    if xls.exists():
        try:
            import pandas as pd
            df = pd.read_excel(xls, sheet_name="all_data")
            if "object_type" in df.columns:
                df = df[df["object_type"] == "cilia"]
            for _, r in df.iterrows():
                cid = r.get("cilia_id")
                fn = r.get("filename")
                if cid is None or fn is None or (isinstance(cid, float) and cid != cid):
                    continue
                cid = int(cid)
                stem = _img_stem(fn)
                png = f"{stem}_cilia{cid}.png"
                if png in pngs:
                    out.append({"id": f"{stem}__cilia{cid}", "filename": str(fn),
                                "cilia_id": cid, "png": png})
            if out:
                return out
        except Exception as exc:                              # noqa: BLE001
            print(f"[labeler] could not read {xls}: {exc} — scanning PNGs instead")
    # Fallback: parse the PNG names (filename := stem, no extension known).
    for png in sorted(pngs):
        if "_cilia" not in png:
            continue
        stem, tail = png.rsplit("_cilia", 1)
        try:
            cid = int(os.path.splitext(tail)[0])
        except ValueError:
            continue
        out.append({"id": f"{stem}__cilia{cid}", "filename": stem,
                    "cilia_id": cid, "png": png})
    return out


# ── persistence ──────────────────────────────────────────────────────────────
def _labels_path(run_dir: Path) -> Path:
    return run_dir / "csv" / "labels.csv"


def _val_path(run_dir: Path) -> Path:
    return run_dir / "csv" / "human_validation.csv"


def read_labels(run_dir: Path) -> list[dict]:
    p = _labels_path(run_dir)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_label(run_dir: Path, user: str, filename: str, cilia_id: int,
                 keep: bool) -> None:
    """Append one answer to labels.csv and regenerate the consensus
    human_validation.csv (majority vote per cilium; ties → keep)."""
    p = _labels_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "user", "filename", "cilia_id", "keep"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user, filename,
                    cilia_id, int(bool(keep))])
    _rebuild_consensus(run_dir)


def _rebuild_consensus(run_dir: Path) -> None:
    votes: dict[tuple[str, int], list[int]] = {}
    for row in read_labels(run_dir):
        try:
            key = (str(row["filename"]), int(row["cilia_id"]))
        except (KeyError, ValueError):
            continue
        votes.setdefault(key, []).append(int(row["keep"]))
    with _val_path(run_dir).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "cilia_id", "human_validated"])
        for (fn, cid), vs in votes.items():
            keep = sum(vs) >= (len(vs) - sum(vs))      # ties → keep (True)
            w.writerow([fn, cid, bool(keep)])


def user_done_ids(run_dir: Path, user: str, by_id: dict) -> set[str]:
    """ROI ids already answered by ``user`` (matched back to the queue id)."""
    key_to_id = {(r["filename"], r["cilia_id"]): r["id"] for r in by_id.values()}
    done = set()
    for row in read_labels(run_dir):
        if row.get("user") != user:
            continue
        try:
            rid = key_to_id.get((str(row["filename"]), int(row["cilia_id"])))
        except ValueError:
            rid = None
        if rid:
            done.add(rid)
    return done


def leaderboard(run_dir: Path) -> list[dict]:
    counts: dict[str, int] = {}
    for row in read_labels(run_dir):
        u = row.get("user") or "?"
        counts[u] = counts.get(u, 0) + 1
    return sorted(({"user": u, "count": c} for u, c in counts.items()),
                  key=lambda d: -d["count"])


# ── HTTP handler ─────────────────────────────────────────────────────────────
_CLAIM_TTL = 120.0      # seconds a handed-out ROI is reserved before it frees


class App:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.members = json.loads((HERE / "members.json").read_text("utf-8"))
        self.member_names = {m["name"] for m in self.members}
        self.rois = load_rois(run_dir)
        self.by_id = {r["id"]: r for r in self.rois}
        self.key_to_id = {(r["filename"], r["cilia_id"]): r["id"]
                          for r in self.rois}
        # One shared, shuffled work pool — labels are distributed across players.
        import random
        self.order = list(self.rois)
        random.Random(1234).shuffle(self.order)
        # In-flight claims: roi_id → (user, expiry_ts). Stops two players getting
        # the same ROI at once; expires so an idle player's ROI returns to pool.
        self.claims: dict[str, tuple[str, float]] = {}
        print(f"[labeler] run={run_dir}")
        print(f"[labeler] {len(self.rois)} ROIs · {len(self.members)} members")

    # ── shared-pool work distribution (call under _LOCK) ──────────────────────
    def global_done(self) -> set[str]:
        """ROI ids that have at least one label from anyone."""
        done = set()
        for row in read_labels(self.run_dir):
            try:
                rid = self.key_to_id.get((str(row["filename"]), int(row["cilia_id"])))
            except (KeyError, ValueError):
                rid = None
            if rid:
                done.add(rid)
        return done

    def purge_claims(self):
        now = time.time()
        for rid in [k for k, (_, exp) in self.claims.items() if exp < now]:
            del self.claims[rid]

    def claim_next(self, user: str, gdone: set[str]):
        """Next unlabeled ROI not currently claimed by another player; reserve it
        for ``user``. Re-hands the user their own existing claim first."""
        now = time.time()
        mine = [rid for rid, (u, _) in self.claims.items() if u == user]
        for rid in mine:
            if rid not in gdone:
                self.claims[rid] = (user, now + _CLAIM_TTL)
                return self.by_id[rid]
        for r in self.order:
            rid = r["id"]
            if rid in gdone:
                continue
            c = self.claims.get(rid)
            if c and c[0] != user and c[1] > now:
                continue                                  # claimed by someone else
            self.claims[rid] = (user, now + _CLAIM_TTL)
            return r
        return None

    def release(self, rid: str):
        self.claims.pop(rid, None)

    def user_count(self, user: str) -> int:
        return sum(1 for row in read_labels(self.run_dir)
                   if row.get("user") == user)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                            # quieter logs
            pass

        # ---- helpers ----
        def _send(self, code, body=b"", ctype="application/json", extra=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj), "application/json")

        def _static(self, rel):
            # Map "/" → index.html; serve only files under static/.
            rel = rel.lstrip("/") or "index.html"
            path = (STATIC / rel).resolve()
            if not str(path).startswith(str(STATIC)) or not path.is_file():
                return self._send(404, "not found", "text/plain")
            ctype = {
                ".html": "text/html", ".js": "application/javascript",
                ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml",
                ".webmanifest": "application/manifest+json",
                ".json": "application/json", ".ico": "image/x-icon",
            }.get(path.suffix, "application/octet-stream")
            self._send(200, path.read_bytes(), ctype)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:                                 # noqa: BLE001
                return {}

        # ---- routing ----
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path
            if path == "/api/members":
                return self._json(app.members)
            if path == "/api/img":
                rid = (q.get("id") or [""])[0]
                roi = app.by_id.get(rid)
                if not roi:
                    return self._send(404, "no roi", "text/plain")
                fp = app.run_dir / "figures" / "cilia_rois" / roi["png"]
                if not fp.is_file():
                    return self._send(404, "no img", "text/plain")
                return self._send(200, fp.read_bytes(), "image/png",
                                  {"Cache-Control": "max-age=86400"})
            if path == "/api/next":
                user = (q.get("user") or [""])[0]
                if user not in app.member_names:
                    return self._json({"error": "login"}, 401)
                with _LOCK:
                    gdone = app.global_done()
                    app.purge_claims()
                    nxt = app.claim_next(user, gdone)
                    mine = app.user_count(user)
                    board = leaderboard(app.run_dir)
                total = len(app.rois)
                answered = len(gdone)            # GLOBAL progress (shared work)
                if nxt is None:                  # whole pool labelled by the team
                    return self._json({"done": True, "answered": answered,
                                       "total": total, "mine": mine,
                                       "winner": board[0] if board else None})
                return self._json({
                    "id": nxt["id"], "filename": nxt["filename"],
                    "cilia_id": nxt["cilia_id"],
                    "img": f"/api/img?id={nxt['id']}",
                    "answered": answered, "total": total, "mine": mine})
            if path == "/api/leaderboard":
                with _LOCK:
                    return self._json(leaderboard(app.run_dir))
            if path.startswith("/api/"):
                return self._json({"error": "unknown"}, 404)
            # static / PWA shell
            return self._static(path)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/login":
                name = (self._body().get("name") or "").strip()
                if name in app.member_names:
                    return self._json({"ok": True, "user": name})
                return self._json({"ok": False, "error": "not a group member"}, 403)
            if u.path == "/api/label":
                b = self._body()
                user = (b.get("user") or "").strip()
                rid = b.get("id") or ""
                keep = bool(b.get("keep"))
                if user not in app.member_names:
                    return self._json({"error": "login"}, 401)
                roi = app.by_id.get(rid)
                if not roi:
                    return self._json({"error": "no roi"}, 404)
                with _LOCK:
                    append_label(app.run_dir, user, roi["filename"],
                                 roi["cilia_id"], keep)
                    app.release(rid)
                    answered = len(app.global_done())     # global shared progress
                    mine = app.user_count(user)
                return self._json({"ok": True, "answered": answered, "mine": mine,
                                   "total": len(app.rois)})
            return self._json({"error": "unknown"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(_DEFAULT_RUN),
                    help="lc-analysis run dir (csv/ + figures/cilia_rois/)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run_dir = Path(args.run).resolve()
    if not (run_dir / "figures" / "cilia_rois").is_dir():
        raise SystemExit(f"No figures/cilia_rois in {run_dir}")
    app = App(run_dir)
    if not app.rois:
        raise SystemExit("No ROIs found to label.")
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"[labeler] serving on http://{ip}:{args.port}  (phones on same wifi)")
    print(f"[labeler] local:      http://localhost:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[labeler] bye")


if __name__ == "__main__":
    main()
