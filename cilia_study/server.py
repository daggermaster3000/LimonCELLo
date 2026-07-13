"""Cilia Consensus Study — tiny self-contained PWA server (stdlib only).

Every Bachmann-group member logs in (roster in ``members.json``), judges the
*same* 20 cilia ROIs (built by ``build_dataset.py``) as **cilia / not /
uncertain**, and the homepage dashboard shows how much people (dis)agree, how
they compare to the ``cilialpha-3`` CNN, confusion matrices and correlations.

No external dependencies — Python 3.8+ standard library only.

    python cilia_study/server.py --host 0.0.0.0 --port 8000

Then group members open ``http://<your-LAN-ip>:8000`` on their phones.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ROI_DIR = STATIC / "rois"
DATASET = HERE / "dataset.json"
SELECTION = HERE / "selection.json"
MEMBERS = HERE / "members.json"
LABELS = HERE / "labels.csv"

CLASSES = ("cilia", "not", "uncertain")
ADMIN = "Quillan Favey"          # only this user may see the statistics dashboard
_LOCK = threading.Lock()

_CTYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".webmanifest": "application/manifest+json", ".png": "image/png",
    ".svg": "image/svg+xml", ".ico": "image/x-icon",
}


# ─────────────────────────────────────────────────────────────────────────────
# In-memory app state
# ─────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self) -> None:
        self.dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        self.by_id = {d["id"]: d for d in self.dataset}
        self.members = json.loads(MEMBERS.read_text(encoding="utf-8"))
        self.member_names = {m["name"] for m in self.members}
        # labels[(user, roi_id)] = {"label": str, "ts": str} ; latest wins
        self.labels: dict[tuple[str, str], dict] = {}
        self._load_labels()
        self.selected: list[str] = self._load_selection()

    def _load_selection(self) -> list[str]:
        """The subset of pool ROIs shown to the group (admin-curated)."""
        if SELECTION.is_file():
            try:
                got = json.loads(SELECTION.read_text(encoding="utf-8")).get("selected", [])
                sel = [r for r in got if r in self.by_id]
                if sel:
                    return sel
            except Exception:
                pass
        return [d["id"] for d in self.dataset]           # fallback: whole pool

    def set_selection(self, ids: list[str]) -> list[str]:
        sel = [d["id"] for d in self.dataset if d["id"] in set(ids)]  # keep pool order
        with _LOCK:
            self.selected = sel
            SELECTION.write_text(json.dumps({"selected": sel}, indent=2), encoding="utf-8")
        return sel

    def shown(self) -> list[dict]:
        """Pool entries currently shown to the group, in pool order."""
        sset = set(self.selected)
        return [d for d in self.dataset if d["id"] in sset]

    def _load_labels(self) -> None:
        if not LABELS.is_file():
            return
        with LABELS.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("roi_id") in self.by_id and row.get("label") in CLASSES:
                    self.labels[(row["user"], row["roi_id"])] = {
                        "label": row["label"], "ts": row.get("ts", "")}

    def record(self, user: str, roi_id: str, label: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _LOCK:
            new = not LABELS.is_file()
            with LABELS.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["ts", "user", "roi_id", "label"])
                w.writerow([ts, user, roi_id, label])
            self.labels[(user, roi_id)] = {"label": label, "ts": ts}

    def user_labels(self, user: str) -> dict[str, str]:
        return {rid: v["label"] for (u, rid), v in self.labels.items() if u == user}

    def reset(self, target: str) -> int:
        """Drop all labels for ``target`` (or everyone if ``target == '*'``) and
        rewrite labels.csv. Returns how many label rows were removed."""
        with _LOCK:
            before = len(self.labels)
            if target == "*":
                self.labels = {}
            else:
                self.labels = {k: v for k, v in self.labels.items() if k[0] != target}
            removed = before - len(self.labels)
            with LABELS.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "user", "roi_id", "label"])
                for (u, rid), v in self.labels.items():
                    w.writerow([v.get("ts", ""), u, rid, v["label"]])
        return removed


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────
def _majority(votes: dict[str, int]):
    """Most-common class, or None on a tie for the top spot."""
    if not votes:
        return None
    top = max(votes.values())
    winners = [c for c, n in votes.items() if n == top]
    return winners[0] if len(winners) == 1 else None


def _entropy_norm(counts: list[int]) -> float:
    """Shannon entropy of a vote distribution, normalised to [0, 1]."""
    n = sum(counts)
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log(p)
    return h / math.log(len(CLASSES))          # max entropy = log(#classes)


def _confusion(pairs: list[tuple[str, str]]):
    """3x3 matrix of (row=truth, col=pred) over (truth,pred) label pairs."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = [[0, 0, 0] for _ in CLASSES]
    correct = 0
    for truth, pred in pairs:
        m[idx[truth]][idx[pred]] += 1
        if truth == pred:
            correct += 1
    acc = correct / len(pairs) if pairs else None
    return m, acc, len(pairs)


def compute_results(app: App) -> dict:
    ds = app.shown()                                       # only the curated set
    shown_ids = {d["id"] for d in ds}
    users = sorted({u for (u, rid) in app.labels if rid in shown_ids})

    # per-ROI vote tallies
    rois = []
    for d in ds:
        rid = d["id"]
        votes = {c: 0 for c in CLASSES}
        for (u, r), v in app.labels.items():
            if r == rid:
                votes[v["label"]] += 1
        n = sum(votes.values())
        counts = [votes[c] for c in CLASSES]
        cilia_frac = (votes["cilia"] / n) if n else None
        rois.append({
            "id": rid, "img": f"/api/img?id={quote(rid)}",
            "model_score": d["model_score"], "model_label": d["model_label"],
            "votes": votes, "n": n,
            "entropy": round(_entropy_norm(counts), 4),
            "human_cilia_frac": cilia_frac,
        })

    # consensus label per ROI (majority, None on tie)
    consensus = {}
    for r in rois:
        consensus[r["id"]] = _majority({c: r["votes"][c] for c in CLASSES}) if r["n"] else None

    # confusion matrices per user (vs model, vs consensus) + accuracies
    vs_model, vs_cons, per_user = {}, {}, []
    ulabels = {u: {rid: lab for rid, lab in app.user_labels(u).items()
                   if rid in shown_ids} for u in users}
    for u in users:
        ul = ulabels[u]
        model_pairs = [(app.by_id[rid]["model_label"], lab) for rid, lab in ul.items()]
        cons_pairs = [(consensus[rid], lab) for rid, lab in ul.items()
                      if consensus.get(rid) is not None]
        mm, m_acc, m_n = _confusion(model_pairs)
        cm, c_acc, c_n = _confusion(cons_pairs)
        vs_model[u] = {"matrix": mm, "acc": m_acc, "n": m_n}
        vs_cons[u] = {"matrix": cm, "acc": c_acc, "n": c_n}
        per_user.append({"user": u, "n_labeled": len(ul),
                         "acc_vs_model": m_acc, "acc_vs_consensus": c_acc})

    # user x user pairwise agreement (over commonly-labelled ROIs)
    agree = [[None] * len(users) for _ in users]
    pair_scores = []
    for i, ui in enumerate(users):
        agree[i][i] = 1.0
        for j in range(i + 1, len(users)):
            uj = users[j]
            common = set(ulabels[ui]) & set(ulabels[uj])
            if common:
                same = sum(ulabels[ui][r] == ulabels[uj][r] for r in common)
                frac = same / len(common)
                agree[i][j] = agree[j][i] = round(frac, 3)
                pair_scores.append(frac)
    mean_pair = round(sum(pair_scores) / len(pair_scores), 3) if pair_scores else None

    # Fleiss' kappa (variable raters per item)
    kappa = _fleiss(rois)

    # correlation: model P(cilia) vs human cilia-vote fraction
    pts = [{"id": r["id"], "x": r["model_score"], "y": r["human_cilia_frac"],
            "model_label": r["model_label"]}
           for r in rois if r["human_cilia_frac"] is not None]
    r_val = _pearson([p["x"] for p in pts], [p["y"] for p in pts])

    top = sorted([r for r in rois if r["n"] > 0],
                 key=lambda r: (-r["entropy"], r["id"]))

    return {
        "n_rois": len(ds), "raters": users, "n_raters": len(users),
        "fleiss_kappa": kappa, "mean_pairwise_agreement": mean_pair,
        "classes": list(CLASSES), "rois": rois, "top_disagree": top,
        "consensus": consensus, "per_user": per_user,
        "confusion": {"vs_model": vs_model, "vs_consensus": vs_cons},
        "agreement_matrix": {"users": users, "matrix": agree},
        "correlation": {"points": pts, "r": r_val,
                        "r2": (round(r_val ** 2, 4) if r_val is not None else None)},
    }


def _fleiss(rois: list[dict]):
    """Fleiss' kappa allowing a different number of raters per item."""
    items = [[r["votes"][c] for c in CLASSES] for r in rois if r["n"] >= 2]
    if len(items) < 2:
        return None
    # overall category proportions
    total = sum(sum(it) for it in items)
    if total == 0:
        return None
    p_j = [sum(it[k] for it in items) / total for k in range(len(CLASSES))]
    P_e = sum(p * p for p in p_j)
    # per-item agreement
    P_is = []
    for it in items:
        n = sum(it)
        if n < 2:
            continue
        P_is.append((sum(c * c for c in it) - n) / (n * (n - 1)))
    P_bar = sum(P_is) / len(P_is)
    if P_e >= 1.0:
        return None
    return round((P_bar - P_e) / (1 - P_e), 4)


def _pearson(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return round(sxy / math.sqrt(sxx * syy), 4)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    app: App = None                                        # set in main()

    def log_message(self, *a):                             # quieter console
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel: str):
        rel = rel.lstrip("/") or "index.html"
        fp = (STATIC / rel).resolve()
        if not str(fp).startswith(str(STATIC.resolve())) or not fp.is_file():
            return self._send(404, "not found", "text/plain")
        ctype = _CTYPES.get(fp.suffix, "application/octet-stream")
        self._send(200, fp.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        app = self.app
        if path == "/api/members":
            return self._send(200, app.members)
        if path == "/api/dataset":
            # only the admin-selected ROIs; ids + image url only (model score
            # withheld from labelers)
            return self._send(200, [{"id": d["id"], "img": f"/api/img?id={quote(d['id'])}"}
                                    for d in app.shown()])
        if path == "/api/pool":
            # admin-only: full 50-ROI pool with model score/label + selected flag
            who = (q.get("user") or [""])[0]
            if who != ADMIN:
                return self._send(403, {"error": "admins only"})
            sset = set(app.selected)
            return self._send(200, {"selected_count": len(app.selected), "pool": [
                {"id": d["id"], "img": f"/api/img?id={quote(d['id'])}",
                 "model_score": d["model_score"], "model_label": d["model_label"],
                 "selected": d["id"] in sset}
                for d in app.dataset]})
        if path == "/api/img":
            rid = (q.get("id") or [""])[0]
            d = app.by_id.get(rid)
            if not d:
                return self._send(404, "no roi", "text/plain")
            fp = (ROI_DIR / d["file"]).resolve()
            if not str(fp).startswith(str(ROI_DIR.resolve())) or not fp.is_file():
                return self._send(404, "no img", "text/plain")
            return self._send(200, fp.read_bytes(), "image/png",
                              {"Cache-Control": "no-store, must-revalidate"})
        if path == "/api/mine":
            user = (q.get("user") or [""])[0]
            if user not in app.member_names:
                return self._send(403, {"error": "unknown user"})
            return self._send(200, {"labels": app.user_labels(user)})
        if path == "/api/results":
            who = (q.get("user") or [""])[0]
            if who != ADMIN:
                return self._send(403, {"error": "stats are restricted"})
            return self._send(200, compute_results(app))
        # static
        if path.startswith("/api/"):
            return self._send(404, {"error": "no route"})
        return self._static(path)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        app = self.app

        if u.path == "/api/login":
            name = (body.get("name") or "").strip()
            if name not in app.member_names:
                return self._send(403, {"ok": False, "error": "not on the roster"})
            return self._send(200, {"ok": True, "user": name,
                                    "is_admin": name == ADMIN})

        if u.path == "/api/label":
            user = (body.get("user") or "").strip()
            rid = body.get("id")
            label = body.get("label")
            if user not in app.member_names:
                return self._send(403, {"ok": False, "error": "unknown user"})
            if rid not in app.by_id or label not in CLASSES:
                return self._send(400, {"ok": False, "error": "bad id/label"})
            app.record(user, rid, label)
            done = len(app.user_labels(user))
            return self._send(200, {"ok": True, "answered": done,
                                    "total": len(app.dataset)})

        if u.path == "/api/select":
            admin = (body.get("admin") or "").strip()
            ids = body.get("ids")
            if admin != ADMIN:
                return self._send(403, {"ok": False, "error": "admins only"})
            if not isinstance(ids, list) or not ids:
                return self._send(400, {"ok": False, "error": "pick at least one ROI"})
            sel = app.set_selection([str(i) for i in ids])
            return self._send(200, {"ok": True, "selected_count": len(sel)})

        if u.path == "/api/reset":
            admin = (body.get("admin") or "").strip()
            target = (body.get("target") or "").strip()   # a member name, or "*"
            if admin != ADMIN:
                return self._send(403, {"ok": False, "error": "admins only"})
            if target != "*" and target not in app.member_names:
                return self._send(400, {"ok": False, "error": "unknown target"})
            removed = app.reset(target)
            return self._send(200, {"ok": True, "removed": removed, "target": target})

        return self._send(404, {"error": "no route"})


def _lan_ip():
    """Best-guess LAN IP of this machine (no packets actually sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))               # picks the outbound interface
        ip = s.getsockname()[0]
    except Exception:
        ip = None
    finally:
        s.close()
    return None if ip in (None, "127.0.0.1") else ip


def main():
    ap = argparse.ArgumentParser(description="Cilia Consensus Study server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not DATASET.is_file():
        raise SystemExit("dataset.json missing — run: python cilia_study/build_dataset.py")

    Handler.app = App()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    lan = _lan_ip()
    print(f"[cilia-study] {len(Handler.app.dataset)} ROIs, "
          f"{len(Handler.app.members)} members")
    print(f"[cilia-study] this PC   : http://localhost:{args.port}")
    if lan:
        print(f"[cilia-study] phones   : http://{lan}:{args.port}  "
              f"(same wifi — share this)")
    else:
        print(f"[cilia-study] phones   : http://<this-PC-LAN-ip>:{args.port}  "
              f"(couldn't auto-detect IP — run `ipconfig`)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[cilia-study] bye")


if __name__ == "__main__":
    main()
