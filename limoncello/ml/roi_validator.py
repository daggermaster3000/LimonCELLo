"""
ROI cilia validator 🍋🤖
=========================

A convolutional neural network that learns, from the human screening decisions,
to tell a *real* cilium ROI apart from junk. It assists (not replaces) the manual
review in the data app's Screening tab:

    1. You keep / reject some cilia              → labelled ROI images.
    2. ``train_validator`` fits a CNN on them    → on the GPU if available.
    3. ``predict_proba`` scores the rest         → suggest keep/reject + surface
                                                   the most uncertain for review.

Two architectures are available:
  * ``"tiny"`` – a ~31k-param net (64²), trains in seconds on a CPU.
  * ``"big"``  – a ~1.2M-param BatchNorm CNN (96²) for a CUDA GPU; with proper
                 mini-batching, augmentation and early stopping it makes good use
                 of a card like an RTX 4090 without overfitting the small set.

Bundles record their architecture + input size, so old ``tiny`` models still
load. Torch is imported lazily by the data app, so importing this module is only
done on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

CLASS_NAMES = ("reject", "keep")          # index 0 = junk, 1 = real cilium
DEFAULT_SIZE = {"tiny": 64, "big": 96}    # input resolution per architecture
IMG_SIZE = DEFAULT_SIZE["tiny"]           # back-compat default


_CUDNN_HANDLED = False


def get_device() -> torch.device:
    """CUDA when available (e.g. an RTX 4090), else CPU.

    Some installs ship a broken cuDNN (the ``cudnn_*`` DLL fails to load and
    aborts the process, which can't be caught). We disable cuDNN the first time
    CUDA is requested so PyTorch uses its built-in CUDA kernels instead — still
    full GPU acceleration, only without cuDNN's extra conv autotuning.
    """
    global _CUDNN_HANDLED
    if torch.cuda.is_available():
        if not _CUDNN_HANDLED:
            try:
                torch.backends.cudnn.enabled = False
            except Exception:                             # noqa: BLE001
                pass
            _CUDNN_HANDLED = True
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Image IO
# ─────────────────────────────────────────────────────────────────────────────
def load_roi_image(path: str, size: int = IMG_SIZE) -> np.ndarray | None:
    """Load an ROI PNG as a ``(C, H, W)`` float array in ``[0, 1]``, or None."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size))
    except Exception:                                     # noqa: BLE001
        return None
    arr = np.asarray(img, dtype=np.float32) / 255.0       # H, W, C
    return np.transpose(arr, (2, 0, 1))                   # C, H, W


def _stack_images(paths: Sequence[str], size: int) -> tuple[np.ndarray, list[int]]:
    """Load many images; return the stacked array + indices that loaded OK."""
    imgs, ok = [], []
    for i, p in enumerate(paths):
        a = load_roi_image(p, size)
        if a is not None:
            imgs.append(a)
            ok.append(i)
    if not imgs:
        return np.empty((0, 3, size, size), dtype=np.float32), []
    return np.stack(imgs), ok


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class TinyROINet(nn.Module):
    """A small 3-block CNN → 2-class head. ~31k parameters (CPU-friendly)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(32 * 4 * 4, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


class ROINetBig(nn.Module):
    """A 4-block BatchNorm CNN with global pooling → ~1.2M params (GPU)."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


def make_model(arch: str = "big") -> nn.Module:
    return ROINetBig() if arch == "big" else TinyROINet()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def _augment(xb: torch.Tensor) -> torch.Tensor:
    """Label-preserving augmentation for square microscopy crops: random
    horizontal/vertical flips + a random 90° rotation."""
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[3])
    if torch.rand(1).item() < 0.5:
        xb = torch.flip(xb, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        xb = torch.rot90(xb, k, dims=[2, 3])
    return xb


def train_validator(
    samples: Sequence[tuple[str, int]],
    *,
    arch: str = "big",
    img_size: int | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.25,
    patience: int = 8,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Train the validator on ``(image_path, label)`` pairs (1 = keep, 0 = reject).

    Uses the GPU when available, mini-batch SGD with on-the-fly augmentation, and
    early stopping on the validation loss (keeps the best checkpoint). Returns the
    fitted ``model`` plus metrics. Raises ``ValueError`` on too little data.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    size = int(img_size or DEFAULT_SIZE.get(arch, 64))

    paths = [p for p, _ in samples]
    labels = np.array([int(y) for _, y in samples], dtype=np.int64)
    X, ok = _stack_images(paths, size)
    if len(ok) < 8:
        raise ValueError(
            f"Need at least 8 ROI images to train (got {len(ok)} with an image).")
    y = labels[ok]
    if len(np.unique(y)) < 2:
        raise ValueError("Need both kept and rejected examples to train.")

    # Train / val split (ensure both classes present in train).
    idx = rng.permutation(len(y))
    n_val = max(2, int(round(val_frac * len(y))))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if len(np.unique(y[tr_idx])) < 2:
        tr_idx, val_idx = idx, idx[:n_val]

    Xt = torch.from_numpy(X[tr_idx]); yt = torch.from_numpy(y[tr_idx])
    Xv = torch.from_numpy(X[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)

    # Class weights counter the usual keep/reject imbalance.
    counts = np.bincount(y[tr_idx], minlength=2).astype(np.float32)
    w = torch.from_numpy((counts.sum() / np.maximum(counts, 1)) / 2.0).to(device)

    model = make_model(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    bs = min(batch_size, len(tr_idx))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=max(2, bs), shuffle=True)

    history = []
    best_val, best_state, since_best = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in loader:
            xb = _augment(xb.to(device, non_blocking=True))
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        train_loss = run_loss / max(n_seen, 1)

        # Validation.
        model.eval()
        with torch.no_grad():
            vlogits = model(Xv)
            vloss = float(loss_fn(vlogits, yv).item())
            vp = torch.softmax(vlogits, dim=1)[:, 1].float().cpu().numpy()
        vacc = float(((vp >= 0.5).astype(int) == yv.cpu().numpy()).mean())
        history.append({"epoch": ep + 1, "train_loss": train_loss,
                        "val_loss": vloss, "val_acc": vacc})
        if progress:
            progress((ep + 1) / epochs,
                     f"epoch {ep + 1}/{epochs} · val acc {vacc:.2f} ({device.type})")

        # Early stopping on val loss; keep the best checkpoint.
        if vloss < best_val - 1e-4:
            best_val, since_best = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics from the best model.
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Xv), dim=1)[:, 1].float().cpu().numpy()
    pred = (pv >= 0.5).astype(int)
    yv_np = yv.cpu().numpy()
    val_acc = float((pred == yv_np).mean())
    tp = int(((pred == 1) & (yv_np == 1)).sum())
    tn = int(((pred == 0) & (yv_np == 0)).sum())
    fp = int(((pred == 1) & (yv_np == 0)).sum())
    fn = int(((pred == 0) & (yv_np == 1)).sum())

    return {
        "model": model,
        "arch": arch,
        "size": size,
        "device": device.type,
        "history": history,
        "val_acc": val_acc,
        "val_auc": _safe_auc(yv_np, pv),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n_total": len(y),
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
        "n_missing": len(samples) - len(ok),
        "class_counts": {"reject": int((y == 0).sum()), "keep": int((y == 1).sum())},
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC without a hard sklearn dependency; NaN if undefined."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_proba(model: nn.Module, paths: Sequence[str],
                  size: int = IMG_SIZE, batch_size: int = 256) -> np.ndarray:
    """Return P(real cilium) per path; ``NaN`` where the image is missing."""
    out = np.full(len(paths), np.nan, dtype=np.float32)
    X, ok = _stack_images(paths, size)
    if not ok:
        return out
    device = get_device()
    model = model.to(device).eval()
    use_amp = device.type == "cuda"
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                p = torch.softmax(model(xb), dim=1)[:, 1]
            preds.append(p.float().cpu().numpy())
    out[ok] = np.concatenate(preds)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Introspection — activation maps & Grad-CAM (what the net "looks at")
# ─────────────────────────────────────────────────────────────────────────────
def _conv_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every ``Conv2d`` in execution order, named conv1, conv2, … for display."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return [(f"conv{i + 1}", m) for i, m in enumerate(convs)]


def collect_activations(model: nn.Module, image_path: str,
                        size: int = IMG_SIZE) -> dict | None:
    """Forward one ROI image and capture each conv layer's output feature maps.

    Returns ``{"input": (H,W,3) float, "prob": P(keep), "layers": [
    {"name", "act": (C,H,W) float, "n_channels"}…]}`` or None if the image is
    unreadable. Run on the device the model lives on; no grad needed.
    """
    arr = load_roi_image(image_path, size)
    if arr is None:
        return None
    device = get_device()
    model = model.to(device).eval()
    x = torch.from_numpy(arr[None]).to(device)

    acts: list[dict] = []
    handles = []

    def _mk(name):
        def _hook(_m, _inp, out):
            a = out.detach().float().cpu().numpy()[0]      # (C, H, W)
            acts.append({"name": name, "act": a, "n_channels": int(a.shape[0])})
        return _hook

    for name, m in _conv_layers(model):
        handles.append(m.register_forward_hook(_mk(name)))
    try:
        with torch.no_grad():
            logits = model(x)
            prob = float(torch.softmax(logits, dim=1)[0, 1].item())
    finally:
        for h in handles:
            h.remove()
    return {"input": np.transpose(arr, (1, 2, 0)), "prob": prob, "layers": acts}


def grad_cam(model: nn.Module, image_path: str, size: int = IMG_SIZE,
             target_class: int = 1) -> dict | None:
    """Grad-CAM heatmap on the *last* conv layer (the class-discriminative map).

    Returns ``{"input": (H,W,3), "cam": (H,W) in [0,1], "prob": P(keep),
    "target": class_name}`` or None. Highlights the image regions that most
    drove the model's decision for ``target_class`` (1 = keep, 0 = reject).
    """
    arr = load_roi_image(image_path, size)
    if arr is None:
        return None
    device = get_device()
    model = model.to(device).eval()
    x = torch.from_numpy(arr[None]).to(device)

    convs = _conv_layers(model)
    if not convs:
        return None
    target = convs[-1][1]
    store: dict = {}

    def _fwd(_m, _inp, out):
        store["act"] = out

    def _bwd(_m, _gin, gout):
        store["grad"] = gout[0].detach()

    h1 = target.register_forward_hook(_fwd)
    h2 = target.register_full_backward_hook(_bwd)
    try:
        logits = model(x)                                  # grad enabled
        prob = float(torch.softmax(logits, dim=1)[0, 1].item())
        model.zero_grad(set_to_none=True)
        logits[0, int(target_class)].backward()
        act = store["act"][0]                              # (C, H, W)
        grad = store["grad"][0]                            # (C, H, W)
        weights = grad.mean(dim=(1, 2))                    # (C,) GAP of gradients
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam_np = cam.detach().float().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    return {"input": np.transpose(arr, (1, 2, 0)), "cam": cam_np, "prob": prob,
            "target": CLASS_NAMES[int(target_class)]}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_bundle(path: str, model: nn.Module, meta: dict) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": meta.get("arch", "big"),
        "size": meta.get("size", DEFAULT_SIZE.get(meta.get("arch", "big"), 64)),
        "meta": {k: v for k, v in meta.items() if k != "model"},
    }
    torch.save(payload, path)
    return path


def load_bundle(path: str) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload.get("arch", "tiny")            # old bundles were the tiny net
    model = make_model(arch)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = payload.get("meta", {})
    meta["arch"] = arch
    meta.setdefault("size", payload.get("size", DEFAULT_SIZE.get(arch, 64)))
    return model, meta
