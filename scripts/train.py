"""Train SideGRU on the shards written by ``collect_data.py``.

Class-weighted cross-entropy, AdamW (lr 1e-3, weight decay 1e-4), batches of
128 sequences, gradient clipping at 5, a fixed 50 epochs.  The validation set
is not used to stop training.  Retraining reproduces the released model up to
run-to-run training noise.

    python scripts/train.py --out checkpoints/sidegru_retrained.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from predictfix.model import Predictor, SideGRU
from common import ROOT


def load_shards(folder):
    paths = sorted(Path(folder).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no shards in {folder}; run scripts/collect_data.py first")
    xs, ys = [], []
    for path in paths:
        with np.load(path) as shard:
            xs.append(shard["features"])
            ys.append(shard["labels"].astype(np.int64))
    print(f"{len(paths)} shards")
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=ROOT / "data" / "train")
    ap.add_argument("--out", default=ROOT / "checkpoints" / "sidegru_retrained.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x, y = load_shards(args.data)
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    x -= mean
    x /= std
    counts = np.bincount(y.reshape(-1), minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(1.0, 4.0 * counts)
    print(f"{len(x)} sequences of {x.shape[1]} steps; side counts {counts.astype(int).tolist()}")

    torch.manual_seed(args.seed)
    net = SideGRU(x.shape[-1]).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=args.device))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)

    start = time.time()
    net.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(yt))
        total = 0.0
        for b in range(0, len(yt), 128):
            idx = order[b:b + 128]
            xb, yb = xt[idx].to(args.device), yt[idx].to(args.device)
            opt.zero_grad()
            loss = loss_fn(net(xb).reshape(-1, 4), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1:2d}  loss {total / len(yt):.4f}  "
              f"{time.time() - start:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Predictor(net.cpu(), mean, std).save(args.out, training={"epochs": args.epochs,
                                                             "torch_seed": args.seed})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
