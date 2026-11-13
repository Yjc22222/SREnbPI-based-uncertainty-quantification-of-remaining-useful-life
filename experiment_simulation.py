"""Paired simulation. Reads no external files. All intervals use frozen calibration."""

import argparse
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

N, M, B, H = 10000, 200, 30, 16
REPS = tuple(range(701, 801))
LAMBDAS = (0.0, 0.1, 0.2, 0.5)


def scale(t, case):
    if case == 1:
        return 1 + 5 * t
    if case == 2:
        return np.abs(1 + 5 * np.sin(2 * np.pi * t))
    return np.where(t < 0.5, 1.0, 5.0)


def design(t, arch):
    offsets = np.arange(3, -1, -1) / 200 if arch == "RNN" else np.array([0.0])
    v = np.maximum(0, t[:, None] - offsets[None, :])
    return np.stack(
        [2 * v - 1, np.sin(2 * np.pi * v), np.cos(2 * np.pi * v)], axis=-1
    ).astype("f")


def signal(t):
    return 100 * np.sin(2 * np.pi * t)


class Panel(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        def param(shape, s):
            return nn.Parameter(torch.randn(shape) * s)

        p = B + 5
        self.a = param((p, 3, H), 0.3)
        self.b = param((p, 1, H), 0.1)
        self.o = param((p, H, 1), 0.25)
        self.c = param((p, 1, 1), 0.1)
        if arch == "RNN":
            self.r = param((p, H, H), 0.15)

    def forward(self, x, drop=False):
        h = torch.tanh(x[:, :, 0] @ self.a + self.b)
        for j in range(1, x.shape[2]):
            h = torch.tanh(x[:, :, j] @ self.a + h @ self.r + self.b)
        h = nn.functional.dropout(h, p=0.1, training=drop)
        return (h @ self.o + self.c).squeeze(-1)


def calc(arch, case, seed, training_y, test_y, base, target):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed * 100 + case * 10 + (arch == "RNN"))
    torch.manual_seed(int(rng.integers(2**31)))
    x = torch.tensor(design(np.linspace(0, 1, N), arch))
    tx = torch.tensor(design(np.linspace(0, 1, M), arch))
    while True:
        ix = rng.integers(N, size=(B, N))
        keep = np.stack([np.bincount(v, minlength=N) == 0 for v in ix], axis=1)
        if keep.sum(1).min() >= 2:
            break
    pool = np.vstack([ix, np.tile(np.arange(N), (5, 1))])
    ids = np.arange(B + 5)[:, None]
    m = Panel(arch)
    optim = torch.optim.Adam(m.parameters(), lr=0.008)
    y = torch.tensor(training_y / 100, dtype=torch.float32)
    for _ in range(220):
        sample = pool[ids, rng.integers(N, size=(B + 5, 128))]
        ind = torch.from_numpy(sample)
        loss = (m(x[ind], drop=True) - y[ind]).square().mean(1).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():

        def pred(xx, drop=False):
            return (
                m(xx[None].expand(B + 5, -1, -1, -1), drop).numpy().astype(float) * 100
            )

        train_pred = np.hstack([pred(t)[:B] for t in x.split(500)])
        test_pred = pred(tx)
        mc = np.stack([pred(tx, True)[B] for _ in range(30)])
    w = keep / keep.sum(1, keepdims=True)
    pred_oob = (w.T * train_pred).sum(0)
    sd = np.sqrt(np.maximum(0, (w.T * (train_pred - pred_oob) ** 2).sum(0))) + 1e-6
    err = np.abs(training_y - pred_oob)
    loo = w @ test_pred[:B]
    sd_test = (np.sqrt(np.maximum(0, w @ (test_pred[:B] ** 2) - loo**2)) + 1e-6).mean(0)
    center = np.quantile(loo, 0.95, axis=0, method="inverted_cdf")
    h = np.quantile(err, 0.95, method="inverted_cdf")
    hs = np.quantile(err / sd, 0.95, method="inverted_cdf") * sd_test
    values = {
        "EnbPI": (center, np.full(M, h)),
        "SREnbPI": (center, hs),
        "MC": (mc.mean(0), 1.96 * mc.std(0)),
        "DE": (test_pred[B:].mean(0), 1.96 * test_pred[B:].std(0)),
    }
    rows = []
    for method, (mid, half) in values.items():
        lo, hi = mid - half, mid + half
        common = np.maximum(
            0, np.minimum(hi, base + target) - np.maximum(lo, base - target)
        )
        denominator = 4 * half * target
        fit = np.divide(
            common**2, denominator, out=np.zeros_like(target), where=denominator > 0
        )
        row = dict(
            arch=arch,
            case=case,
            seed=seed,
            method=method,
            picp=np.mean((lo <= test_y) & (test_y <= hi)) * 100,
            mpiw=np.mean(2 * half),
        )
        for lam in LAMBDAS:
            row[f"pifr_{lam}"] = np.mean(fit / (1 + lam * np.abs(mid - base)))
        if method in ("EnbPI", "SREnbPI"):
            factors = np.ones(M) if method == "EnbPI" else sd_test
            history = err.copy() if method == "EnbPI" else (err / sd).copy()
            seen = np.abs(test_y - mid) / factors
            moving = np.zeros(M)
            for begin in range(0, M, 5):
                stop = min(begin + 5, M)
                coeff = np.quantile(history, 0.95, method="inverted_cdf")
                moving[begin:stop] = coeff * factors[begin:stop]
                history = np.concatenate((history[stop - begin :], seen[begin:stop]))
            row["feedback5_picp"] = 100 * np.mean(np.abs(test_y - mid) <= moving)
            row["feedback5_mpiw"] = 2 * moving.mean()
        for key, idx in [
            ("first", np.arange(M) < M // 2),
            ("second", np.arange(M) >= M // 2),
        ]:
            row[key + "_picp"] = 100 * np.mean(
                (lo[idx] <= test_y[idx]) & (test_y[idx] <= hi[idx])
            )
            row[key + "_mpiw"] = 2 * np.mean(half[idx])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    a = parser.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"simulation_{a.seed}.csv"
    if path.exists():
        return
    t = np.linspace(0, 1, N)
    u = np.linspace(0, 1, M)
    rows = []
    for case in (1, 2, 3):
        rng = np.random.default_rng(a.seed * 10 + case)
        y = signal(t) + scale(t, case) * rng.normal(size=N)
        yy = signal(u) + scale(u, case) * rng.normal(size=M)
        for arch in ("MLP", "RNN"):
            rows.extend(
                calc(
                    arch, case, a.seed, y, yy, signal(u), 1.95996398454 * scale(u, case)
                )
            )
    with tempfile.NamedTemporaryFile(dir=a.out, delete=False) as fh:
        tmpname = fh.name
    try:
        pd.DataFrame(rows).to_csv(tmpname, index=False)
        os.replace(tmpname, path)
    finally:
        Path(tmpname).unlink(missing_ok=True)
    print("done", a.seed, flush=True)


if __name__ == "__main__":
    main()
