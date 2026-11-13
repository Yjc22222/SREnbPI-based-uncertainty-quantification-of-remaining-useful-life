"""Engine and bearing experiments. Run from the directory containing this script.
Inputs: CMAPSSData/{train,test,RUL}_FD001.txt and bearing_features.csv.
The publication uses independently initialized networks, device-level bootstraps and no test-label feedback.
"""

import argparse
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

WINDOW = 50
BOOT = 30
BATCH = 64
KEEP = [0, 1, 2] + [2 + i for i in range(1, 22) if i not in (1, 5, 6, 10, 16, 18, 19)]


def read_fd(folder):
    def data(split):
        arr = np.loadtxt(folder / f"{split}_FD001.txt")
        end = np.loadtxt(folder / "RUL_FD001.txt") if split == "test" else None
        result = {}
        for uid in np.unique(arr[:, 0]).astype(int):
            rows = arr[arr[:, 0] == uid]
            y = np.minimum(
                130, rows[-1, 1] - rows[:, 1] + (end[uid - 1] if end is not None else 0)
            )
            result[uid] = (rows[:, 2:][:, KEEP].astype("f"), y.astype("f"))
        return result

    return data("train"), data("test")


def read_bearings(path):
    df = pd.read_csv(path)
    return {
        int(k): (
            v.drop(columns=["id", "rul"]).to_numpy(dtype="f"),
            v.rul.to_numpy(dtype="f"),
        )
        for k, v in df.groupby("id")
    }


def indices(n, limit):
    if limit is None or n <= limit:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).round().astype(int)


def windows(v, positions, scaler):
    x, y = v
    x = scaler.transform(x)
    ix = np.maximum(0, positions[:, None] + np.arange(1 - WINDOW, 1))
    return x[ix].astype("f"), y[positions]


class CNNLSTM(nn.Module):
    def __init__(self, p):
        super().__init__()
        channels = [p, 18, 36, 72]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.ConstantPad1d((0, 1), 0),
                    nn.Conv1d(a, b, 2),
                    nn.ReLU(),
                    nn.MaxPool1d(2, ceil_mode=True),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.r1 = nn.LSTM(84, 3 * p, batch_first=True)
        self.r2 = nn.LSTM(3 * p, 3 * p, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Sequential(
            nn.Linear(3 * p, 50), nn.ReLU(), nn.Dropout(0.2), nn.Linear(50, 1)
        )

    def forward(self, x):
        z = (
            self.conv(x.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
            .reshape(len(x), 6, 84)
        )
        z = self.drop(self.r1(z)[0])
        z = self.drop(self.r2(z)[0][:, -1])
        return self.out(z).squeeze(-1)


def predict(net, x, drop=False):
    net.train(drop)
    with torch.no_grad():
        return torch.cat([net(t) for t in x.split(256)]).numpy().astype(float) * 100


def train(x, y, idx, seed, epochs):
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    net = CNNLSTM(x.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    net.train()
    for _ in range(epochs):
        for batch in np.array_split(
            rng.permutation(idx), int(np.ceil(len(idx) / BATCH))
        ):
            ix = torch.from_numpy(batch)
            opt.zero_grad()
            loss = (net(x[ix]) - y[ix]).square().mean()
            loss.backward()
            opt.step()
    return net


def bag_design(num, rng):
    while True:
        draws = rng.integers(num, size=(BOOT, num))
        keep = np.stack([~np.isin(np.arange(num), d) for d in draws], axis=1)
        if keep.sum(1).min() >= 2:
            return draws, keep


def fit_protocol(training, evaluation, out, run, kind, fold, epochs, limit):
    path = out / f"{kind}_{fold}_{run}.csv.gz"
    if path.exists():
        return
    rng = np.random.default_rng(run + 1000 * fold)
    ids = list(sorted(training))
    G = len(ids)
    torch.set_num_threads(1)
    scaler = MinMaxScaler().fit(np.vstack([training[g][0] for g in ids]))
    trainx = []
    trainy = []
    groups = []
    for j, g in enumerate(ids):
        v = training[g]
        ix = indices(len(v[1]), limit)
        xx, yy = windows(v, ix, scaler)
        trainx.extend(xx)
        trainy.extend(yy)
        groups.extend([j] * len(yy))
    x = torch.from_numpy(np.array(trainx, dtype="f"))
    truth = np.asarray(trainy)
    y = torch.from_numpy(truth / 100)
    groups = np.asarray(groups)
    gx = [np.flatnonzero(groups == j) for j in range(G)]
    xx = []
    rr = []
    for subset, g, v, pos in evaluation:
        wx, wy = windows(v, pos, scaler)
        xx.extend(wx)
        rr.extend(zip([subset] * len(pos), [g] * len(pos), pos + 1, wy))
    tx = torch.from_numpy(np.array(xx, dtype="f"))
    both = torch.cat([x, tx])
    N = len(x)
    draws, keep = bag_design(G, rng)
    predictions = []
    for b, row in enumerate(draws):
        ix = np.concatenate([gx[j] for j in row])
        net = train(x, y, ix, int(rng.integers(2**31)), epochs)
        predictions.append(predict(net, both))
    # A row of w aggregates only the networks that omitted that device.
    predictions = np.stack(predictions)
    w = keep / keep.sum(axis=1, keepdims=True)
    centers = w @ predictions
    sd = np.sqrt(np.maximum(0, w @ (predictions**2) - centers**2)) + 1e-6
    ix = np.arange(N)
    err = abs(truth - centers[groups, ix])
    scaled = err / sd[groups, ix]
    center = np.quantile(centers[:, N:], 0.95, axis=0, method="inverted_cdf")
    st = sd[:, N:].mean(0)
    q = np.quantile(err, 0.95, method="inverted_cdf")
    qs = np.quantile(scaled, 0.95, method="inverted_cdf")
    # Labels are retained below for scoring. They never enter the interval calculation.
    rec = pd.DataFrame(rr, columns=["subset", "id", "elapsed", "y"])

    def save(method, m, wd):
        rec[method + "_mid"] = m
        rec[method + "_half"] = np.broadcast_to(wd, m.shape)

    save("EnbPI", center, q)
    save("SREnbPI", center, qs * st)
    allrows = np.arange(N)
    de = []
    for i in range(5):
        net = train(x, y, allrows, int(rng.integers(2**31)), epochs)
        de.append(predict(net, tx))
        if i == 0:
            mc = np.stack([predict(net, tx, True) for _ in range(30)])
            save("MC", mc.mean(0), 1.96 * mc.std(0))
    de = np.stack(de)
    save("DE", de.mean(0), 1.96 * de.std(0))
    rec["seed"] = run
    rec["fold"] = fold
    with tempfile.NamedTemporaryFile(dir=out, delete=False) as fh:
        tmpname = fh.name
    try:
        rec.to_csv(tmpname, index=False, compression="gzip")
        os.replace(tmpname, path)
    finally:
        Path(tmpname).unlink(missing_ok=True)
    print("done", kind, fold, run, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["fd", "bearing"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.epochs is None:
        a.epochs = 20 if a.kind == "fd" else 24
    a.out.mkdir(parents=True, exist_ok=True)
    if a.kind == "fd":
        d, test = read_fd(a.data)
        tr = {g: v for g, v in d.items() if g % 5}
        ev = [
            ("trajectory", g, v, np.arange(len(v[1])))
            for g, v in d.items()
            if g % 5 == 0
        ]
        ev += [("endpoint", g, v, np.array([len(v[1]) - 1])) for g, v in test.items()]
        fit_protocol(tr, ev, a.out, a.seed, "fd", 0, a.epochs, 16)
    else:
        d = read_bearings(a.data)
        tr = {g: v for g, v in d.items() if g != a.fold}
        v = d[a.fold]
        fit_protocol(
            tr,
            [("bearing", a.fold, v, np.arange(len(v[1])))],
            a.out,
            a.seed,
            "bearing",
            a.fold,
            a.epochs,
            32,
        )


if __name__ == "__main__":
    main()
