"""Extract 8 statistics per axis plus elapsed minutes from the 5 condition-1 bearings.
Accept the original ZIP or a directory containing Bearing1_i/N.csv files.
"""

import sys, zipfile, re
from pathlib import Path
import numpy as np
import pandas as pd


def features(a):
    mean = a.mean(axis=0)
    v = a - mean
    sd = np.sqrt((v * v).mean(axis=0))
    z = v / np.maximum(sd, 1e-12)
    return np.stack(
        [
            mean,
            sd,
            np.sqrt((a * a).mean(0)),
            a.max(0),
            a.min(0),
            np.ptp(a, axis=0),
            (z**3).mean(0),
            (z**4).mean(0) - 3,
        ],
        axis=0,
    ).T.ravel()


def run(source, out):
    arc = zipfile.ZipFile(source) if source.is_file() else None
    names = arc.namelist() if arc else [str(p) for p in source.rglob("*.csv")]
    pattern = re.compile(r"Bearing1_([1-5])/(\d+)\.csv$")
    data = [
        (int(m[1]), int(m[2]), name)
        for name in names
        if (m := pattern.search(name.replace("\\", "/")))
    ]
    data.sort()
    names0 = ["mean", "sd", "rms", "max", "min", "ptp", "skew", "exkurt"]
    rows = []
    for g, count in enumerate([123, 161, 158, 122, 52], 1):
        block = [r for r in data if r[0] == g]
        assert [r[1] for r in block] == list(range(1, count + 1)), (g, len(block))
        for _, minute, name in block:
            with arc.open(name) if arc else open(name, "rb") as f:
                a = pd.read_csv(f).to_numpy(dtype=float)
            assert a.shape == (32768, 2) and np.isfinite(a).all(), name
            rows.append([g, *features(a), minute, count - minute])
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            *[k + "_" + n for k in ("H", "V") for n in names0],
            "elapsed",
            "rul",
        ],
    )
    df.to_csv(out, index=False)
    if arc:
        arc.close()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
