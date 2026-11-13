"""Recompute the paper's descriptive results and paired engine bootstrap intervals."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

METHODS = ("SREnbPI", "EnbPI", "MC", "DE")


def average(df, keys, cols):
    return df.groupby(keys)[cols].agg(["mean", "std"]).round(8)


def engine_ci(metrics):
    rng = np.random.default_rng(2901)
    rows = []
    for ss in ["trajectory", "endpoint"]:
        d = (
            metrics[metrics.subset == ss]
            .groupby(["id", "method"])[["picp", "mpiw"]]
            .mean()
        )
        dif = d.xs("SREnbPI", level="method") - d.xs("EnbPI", level="method")
        ix = rng.integers(len(dif), size=(10000, len(dif)))
        for col in dif:
            q = np.quantile(dif[col].to_numpy()[ix].mean(1), [0.025, 0.975])
            rows.append([ss, col, dif[col].mean(), *q])
    return pd.DataFrame(rows, columns=["subset", "metric", "delta", "low", "high"])


def run(base, out, require=True):
    out.mkdir(parents=True, exist_ok=True)
    sim = pd.concat(pd.read_csv(p) for p in sorted(base.glob("simulation_*.csv")))
    if require:
        keys = ["arch", "case", "seed", "method"]
        expected = pd.MultiIndex.from_product(
            [("MLP", "RNN"), (1, 2, 3), range(701, 801), METHODS], names=keys
        )
        observed = pd.MultiIndex.from_frame(sim[keys])
        if (
            len(sim) != len(expected)
            or observed.has_duplicates
            or not observed.difference(expected).empty
            or not expected.difference(observed).empty
        ):
            raise ValueError(
                "Expected exactly 2400 unique simulation rows: "
                "100 seeds (701-800) x 3 cases x 2 architectures x 4 methods."
            )
    sim.to_csv(out / "simulation_runs.csv", index=False)
    avg_sim = average(
        sim,
        ["arch", "case", "method"],
        [
            x
            for x in sim
            if x.startswith(("pifr", "mpiw", "picp", "first_", "second_", "feedback5_"))
        ],
    )
    avg_sim.to_csv(out / "simulation_summary.csv")
    files = list(base.glob("fd_*.csv.gz")) + list(base.glob("bearing_*.csv.gz"))
    if require:
        assert len(files) == 30
    dt = pd.concat(pd.read_csv(p) for p in files)
    parts = []
    for m in METHODS:
        d = dt[["subset", "id", "elapsed", "fold", "seed", "y"]].copy()
        d["method"] = m
        d["picp"] = 100 * (abs(d.y - dt[m + "_mid"]) <= dt[m + "_half"]).astype(float)
        d["mpiw"] = 2 * dt[m + "_half"]
        parts.append(d)
    long = pd.concat(parts)
    dev = (
        long.groupby(["subset", "id", "seed", "method"])[["picp", "mpiw"]]
        .mean()
        .reset_index()
    )
    byseed = (
        dev.groupby(["subset", "seed", "method"])[["picp", "mpiw"]].mean().reset_index()
    )
    byseed.to_csv(out / "device_macro_by_seed.csv", index=False)
    dev.to_csv(out / "device_metrics.csv", index=False)
    average(byseed, ["subset", "method"], ["picp", "mpiw"]).to_csv(
        out / "device_summary.csv"
    )
    average(dev[dev.subset == "bearing"], ["id", "method"], ["picp", "mpiw"]).to_csv(
        out / "bearing_by_id.csv"
    )
    ci = engine_ci(dev)
    ci.to_csv(out / "fd_paired_bootstrap.csv", index=False)
    late = (
        long[(long.subset == "trajectory") & (long.y <= 20)]
        .groupby(["id", "seed", "method"])[["picp", "mpiw"]]
        .mean()
        .groupby(["seed", "method"])
        .mean()
        .reset_index()
    )
    average(late, ["method"], ["picp", "mpiw"]).to_csv(out / "fd_last21.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.results, a.out)
