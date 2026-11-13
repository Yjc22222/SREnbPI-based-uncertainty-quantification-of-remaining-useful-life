"""Recompute the numerical tables and figures from the public data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cmapss",
        type=Path,
        required=True,
        help="Folder containing the three extracted FD001 text files.",
    )
    p.add_argument(
        "--bearing-features", type=Path, default=ROOT / "bearing_features.csv"
    )
    p.add_argument("--output", type=Path, default=ROOT / "rerun")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent CPU jobs to run concurrently.",
    )
    a = p.parse_args()
    if a.jobs < 1:
        p.error("--jobs must be positive")
    fd = a.cmapss.resolve()
    bf = a.bearing_features.resolve()
    out = a.output.resolve()
    for required in [
        bf,
        *[fd / f"{stem}_FD001.txt" for stem in ("train", "test", "RUL")],
    ]:
        if not required.is_file():
            p.error(f"File missing: {required}")
    raw = out / "results"
    summary = out / "summary"
    fig = out / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )

    def run(args):
        subprocess.run(
            [sys.executable, str(ROOT / args[0]), *map(str, args[1:])],
            check=True,
            env=env,
        )

    jobs = [
        ["experiment_simulation.py", "--seed", s, "--out", raw] for s in range(701, 801)
    ]
    for s in (101, 202, 303, 404, 505):
        jobs.append(
            [
                "experiment_devices.py",
                "fd",
                "--data",
                fd,
                "--seed",
                s,
                "--epochs",
                20,
                "--out",
                raw,
            ]
        )
        for fold in range(1, 6):
            jobs.append(
                [
                    "experiment_devices.py",
                    "bearing",
                    "--data",
                    bf,
                    "--seed",
                    s,
                    "--fold",
                    fold,
                    "--epochs",
                    24,
                    "--out",
                    raw,
                ]
            )
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run, jobs))
    run(["summarize.py", raw, summary])
    run(["figures.py", raw, fig])


if __name__ == "__main__":
    main()
