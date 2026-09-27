#!/usr/bin/env python3

import csv
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT = Path("/home01/a2149a01/RFdiffusion3")

PLUS_ROOT = PROJECT / "outputs/binderB_af3/with_rap"
MINUS_ROOT = PROJECT / "outputs/binderB_af3/without_rap"

OUT_CSV = PROJECT / "outputs/binderB_af3_ranking.csv"
TOP_DIR = PROJECT / "outputs/binderB_af3_top10"


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def pair_iptm(summary):
    matrix = summary.get("chain_pair_iptm", [])

    try:
        # A = Binder A, B = Binder B
        return float(matrix[0][1])
    except Exception:
        raise RuntimeError(
            "Could not read A-B pair ipTM from "
            f"{summary}"
        )


def read_candidate(root, dirname):
    cdir = root / dirname

    values = []
    sample_cifs = []

    for sample in range(5):
        sdir = cdir / f"seed-1_sample-{sample}"

        summaries = list(
            sdir.glob("*_summary_confidences.json")
        )
        cifs = list(
            sdir.glob("*_model.cif")
        )

        if len(summaries) != 1:
            raise RuntimeError(
                f"{dirname} sample {sample}: "
                f"summary files={len(summaries)}"
            )

        if len(cifs) != 1:
            raise RuntimeError(
                f"{dirname} sample {sample}: "
                f"CIF files={len(cifs)}"
            )

        summary = load_summary(summaries[0])
        values.append(pair_iptm(summary))
        sample_cifs.append(cifs[0])

    return values, sample_cifs


def main():
    plus_dirs = sorted(
        p for p in PLUS_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_plus_RAP")
    )

    if len(plus_dirs) != 200:
        raise RuntimeError(
            f"Expected 200 +RAP candidates, "
            f"found {len(plus_dirs)}"
        )

    rows = []

    for n, plus_dir in enumerate(plus_dirs, 1):
        candidate = plus_dir.name.removesuffix("_plus_RAP")

        minus_name = candidate + "_no_RAP"
        minus_dir = MINUS_ROOT / minus_name

        if not minus_dir.is_dir():
            raise RuntimeError(
                f"Missing -RAP result: {minus_name}"
            )

        plus_vals, plus_cifs = read_candidate(
            PLUS_ROOT,
            plus_dir.name,
        )

        minus_vals, _ = read_candidate(
            MINUS_ROOT,
            minus_name,
        )

        plus_mean = float(np.mean(plus_vals))
        minus_mean = float(np.mean(minus_vals))

        delta = plus_mean - minus_mean

        best_sample = int(np.argmax(plus_vals))

        rows.append({
            "candidate": candidate,

            "plus_mean_A_B_iptm": plus_mean,
            "minus_mean_A_B_iptm": minus_mean,
            "delta_A_B_iptm": delta,

            "plus_min_A_B_iptm": min(plus_vals),
            "plus_max_A_B_iptm": max(plus_vals),

            "minus_min_A_B_iptm": min(minus_vals),
            "minus_max_A_B_iptm": max(minus_vals),

            "plus_samples": ";".join(
                f"{x:.3f}" for x in plus_vals
            ),
            "minus_samples": ";".join(
                f"{x:.3f}" for x in minus_vals
            ),

            "best_plus_sample": best_sample,
            "best_plus_sample_iptm": plus_vals[best_sample],

            "_best_cif": str(plus_cifs[best_sample]),
        })

        if n % 20 == 0:
            print(f"[{n}/200]")

    # Rank using both absolute +RAP confidence and RAP dependence.
    plus = np.array([r["plus_mean_A_B_iptm"] for r in rows])
    delta = np.array([r["delta_A_B_iptm"] for r in rows])

    plus_norm = (plus - plus.min()) / (plus.max() - plus.min())
    delta_norm = (delta - delta.min()) / (delta.max() - delta.min())

    for row, pn, dn in zip(rows, plus_norm, delta_norm):
        row["plus_norm"] = float(pn)
        row["delta_norm"] = float(dn)
        row["score"] = 0.60 * pn + 0.40 * dn

    rows.sort(
        key=lambda r: r["score"],
        reverse=True,
    )

    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "candidate",
        "plus_mean_A_B_iptm",
        "minus_mean_A_B_iptm",
        "delta_A_B_iptm",
        "plus_norm",
        "delta_norm",
        "score",
        "plus_min_A_B_iptm",
        "plus_max_A_B_iptm",
        "minus_min_A_B_iptm",
        "minus_max_A_B_iptm",
        "plus_samples",
        "minus_samples",
        "best_plus_sample",
        "best_plus_sample_iptm",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                k: row[k]
                for k in fieldnames
            })

    if TOP_DIR.exists():
        shutil.rmtree(TOP_DIR)

    TOP_DIR.mkdir(parents=True)

    for row in rows[:10]:
        rank = row["rank"]
        candidate = row["candidate"]
        sample = row["best_plus_sample"]

        src = Path(row["_best_cif"])

        dst = (
            TOP_DIR
            / f"rank{rank:02d}__{candidate}"
              f"__sample{sample}.cif"
        )

        shutil.copy2(src, dst)

    print()
    print("=== TOP 10 ===")

    for row in rows[:10]:
        print(
            f"{row['rank']:2d}. "
            f"{row['candidate']}  "
            f"+RAP={row['plus_mean_A_B_iptm']:.3f}  "
            f"-RAP={row['minus_mean_A_B_iptm']:.3f}  "
            f"delta={row['delta_A_B_iptm']:+.3f}  "
            f"score={row['score']:.3f}  "
            f"best_sample={row['best_plus_sample']}"
        )

    print()
    print(f"Ranking CSV: {OUT_CSV}")
    print(f"Top10 CIFs:  {TOP_DIR}")


if __name__ == "__main__":
    main()
