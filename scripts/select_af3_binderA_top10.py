#!/usr/bin/env python3

import shutil
from pathlib import Path
import pandas as pd

PROJECT = Path("/home01/a2149a01/RFdiffusion3")
ROOT = PROJECT / "outputs/af3_binderA_pilot"
SAMPLE_CSV = PROJECT / "outputs/af3_binderA_pilot_metrics_samples.csv"
CAND_CSV = PROJECT / "outputs/af3_binderA_pilot_metrics_candidates.csv"
OUT = PROJECT / "outputs/binderA_top10_review"

OUT.mkdir(parents=True, exist_ok=True)

s = pd.read_csv(SAMPLE_CSV)
c = pd.read_csv(CAND_CSV)

# 1. Robust correct-interface recovery 우선
pool = c[c["samples_with_both"] >= 3].copy()

# 2. 14 candidates -> 10
# Lexicographic selection:
# correct-interface reproducibility > fewer decoys >
# interface recovery > fold/pose recovery
pool = pool.sort_values(
    by=[
        "samples_with_both",
        "obvious_decoy_samples",
        "max_FRB_interface_Jaccard",
        "max_RAP_atom_interface_Jaccard",
        "best_BinderFold",
        "best_TargetPose",
    ],
    ascending=[False, True, False, False, True, True]
)

top10 = pool.head(10).copy()

selected_rows = []

for rank, (_, cand) in enumerate(top10.iterrows(), 1):
    name = cand["candidate"]

    x = s[s["candidate"] == name].copy()

    # Representative structure must contact BOTH RAP and intended FRB patch
    x = x[
        (x["RAP_contact_res_count"] > 0) &
        (x["intended_FRB_patch_count"] > 0)
    ].copy()

    # Best representative sample:
    # interface recovery first, then fold/pose, then confidence.
    x = x.sort_values(
        by=[
            "FRB_interface_Jaccard",
            "RAP_atom_interface_Jaccard",
            "BinderFold",
            "TargetPose",
            "ranking_score",
        ],
        ascending=[False, False, True, True, False]
    )

    best = x.iloc[0]
    sample = int(best["sample"])

    src_dir = ROOT / f"{name}_plus_RAP" / f"seed-1_sample-{sample}"
    cif = list(src_dir.glob("*_model.cif"))

    if len(cif) != 1:
        raise RuntimeError(f"{name}: expected 1 CIF, found {len(cif)}")

    dst = OUT / f"{rank:02d}_{name}_sample{sample}.cif"
    shutil.copy2(cif[0], dst)

    selected_rows.append({
        "rank": rank,
        "candidate": name,
        "parent": cand["parent"],
        "sample": sample,
        "samples_with_both": int(cand["samples_with_both"]),
        "decoy_samples": int(cand["obvious_decoy_samples"]),
        "BinderFold": best["BinderFold"],
        "TargetPose": best["TargetPose"],
        "RAPpose": best["RAPpose"],
        "RAP_contact_res": int(best["RAP_contact_res_count"]),
        "FRB_contact_res": int(best["FRB_contact_res_count"]),
        "intended_patch": int(best["intended_FRB_patch_count"]),
        "FRB_Jaccard": best["FRB_interface_Jaccard"],
        "RAP_Jaccard": best["RAP_atom_interface_Jaccard"],
        "RAP_relative_SASA": best["RAP_relative_SASA"],
        "ranking_score": best["ranking_score"],
        "file": dst.name,
    })

review = pd.DataFrame(selected_rows)
review.to_csv(OUT / "top10_metrics.csv", index=False)

print(review.to_string(index=False))
print()
print(f"Review directory: {OUT}")
print(f"CIF files: {len(list(OUT.glob('*.cif')))}")
