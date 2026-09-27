#!/usr/bin/env python3

import shutil
from pathlib import Path
import pandas as pd

OUTROOT = Path("/scratch/a2149a01/RFdiffusion3_data/outputs")

PLUS_CSV = OUTROOT / "af3_binderA_pilot_metrics_samples.csv"
MINUS_CSV = OUTROOT / "af3_binderA_noRAP_metrics_samples.csv"
PLUS_ROOT = OUTROOT / "af3_binderA_pilot"
TOP10_DIR = OUTROOT / "binderA_top10"

N_TOP = 10

plus = pd.read_csv(PLUS_CSV)
minus = pd.read_csv(MINUS_CSV)

# +RAP 핵심 지표
plus_summary = plus.groupby("candidate").agg(
    plus_decoy_samples=("obvious_wrong_FRB_face_decoy", "sum"),
    plus_FRB_Jaccard=("FRB_interface_Jaccard", "mean"),
    plus_BinderFold=("BinderFold", "mean"),
    plus_TargetPose=("TargetPose", "mean"),
)

plus_summary["plus_both_samples"] = (
    (
        plus["RAP_contact_res_count"].gt(0)
        & plus["intended_FRB_patch_count"].gt(0)
    )
    .groupby(plus["candidate"])
    .sum()
)

# -RAP 핵심 지표
minus_summary = minus.groupby("candidate").agg(
    minus_intended_samples=(
        "intended_FRB_patch_count",
        lambda x: (x > 0).sum(),
    ),
    minus_AB_iptm=("A_B_iptm", "mean"),
)

cand = plus_summary.join(minus_summary, how="inner").reset_index()

# Hard filter: +RAP 5개 sample 중 최소 3개가
# RAP contact + intended FRB patch contact를 동시에 만족
pool = cand[cand["plus_both_samples"] >= 3].copy()

# Ranking:
# +RAP 재현성 > -RAP 결합 억제 > decoy 억제
# > interface recovery > fold/pose recovery
pool = pool.sort_values(
    [
        "plus_both_samples",
        "minus_intended_samples",
        "minus_AB_iptm",
        "plus_decoy_samples",
        "plus_FRB_Jaccard",
        "plus_BinderFold",
        "plus_TargetPose",
    ],
    ascending=[False, True, True, True, False, True, True],
)

top10 = pool.head(N_TOP).copy()

TOP10_DIR.mkdir(parents=True, exist_ok=True)

for f in TOP10_DIR.glob("*.cif"):
    f.unlink()

rows = []

for rank, (_, c) in enumerate(top10.iterrows(), 1):
    name = c["candidate"]

    x = plus[
        (plus["candidate"] == name)
        & (plus["RAP_contact_res_count"] > 0)
        & (plus["intended_FRB_patch_count"] > 0)
    ].copy()

    # 대표 구조는 올바른 interface recovery를 최우선
    x = x.sort_values(
        [
            "FRB_interface_Jaccard",
            "BinderFold",
            "TargetPose",
            "A_B_iptm",
        ],
        ascending=[False, True, True, False],
    )

    best = x.iloc[0]
    sample = int(best["sample"])

    src_dir = PLUS_ROOT / f"{name}_plus_RAP" / f"seed-1_sample-{sample}"
    cif = list(src_dir.glob("*_model.cif"))

    if len(cif) != 1:
        raise RuntimeError(
            f"{name} sample {sample}: expected 1 CIF, found {len(cif)}"
        )

    dst = TOP10_DIR / f"{rank:02d}_{name}_sample{sample}.cif"
    shutil.copy2(cif[0], dst)

    rows.append({
        "rank": rank,
        "candidate": name,
        "sample": sample,
        "plus_both_samples": int(c["plus_both_samples"]),
        "minus_intended_samples": int(c["minus_intended_samples"]),
        "minus_AB_iptm": c["minus_AB_iptm"],
        "plus_decoy_samples": int(c["plus_decoy_samples"]),
        "FRB_Jaccard": best["FRB_interface_Jaccard"],
        "BinderFold": best["BinderFold"],
        "TargetPose": best["TargetPose"],
        "file": dst.name,
    })

result = pd.DataFrame(rows)
result.to_csv(TOP10_DIR / "top10.csv", index=False)

print(result.round(3).to_string(index=False))
print()
print(f"Selected: {len(result)}")
print(f"Output:   {TOP10_DIR}")