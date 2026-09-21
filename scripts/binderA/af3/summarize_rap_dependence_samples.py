#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

PROJECT = Path("/home01/a2149a01/RFdiffusion3")

PLUS = PROJECT / "outputs/af3_binderA_pilot_metrics_samples.csv"
MINUS = PROJECT / "outputs/af3_binderA_noRAP_metrics_samples.csv"
OUT = PROJECT / "outputs/binderA_plus_minus_RAP_sample_summary.csv"

p = pd.read_csv(PLUS)
m = pd.read_csv(MINUS)

if len(p) != 440 or len(m) != 440:
    raise RuntimeError(f"Expected 440 + 440 rows, got {len(p)} + {len(m)}")


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


# +RAP
pg = p.groupby("candidate").agg(
    plus_mean_AB_iptm=("A_B_iptm", "mean"),
    plus_median_AB_iptm=("A_B_iptm", "median"),
    plus_min_AB_iptm=("A_B_iptm", "min"),
    plus_max_AB_iptm=("A_B_iptm", "max"),

    plus_mean_BinderFold=("BinderFold", "mean"),
    plus_best_BinderFold=("BinderFold", "min"),

    plus_mean_FRB_Jaccard=("FRB_interface_Jaccard", "mean"),
    plus_max_FRB_Jaccard=("FRB_interface_Jaccard", "max"),

    plus_mean_FRB_contacts=("FRB_contact_res_count", "mean"),

    plus_mean_intended_patch=("intended_FRB_patch_count", "mean"),

    plus_mean_RAP_contacts=("RAP_contact_res_count", "mean"),

    plus_q25_AB_iptm=("A_B_iptm", q25),
    plus_q75_AB_iptm=("A_B_iptm", q75),
)

# Explicit reproducibility counts
pg["plus_RAP_contact_samples"] = (
    p["RAP_contact_res_count"].gt(0)
    .groupby(p["candidate"]).sum()
)

pg["plus_intended_samples"] = (
    p["intended_FRB_patch_count"].gt(0)
    .groupby(p["candidate"]).sum()
)

pg["plus_both_samples"] = (
    (
        p["RAP_contact_res_count"].gt(0)
        & p["intended_FRB_patch_count"].gt(0)
    )
    .groupby(p["candidate"]).sum()
)

pg["plus_decoy_samples"] = (
    p["obvious_wrong_FRB_face_decoy"]
    .groupby(p["candidate"]).sum()
)


# -RAP
mg = m.groupby("candidate").agg(
    minus_mean_AB_iptm=("A_B_iptm", "mean"),
    minus_median_AB_iptm=("A_B_iptm", "median"),
    minus_min_AB_iptm=("A_B_iptm", "min"),
    minus_max_AB_iptm=("A_B_iptm", "max"),

    minus_mean_BinderFold=("BinderFold", "mean"),
    minus_best_BinderFold=("BinderFold", "min"),

    minus_mean_FRB_Jaccard=("FRB_interface_Jaccard", "mean"),
    minus_max_FRB_Jaccard=("FRB_interface_Jaccard", "max"),

    minus_mean_FRB_contacts=("FRB_contact_res_count", "mean"),

    minus_mean_contact_pairs=("AB_contact_residue_pair_count", "mean"),

    minus_mean_intended_patch=("intended_FRB_patch_count", "mean"),

    minus_mean_TargetPose=("TargetPose", "mean"),
    minus_best_TargetPose=("TargetPose", "min"),

    minus_q25_AB_iptm=("A_B_iptm", q25),
    minus_q75_AB_iptm=("A_B_iptm", q75),
)

mg["minus_intended_samples"] = (
    m["intended_FRB_patch_count"].gt(0)
    .groupby(m["candidate"]).sum()
)


df = pg.join(mg, how="inner")

if len(df) != 88:
    raise RuntimeError(f"Expected 88 candidates, got {len(df)}")


# True like-for-like changes
df["delta_mean_AB_iptm"] = (
    df["plus_mean_AB_iptm"]
    - df["minus_mean_AB_iptm"]
)

df["delta_median_AB_iptm"] = (
    df["plus_median_AB_iptm"]
    - df["minus_median_AB_iptm"]
)

df["delta_mean_FRB_Jaccard"] = (
    df["plus_mean_FRB_Jaccard"]
    - df["minus_mean_FRB_Jaccard"]
)

df["delta_mean_FRB_contacts"] = (
    df["plus_mean_FRB_contacts"]
    - df["minus_mean_FRB_contacts"]
)

df["delta_intended_samples"] = (
    df["plus_intended_samples"]
    - df["minus_intended_samples"]
)

df = df.reset_index()

df.to_csv(OUT, index=False)

print("Candidates:", len(df))
print("Output:", OUT)

print("\n=== mean A-B ipTM ===")
print(
    df[
        [
            "plus_mean_AB_iptm",
            "minus_mean_AB_iptm",
            "delta_mean_AB_iptm",
        ]
    ].describe().round(3)
)

print("\n=== candidates with +RAP both >= 3 ===")

x = df[df["plus_both_samples"] >= 3].copy()

x = x.sort_values(
    [
        "plus_both_samples",
        "minus_intended_samples",
        "minus_mean_AB_iptm",
    ],
    ascending=[False, True, True],
)

cols = [
    "candidate",
    "plus_both_samples",
    "plus_mean_AB_iptm",
    "minus_mean_AB_iptm",
    "delta_mean_AB_iptm",
    "plus_mean_FRB_Jaccard",
    "minus_mean_FRB_Jaccard",
    "delta_mean_FRB_Jaccard",
    "plus_intended_samples",
    "minus_intended_samples",
    "plus_mean_RAP_contacts",
    "minus_mean_contact_pairs",
]

print(x[cols].round(3).to_string(index=False))