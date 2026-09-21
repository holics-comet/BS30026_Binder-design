#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

PROJECT = Path("/home01/a2149a01/RFdiffusion3")

PLUS = PROJECT / "outputs/af3_binderA_pilot_metrics_candidates.csv"
MINUS = PROJECT / "outputs/af3_binderA_noRAP_metrics_candidates.csv"
OUT = PROJECT / "outputs/binderA_plus_minus_RAP_comparison.csv"

plus = pd.read_csv(PLUS)
minus = pd.read_csv(MINUS)

print("plus candidates :", len(plus))
print("minus candidates:", len(minus))

# Rename before merge so every metric is unambiguous.
plus = plus.add_prefix("plus_")
minus = minus.add_prefix("minus_")

df = plus.merge(
    minus,
    left_on="plus_candidate",
    right_on="minus_candidate",
    how="inner",
)

if len(df) != 88:
    raise RuntimeError(f"Expected 88 matched candidates, got {len(df)}")

df["candidate"] = df["plus_candidate"]

# -------------------------
# +RAP descriptive metrics
# -------------------------

df["plus_both_fraction"] = (
    df["plus_samples_with_both"] /
    df["plus_n_samples"]
)

df["plus_RAP_contact_fraction"] = (
    df["plus_samples_with_RAP_contact"] /
    df["plus_n_samples"]
)

df["plus_intended_patch_fraction"] = (
    df["plus_samples_with_intended_FRB_patch"] /
    df["plus_n_samples"]
)

df["plus_decoy_fraction"] = (
    df["plus_obvious_decoy_samples"] /
    df["plus_n_samples"]
)

# -------------------------
# -RAP descriptive metrics
# -------------------------

df["minus_intended_patch_fraction"] = (
    df["minus_samples_with_intended_FRB_patch"] /
    df["minus_n_samples"]
)

# -------------------------
# RAP-dependent changes
# -------------------------

# Positive = intended FRB patch is recovered more often with RAP.
df["delta_intended_patch_fraction"] = (
    df["plus_intended_patch_fraction"]
    - df["minus_intended_patch_fraction"]
)

# Positive = A-B pair ipTM is higher with RAP.
df["delta_mean_AB_iptm"] = (
    df["plus_max_A_B_iptm"]
    - df["minus_mean_A_B_iptm"]
)

df["delta_max_AB_iptm"] = (
    df["plus_max_A_B_iptm"]
    - df["minus_max_A_B_iptm"]
)

# NOTE:
# Existing +RAP candidate CSV contains max A-B ipTM but not mean A-B ipTM.
# Therefore these ipTM deltas are descriptive only and are NOT matched
# mean-vs-mean comparisons.

# Keep useful columns near the front.
front = [
    "candidate",

    "plus_samples_with_both",
    "plus_both_fraction",
    "plus_samples_with_RAP_contact",
    "plus_RAP_contact_fraction",
    "plus_samples_with_intended_FRB_patch",
    "plus_intended_patch_fraction",
    "plus_obvious_decoy_samples",
    "plus_decoy_fraction",

    "minus_samples_with_intended_FRB_patch",
    "minus_intended_patch_fraction",

    "plus_max_A_B_iptm",
    "minus_mean_A_B_iptm",
    "minus_max_A_B_iptm",

    "delta_intended_patch_fraction",
    "delta_mean_AB_iptm",
    "delta_max_AB_iptm",

    "minus_mean_FRB_contact_res_count",
    "minus_mean_AB_contact_residue_pair_count",
    "minus_mean_FRB_interface_Jaccard",
    "minus_mean_TargetPose",
]

rest = [c for c in df.columns if c not in front]

df = df[front + rest]

df.to_csv(OUT, index=False)

print()
print("Matched candidates:", len(df))
print("Output:", OUT)

print()
print("=== +RAP composite-interface reproducibility ===")
print(
    df["plus_samples_with_both"]
    .value_counts()
    .sort_index()
)

print()
print("=== -RAP intended-patch persistence ===")
print(
    df["minus_samples_with_intended_FRB_patch"]
    .value_counts()
    .sort_index()
)

print()
print("=== delta intended-patch fraction ===")
print(
    df["delta_intended_patch_fraction"]
    .describe()
    .round(3)
)

print()
print("=== -RAP mean A-B pair ipTM ===")
print(
    df["minus_mean_A_B_iptm"]
    .describe()
    .round(3)
)

print()
print("=== -RAP max A-B pair ipTM ===")
print(
    df["minus_max_A_B_iptm"]
    .describe()
    .round(3)
)

print()
print("=== candidates with strongest +RAP geometry ===")

show = df.sort_values(
    [
        "plus_samples_with_both",
        "plus_obvious_decoy_samples",
    ],
    ascending=[False, True],
).head(20)

print(
    show[
        [
            "candidate",
            "plus_samples_with_both",
            "plus_obvious_decoy_samples",
            "minus_samples_with_intended_FRB_patch",
            "minus_mean_A_B_iptm",
            "minus_max_A_B_iptm",
            "minus_mean_FRB_interface_Jaccard",
        ]
    ].to_string(index=False)
)