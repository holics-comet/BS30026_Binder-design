import os
import shutil
import pandas as pd


PROJECT = "/home01/a2149a01/RFdiffusion3"

METRICS = f"{PROJECT}/outputs/binderB_backbone_metrics.csv"
SOURCE_ROOT = f"{PROJECT}/outputs/binderB_rfd3"

OUTPUT_CSV = f"{PROJECT}/outputs/binderB_backbone_selected.csv"
OUTPUT_DIR = f"{PROJECT}/outputs/binderB_backbone_selected"

N_PER_TARGET = 10


def normalize(series):
    """0–1 normalization within one target."""
    lo = series.min()
    hi = series.max()

    if hi == lo:
        return pd.Series(0.0, index=series.index)

    return (series - lo) / (hi - lo)


def main():
    df = pd.read_csv(METRICS)

    # Hard filters: biological minimum + no severe clash
    df = df[
        (df["composite_contact_4A"] == True) &
        (df["total_inter_clashes_lt2A"] == 0)
    ].copy()

    selected = []

    for target, x in df.groupby("target"):
        x = x.copy()

        # Interface quality
        x["rap_score"] = normalize(
            x["B_RAP_atom_contacts_4A"]
        )

        x["a_score"] = normalize(
            x["B_A_atom_contacts_4A"]
        )

        x["a_res_score"] = normalize(
            x["A_residues_contacted_count"]
        )

        # Hotspots are bonuses, not hard filters
        x["hotspot_score"] = (
            x["RAP_hotspot_contact_4A"].astype(int) +
            x["A_hotspot_contact_4A"].astype(int)
        ) / 2.0

        # Balanced composite-interface score
        x["selection_score"] = (
            0.35 * x["rap_score"] +
            0.30 * x["a_score"] +
            0.20 * x["a_res_score"] +
            0.15 * x["hotspot_score"]
        )

        x = x.sort_values(
            [
                "selection_score",
                "B_RAP_atom_contacts_4A",
                "B_A_atom_contacts_4A",
            ],
            ascending=False,
        )

        top = x.head(N_PER_TARGET).copy()
        top["rank_within_target"] = range(
            1, len(top) + 1
        )

        selected.append(top)

    out = pd.concat(selected, ignore_index=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean old selection
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    # Copy selected RFD3 structures
    for _, row in out.iterrows():
        src = os.path.join(
            SOURCE_ROOT,
            row["target"],
            row["design"] + ".cif.gz",
        )

        dst_name = (
            f"{row['target']}__"
            f"rank{int(row['rank_within_target']):02d}__"
            f"{row['design']}.cif.gz"
        )

        dst = os.path.join(
            OUTPUT_DIR,
            dst_name,
        )

        if not os.path.exists(src):
            raise FileNotFoundError(src)

        shutil.copy2(src, dst)

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Selected: {len(out)}")

    print("\n=== Per target ===")
    print(out.groupby("target").size())

    print("\n=== Selected interface summary ===")
    cols = [
        "B_RAP_atom_contacts_4A",
        "B_A_atom_contacts_4A",
        "A_residues_contacted_count",
        "RAP_hotspot_contact_4A",
        "A_hotspot_contact_4A",
        "selection_score",
    ]

    print(
        out.groupby("target")[cols]
        .mean()
        .round(2)
    )

    print()
    print(f"CSV: {OUTPUT_CSV}")
    print(f"CIFs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
