#!/usr/bin/env python3

import csv
import os
import shutil
from collections import defaultdict

PROJECT = "/home01/a2149a01/RFdiffusion3"

BB_CSV = f"{PROJECT}/outputs/frb_rap_raphotspot_backbone_geometry.csv"
REACH_CSV = f"{PROJECT}/outputs/frb_rap_raphotspot_sidechain_reach.csv"
SRC_DIR = f"{PROJECT}/outputs/frb_rap_raphotspot_pilot"
OUT_CSV = f"{PROJECT}/outputs/binderA_pilot_selection.csv"
OUT_DIR = f"{PROJECT}/outputs/binderA_ligandmpnn_pilot"

MAX_PER_CLASS = 3


def read_csv(path):
    with open(path, newline="") as f:
        return {row["design"]: row for row in csv.DictReader(f)}


def geom_class(d):
    if d <= 5:
        return "near"
    if d <= 8:
        return "mid"
    return "far"


def len_class(n):
    if n <= 59:
        return "50-59"
    if n <= 69:
        return "60-69"
    return "70-80"


bb = read_csv(BB_CSV)
reach = read_csv(REACH_CSV)

designs = sorted(set(bb) & set(reach))

print(f"Matched designs: {len(designs)}")

records = []

for design in designs:
    b = bb[design]
    r = reach[design]

    length = int(b["binder_length"])
    rap_res = int(b["bb_rap_contact_residue_count"])
    frb_res = int(b["bb_frb_contact_residue_count"])
    bridge = int(b["bb_bridge_residue_count"])
    interface = int(b["bb_interface_residue_count"])
    clashes = int(b["bb_total_clashes_lt2A"])

    o3 = float(r["RAP_O3_CBpref_min_distance"])
    o8 = float(r["RAP_O8_CBpref_min_distance"])
    b25 = float(r["B25_ARG_CBpref_min_distance"])
    b88 = float(r["B88_TYR_CBpref_min_distance"])

    hard_pass = (
        clashes == 0
        and rap_res >= 1
        and frb_res >= 1
    )

    records.append({
        "design": design,
        "binder_length": length,
        "length_class": len_class(length),
        "bb_rap_contact_residues": rap_res,
        "bb_frb_contact_residues": frb_res,
        "bridge_residues": bridge,
        "interface_residues": interface,
        "interface_fraction": interface / length,
        "bb_clashes": clashes,
        "O3": o3,
        "O8": o8,
        "B25": b25,
        "B88": b88,
        "O8_class": geom_class(o8),
        "B88_class": geom_class(b88),
        "hard_pass": hard_pass,
        "selected": False,
        "selection_reason": "",
    })


passed = [r for r in records if r["hard_pass"]]
failed = [r for r in records if not r["hard_pass"]]

print(f"\n=== HARD FILTER ===")
print(f"PASS: {len(passed)}/{len(records)}")
print(f"FAIL: {len(failed)}/{len(records)}")


# ------------------------------------------------------------
# 1. Keep all bridge-containing structures
# ------------------------------------------------------------

selected_names = set()

for r in passed:
    if r["bridge_residues"] >= 1:
        r["selected"] = True
        r["selection_reason"] = "bridge"
        selected_names.add(r["design"])


# ------------------------------------------------------------
# 2. Diversity selection by length × O8 × B88 class
# ------------------------------------------------------------

groups = defaultdict(list)

for r in passed:
    key = (
        r["length_class"],
        r["O8_class"],
        r["B88_class"],
    )
    groups[key].append(r)


for key, members in groups.items():

    members.sort(
        key=lambda r: (
            -r["bridge_residues"],
            -r["interface_fraction"],
            -min(
                r["bb_rap_contact_residues"],
                r["bb_frb_contact_residues"],
            ),
            r["O8"] + r["B88"],
        )
    )

    added = 0

    for r in members:
        if r["design"] in selected_names:
            continue

        r["selected"] = True
        r["selection_reason"] = "diversity"
        selected_names.add(r["design"])

        added += 1

        if added >= MAX_PER_CLASS:
            break


selected = [r for r in records if r["selected"]]


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print(f"\n=== SELECTED ===")
print(f"Selected total: {len(selected)}\n")

print(
    f"{'design':65s} "
    f"{'len':>3s} "
    f"{'RAP':>3s} "
    f"{'FRB':>3s} "
    f"{'br':>2s} "
    f"{'int':>3s} "
    f"{'frac':>5s} "
    f"{'O8':>5s} "
    f"{'B88':>5s} "
    f"{'reason':>9s}"
)

for r in sorted(
    selected,
    key=lambda x: (
        x["binder_length"],
        -x["bridge_residues"],
        -x["interface_fraction"],
    ),
):
    print(
        f"{r['design']:65s} "
        f"{r['binder_length']:3d} "
        f"{r['bb_rap_contact_residues']:3d} "
        f"{r['bb_frb_contact_residues']:3d} "
        f"{r['bridge_residues']:2d} "
        f"{r['interface_residues']:3d} "
        f"{r['interface_fraction']:5.3f} "
        f"{r['O8']:5.2f} "
        f"{r['B88']:5.2f} "
        f"{r['selection_reason']:>9s}"
    )


# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------

fields = list(records[0].keys())

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print(f"\nCSV written to:\n{OUT_CSV}")


# ------------------------------------------------------------
# Copy selected CIF files
# ------------------------------------------------------------

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)

os.makedirs(OUT_DIR)

copied = 0

for r in selected:
    design = r["design"]

    for ext in [".cif.gz", ".cif"]:
        src = f"{SRC_DIR}/{design}{ext}"

        if os.path.isfile(src):
            shutil.copy2(src, OUT_DIR)
            copied += 1
            break
    else:
        print(f"WARNING: file not found: {design}")

print(f"\nCopied {copied} structures to:\n{OUT_DIR}")