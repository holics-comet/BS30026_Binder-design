#!/usr/bin/env python3

import csv
import gzip
import glob
import os
import shutil
import numpy as np


PROJECT = "/home01/a2149a01/RFdiffusion3"
INPUT_ROOT = f"{PROJECT}/outputs/binderB_rfd3"

OUTPUT_CSV = f"{PROJECT}/outputs/binderB_backbone_selected.csv"
OUTPUT_DIR = f"{PROJECT}/outputs/binderB_backbone_selected"

CONTACT = 4.0
CLASH = 2.0
N_PER_TARGET = 10


HOTSPOTS = {
    "1_model_7_b0_d0": {
        "A": {44: ["SD", "CE"]},
        "RAP": ["O3", "N7"],
    },
    "3_model_3_b2_d0": {
        "A": {72: ["NE", "NH1", "NH2"]},
        "RAP": ["O8", "O9"],
    },
    "5_model_1_b0_d0": {
        "A": {52: ["OH"]},
        "RAP": ["O10", "O2"],
    },
    "6_model_5_b3_d0": {
        "A": {54: ["NE", "NH1", "NH2"]},
        "RAP": ["O10", "O13"],
    },
    "9_model_4_b0_d0": {
        "A": {24: ["OD1", "OD2"]},
        "RAP": ["O2", "O3"],
    },
}


def read_cif(path):
    atoms = []
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(("ATOM ", "HETATM ")):
                continue

            x = line.split()

            element = x[1]
            atom = x[2]

            if element == "H" or atom.startswith("H"):
                continue

            atoms.append({
                "element": element,
                "atom": atom,
                "resname": x[4],
                "chain": x[5],
                "resid": int(x[9]),
                "xyz": np.array(
                    [float(x[18]), float(x[19]), float(x[20])]
                ),
            })

    return atoms


def select(atoms, chain=None, resname=None):
    return [
        a for a in atoms
        if (chain is None or a["chain"] == chain)
        and (resname is None or a["resname"] == resname)
    ]

def distances(group1, group2):
    if not group1 or not group2:
        return np.empty((0, 0))

    a = np.array([x["xyz"] for x in group1])
    b = np.array([x["xyz"] for x in group2])

    return np.linalg.norm(
        a[:, None, :] - b[None, :, :],
        axis=2,
    )


def contact_count(group1, group2, cutoff):
    d = distances(group1, group2)
    return int(np.sum(d <= cutoff)) if d.size else 0


def contacted_residues(group1, group2, cutoff):
    if not group1 or not group2:
        return set()

    target = np.array([a["xyz"] for a in group2])
    contacted = set()

    for atom in group1:
        if np.any(
            np.linalg.norm(target - atom["xyz"], axis=1) <= cutoff
        ):
            contacted.add((atom["resid"], atom["resname"]))

    return contacted


def hotspot_contact(binder_b, target_atoms, resid, atom_names):
    selected_atoms = [
        a for a in target_atoms
        if a["resid"] == resid
        and a["atom"] in atom_names
    ]

    d = distances(binder_b, selected_atoms)

    if not d.size:
        return False

    return bool(d.min() <= CONTACT)


def analyze(path, target):
    atoms = read_cif(path)

    # RFD3 output:
    # A = fixed Binder A
    # B = generated Binder B
    # C = RAP
    A = select(atoms, chain="A")
    B = select(atoms, chain="B")
    RAP = select(atoms, chain="C", resname="RAP")

    if not A or not B or not RAP:
        raise RuntimeError(f"Missing A/B/RAP: {path}")

    b_rap_contacts = contact_count(B, RAP, CONTACT)
    b_a_contacts = contact_count(B, A, CONTACT)

    a_residues = contacted_residues(A, B, CONTACT)

    rap_hotspot_atoms = [
        a for a in RAP
        if a["atom"] in HOTSPOTS[target]["RAP"]
    ]

    rap_hotspot = (
        distances(B, rap_hotspot_atoms).min() <= CONTACT
        if rap_hotspot_atoms else False
    )

    a_hotspot = any(
        hotspot_contact(B, A, resid, names)
        for resid, names in HOTSPOTS[target]["A"].items()
    )

    clashes = (
        contact_count(B, A, CLASH)
        + contact_count(B, RAP, CLASH)
    )

    return {
        "target": target,
        "design": os.path.basename(path).replace(".cif.gz", ""),
        "B_RAP_atom_contacts_4A": b_rap_contacts,
        "B_A_atom_contacts_4A": b_a_contacts,
        "A_residues_contacted_count": len(a_residues),
        "RAP_hotspot_contact_4A": rap_hotspot,
        "A_hotspot_contact_4A": a_hotspot,
        "composite_contact_4A": (
            b_rap_contacts > 0 and b_a_contacts > 0
        ),
        "total_inter_clashes_lt2A": clashes,
    }

def normalize(values):
    lo = min(values)
    hi = max(values)

    if hi == lo:
        return [0.0] * len(values)

    return [(x - lo) / (hi - lo) for x in values]


def rank_target(rows):
    # Hard filter:
    # Binder B must contact both RAP and Binder A, with no <2 Å clashes.
    rows = [
        r for r in rows
        if r["composite_contact_4A"]
        and r["total_inter_clashes_lt2A"] == 0
    ]

    if not rows:
        return []

    rap_scores = normalize([
        r["B_RAP_atom_contacts_4A"] for r in rows
    ])

    a_scores = normalize([
        r["B_A_atom_contacts_4A"] for r in rows
    ])

    a_res_scores = normalize([
        r["A_residues_contacted_count"] for r in rows
    ])

    for r, rap, a, a_res in zip(
        rows, rap_scores, a_scores, a_res_scores
    ):
        hotspot = (
            int(r["RAP_hotspot_contact_4A"])
            + int(r["A_hotspot_contact_4A"])
        ) / 2.0

        r["selection_score"] = (
            0.35 * rap
            + 0.30 * a
            + 0.20 * a_res
            + 0.15 * hotspot
        )

    rows.sort(
        key=lambda r: (
            r["selection_score"],
            r["B_RAP_atom_contacts_4A"],
            r["B_A_atom_contacts_4A"],
        ),
        reverse=True,
    )

    top = rows[:N_PER_TARGET]

    for rank, r in enumerate(top, 1):
        r["rank_within_target"] = rank

    return top

def main():
    selected = []

    for target in HOTSPOTS:
        pattern = os.path.join(
            INPUT_ROOT,
            target,
            "*.cif.gz",
        )

        files = sorted(glob.glob(pattern))

        print(f"{target}: {len(files)} structures")

        if len(files) != 80:
            print(f"  WARNING: expected 80, found {len(files)}")

        rows = [
            analyze(path, target)
            for path in files
        ]

        top = rank_target(rows)
        selected.extend(top)

        print(f"  passed/selected: {len(top)}")

    if not selected:
        raise RuntimeError("No structures selected.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    for row in selected:
        src = os.path.join(
            INPUT_ROOT,
            row["target"],
            row["design"] + ".cif.gz",
        )

        dst = os.path.join(
            OUTPUT_DIR,
            f"{row['target']}__"
            f"rank{row['rank_within_target']:02d}__"
            f"{row['design']}.cif.gz",
        )

        if not os.path.exists(src):
            raise FileNotFoundError(src)

        shutil.copy2(src, dst)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(selected[0].keys()),
        )
        writer.writeheader()
        writer.writerows(selected)

    print()
    print(f"Selected: {len(selected)}")
    print(f"CSV: {OUTPUT_CSV}")