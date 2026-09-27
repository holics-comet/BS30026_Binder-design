#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import numpy as np

from biotite.structure.io.pdbx import CIFFile, get_structure


ROOT = Path("/home01/a2149a01/RFdiffusion3")
TARGET_DIR = ROOT / "inputs/binderB/targets"
RFD3_DIR = ROOT / "inputs/binderB/rfd3"
CSV_OUT = ROOT / "outputs/binderB_hotspot_candidates.csv"

RAP_MIN = 5.0
RAP_MAX = 9.0
A_NEAR_RAP = 10.0


# candidate: (Binder A length, Binder A hotspot, RAP hotspots)
CONFIG = {
    "1_model_7_b0_d0": (
        51,
        {"A44": "SD,CE"},
        {"RAP": "O3,N7"},
    ),
    "3_model_3_b2_d0": (
        75,
        {"A72": "NE,NH1,NH2"},
        {"RAP": "O8,O9"},
    ),
    "5_model_1_b0_d0": (
        62,
        {"A52": "OH"},
        {"RAP": "O10,O2"},
    ),
    "6_model_5_b3_d0": (
        67,
        {"A54": "NE,NH1,NH2"},
        {"RAP": "O10,O13"},
    ),
    "9_model_4_b0_d0": (
        55,
        {"A24": "OD1,OD2"},
        {"RAP": "O2,O3"},
    ),
}


RFD3_DIR.mkdir(parents=True, exist_ok=True)
rows = []


for candidate, (length, protein_hs, rap_hs) in CONFIG.items():

    target = TARGET_DIR / f"{candidate}_A_RAP.cif"

    if not target.exists():
        raise FileNotFoundError(target)

    structure = get_structure(CIFFile.read(target), model=1)

    A = structure[structure.chain_id == "A"]
    RAP = structure[structure.chain_id == "C"]

    A = A[A.element != "H"]
    RAP = RAP[RAP.element != "H"]

    print(f"\n=== {candidate} ===")

    hotspot_candidates = []

    # --------------------------------------------------------
    # 1. Analyze RAP hotspot candidates
    # --------------------------------------------------------

    for atom in RAP:

        distances = np.linalg.norm(
            A.coord - atom.coord,
            axis=1,
        )

        min_dist = float(distances.min())

        near = A[distances <= A_NEAR_RAP]

        near_residues = sorted(set(
            (int(x.res_id), str(x.res_name))
            for x in near
        ))

        residue_info = []

        for res_id, res_name in near_residues:

            residue = A[
                (A.res_id == res_id)
                & (A.res_name == res_name)
            ]

            d = np.linalg.norm(
                residue.coord - atom.coord,
                axis=1,
            ).min()

            residue_info.append(
                (float(d), res_id, res_name)
            )

        residue_info.sort()

        closest = ";".join(
            f"A{res_id}:{res_name}:{dist:.2f}"
            for dist, res_id, res_name
            in residue_info[:5]
        )

        polar = str(atom.element) in {"O", "N", "S"}
        geometry_ok = RAP_MIN <= min_dist <= RAP_MAX

        row = {
            "candidate": candidate,
            "RAP_atom": str(atom.atom_name),
            "element": str(atom.element),
            "min_distance_to_A": min_dist,
            "polar": polar,
            "geometry_ok": geometry_ok,
            "closest_A_residues": closest,
        }

        rows.append(row)

        if geometry_ok:
            hotspot_candidates.append(row)

    hotspot_candidates.sort(
        key=lambda x: (
            not x["polar"],
            abs(x["min_distance_to_A"] - 7.0),
        )
    )

    print("Hotspot candidates:")
    for x in hotspot_candidates[:12]:
        flag = "*" if x["polar"] else " "
        print(
            f" {flag} RAP {x['RAP_atom']:>4s} "
            f"{x['element']:>2s} "
            f"A_dist={x['min_distance_to_A']:5.2f} "
            f"{x['closest_A_residues']}"
        )

    # --------------------------------------------------------
    # 2. Make RFD3 input JSON using selected hotspots
    # --------------------------------------------------------

    hotspots = {}
    hotspots.update(protein_hs)
    hotspots.update(rap_hs)

    spec = {
        "dialect": 2,
        "input": str(target),
        "ligand": "RAP",
        "contig": f"A1-{length},/0,50-80",
        "infer_ori_strategy": "hotspots",
        "select_hotspots": hotspots,
        "is_non_loopy": True,
    }

    outfile = RFD3_DIR / f"{candidate}_binderB.json"

    with open(outfile, "w") as f:
        json.dump(
            {f"{candidate}_binderB": spec},
            f,
            indent=2,
        )

    print("Selected hotspots:", hotspots)
    print("RFD3 input:", outfile)


# ------------------------------------------------------------
# Save hotspot analysis
# ------------------------------------------------------------

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "candidate",
            "RAP_atom",
            "element",
            "min_distance_to_A",
            "polar",
            "geometry_ok",
            "closest_A_residues",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n=== DONE ===")
print("Hotspot CSV:", CSV_OUT)
print("RFD3 inputs:", RFD3_DIR)