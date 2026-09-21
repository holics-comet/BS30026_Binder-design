from pathlib import Path
import numpy as np
import pandas as pd

from biotite.structure.io.pdbx import CIFFile, get_structure

ROOT = Path("/home01/a2149a01/RFdiffusion3")
TARGET_DIR = ROOT / "inputs/binderB_targets"
OUT = ROOT / "outputs/binderB_hotspot_candidates.csv"

CONTACT = 4.0
RAP_MIN = 5.0
RAP_MAX = 9.0
A_NEAR_RAP = 10.0

rows = []

for path in sorted(TARGET_DIR.glob("*_A_RAP.cif")):
    candidate = path.name.removesuffix("_A_RAP.cif")

    cif = CIFFile.read(path)
    s = get_structure(cif, model=1)

    A = s[s.chain_id == "A"]
    RAP = s[s.chain_id == "C"]

    A = A[A.element != "H"]
    RAP = RAP[RAP.element != "H"]

    print(f"\n=== {candidate} ===")

    candidate_rows = []

    for r in RAP:
        d = np.linalg.norm(A.coord - r.coord, axis=1)
        min_d = float(d.min())

        # Binder-A residues spatially near this RAP atom
        near = A[d <= A_NEAR_RAP]

        near_res = sorted(set(
            (int(x.res_id), str(x.res_name))
            for x in near
        ))

        # Closest distance for each nearby A residue
        residue_info = []

        for res_id, res_name in near_res:
            res = A[
                (A.res_id == res_id) &
                (A.res_name == res_name)
            ]

            rd = np.linalg.norm(
                res.coord - r.coord,
                axis=1
            ).min()

            residue_info.append(
                (float(rd), res_id, res_name)
            )

        residue_info.sort()

        closest_residues = ";".join(
            f"A{res_id}:{res_name}:{dist:.2f}"
            for dist, res_id, res_name
            in residue_info[:5]
        )

        polar = str(r.element) in {"O", "N", "S"}

        # Geometry window for a composite A:RAP interface
        geometry_ok = RAP_MIN <= min_d <= RAP_MAX

        row = {
            "candidate": candidate,
            "RAP_atom": str(r.atom_name),
            "element": str(r.element),
            "min_distance_to_A": min_d,
            "polar": polar,
            "geometry_ok": geometry_ok,
            "closest_A_residues": closest_residues,
        }

        rows.append(row)

        if geometry_ok:
            candidate_rows.append(row)

    # Polar first, then atoms closest to middle of 5-9 Å window
    candidate_rows.sort(
        key=lambda x: (
            not x["polar"],
            abs(x["min_distance_to_A"] - 7.0)
        )
    )

    for x in candidate_rows[:12]:
        flag = "*" if x["polar"] else " "
        print(
            f"{flag} RAP {x['RAP_atom']:>4s} "
            f"{x['element']:>2s}  "
            f"A_dist={x['min_distance_to_A']:5.2f}  "
            f"{x['closest_A_residues']}"
        )

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print("\nOutput:", OUT)
print("* = polar RAP atom")
