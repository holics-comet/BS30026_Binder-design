from pathlib import Path
import csv
import numpy as np

from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure

ROOT = Path("/home01/a2149a01/RFdiffusion3")
AF3 = ROOT / "outputs/af3_binderA_pilot"
OUT = ROOT / "inputs/binderB/targets"
CSV_OUT = ROOT / "outputs/binderB_rap_surface_analysis.csv"

TARGETS = {
    "3_model_3_b2_d0": 3,
    "6_model_5_b3_d0": 4,
    "9_model_4_b0_d0": 1,
    "5_model_1_b0_d0": 3,
    "1_model_7_b0_d0": 4,
}

OUT.mkdir(parents=True, exist_ok=True)

rows = []

for candidate, sample in TARGETS.items():
    sample_dir = (
        AF3
        / f"{candidate}_plus_RAP"
        / f"seed-1_sample-{sample}"
    )

    files = list(sample_dir.glob("*_model.cif"))
    if len(files) != 1:
        raise RuntimeError(
            f"{candidate}: expected 1 model CIF, found {len(files)}"
        )

    src = files[0]

    cif = CIFFile.read(src)
    structure = get_structure(
        cif,
        model=1,
        include_bonds=True,
        extra_fields=["atom_id"],
    )

    # AF3 chains:
    # A = Binder A
    # B = FRB
    # C = RAP
    chains = sorted(set(structure.chain_id))
    print(f"\n=== {candidate} sample {sample} ===")
    print("Input:", src)
    print("Chains:", chains)

    binder = structure[structure.chain_id == "A"]
    rap = structure[structure.chain_id == "C"]

    if len(binder) == 0:
        raise RuntimeError(f"{candidate}: chain A not found")
    if len(rap) == 0:
        raise RuntimeError(f"{candidate}: RAP chain C not found")

    # Keep only Binder A + RAP
    target = structure[
        (structure.chain_id == "A") |
        (structure.chain_id == "C")
    ]

    outfile = OUT / f"{candidate}_A_RAP.cif"

    out_cif = CIFFile()
    set_structure(out_cif, target)
    out_cif.write(outfile)

    # Heavy atoms only
    binder_heavy = binder[binder.element != "H"]
    rap_heavy = rap[rap.element != "H"]

    # Minimum distance of every RAP atom to Binder A
    for atom in rap_heavy:
        distances = np.linalg.norm(
            binder_heavy.coord - atom.coord,
            axis=1,
        )
        min_dist = float(np.min(distances))

        rows.append({
            "candidate": candidate,
            "sample": sample,
            "RAP_atom": atom.atom_name,
            "element": atom.element,
            "min_distance_to_A": min_dist,
        })

    print("Binder atoms:", len(binder))
    print("RAP atoms:", len(rap))
    print("Output:", outfile)

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "candidate",
            "sample",
            "RAP_atom",
            "element",
            "min_distance_to_A",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n=== DONE ===")
print("Targets:", len(TARGETS))
print("Surface CSV:", CSV_OUT)
