#!/usr/bin/env python3

from pathlib import Path
import json
import shutil

PROJECT = Path("/home01/a2149a01/RFdiffusion3")
SRC = PROJECT / "outputs/binderB_ligandmpnn_50x4"

WITH_RAP = PROJECT / "inputs/binderB/af3/with_rap"
WITHOUT_RAP = PROJECT / "inputs/binderB/af3/without_rap"

A_LENGTHS = {
    "1_model_7_b0_d0": 51,
    "3_model_3_b2_d0": 75,
    "5_model_1_b0_d0": 62,
    "6_model_5_b3_d0": 67,
    "9_model_4_b0_d0": 55,
}


def read_fasta(path):
    records = []
    name = None
    seq = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq)))

            name = line[1:].split(",")[0]
            seq = []
        else:
            seq.append(line)

    if name is not None:
        records.append((name, "".join(seq)))

    return records


def protein(chain_id, sequence):
    return {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": [],
        }
    }


def make_input(name, binder_a, binder_b, with_rap):
    sequences = [
        protein("A", binder_a),
        protein("B", binder_b),
    ]

    if with_rap:
        sequences.append({
            "ligand": {
                "id": "C",
                "ccdCodes": ["RAP"],
            }
        })

    suffix = "plus_RAP" if with_rap else "no_RAP"

    return {
        "name": f"{name}_{suffix}",
        "modelSeeds": [1],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }


def main():
    # Avoid stale JSONs from previous runs
    for out in (WITH_RAP, WITHOUT_RAP):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

    n = 0

    for fasta in sorted(SRC.glob("*/*.fa")):
        for name, full_seq in read_fasta(fasta):

            target = next(
                (t for t in A_LENGTHS if name.startswith(t + "__")),
                None,
            )

            if target is None:
                raise ValueError(f"Unknown target: {name}")

            a_len = A_LENGTHS[target]

            binder_a = full_seq[:a_len]
            binder_b = full_seq[a_len:]

            if len(binder_a) != a_len:
                raise ValueError(f"{name}: invalid Binder A length")

            if not 50 <= len(binder_b) <= 80:
                raise ValueError(
                    f"{name}: Binder B length = {len(binder_b)}"
                )

            plus = make_input(
                name, binder_a, binder_b, True
            )

            minus = make_input(
                name, binder_a, binder_b, False
            )

            (WITH_RAP / f"{name}_plus_RAP.json").write_text(
                json.dumps(plus, indent=2) + "\n"
            )

            (WITHOUT_RAP / f"{name}_no_RAP.json").write_text(
                json.dumps(minus, indent=2) + "\n"
            )

            n += 1

    print(f"Designs: {n}")
    print(f"with RAP: {len(list(WITH_RAP.glob('*.json')))}")
    print(f"without RAP: {len(list(WITHOUT_RAP.glob('*.json')))}")


if __name__ == "__main__":
    main()
