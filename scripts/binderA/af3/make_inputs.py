#!/usr/bin/env python3

from pathlib import Path
import json

PROJECT = Path("/home01/a2149a01/RFdiffusion3")
SRC = PROJECT / "outputs/binderA_ligandmpnn_22x4"
OUT = PROJECT / "inputs/binderA/af3/with_rap"

FRB = (
    "RVAILWHEMWHEGLEEASRLYFGERNVKGMFEVLEPLHAMMERGPQTLKETSFNQAY"
    "GRDLMEAQEWCRKYMKSGNVKDLTQAWDLYYHVFRRIS"
)

OUT.mkdir(parents=True, exist_ok=True)


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


count = 0

for fasta in sorted(SRC.glob("*/*.fa")):
    for name, full_seq in read_fasta(fasta):

        if not full_seq.endswith(FRB):
            raise ValueError(f"{name}: sequence does not end with expected FRB")

        binder = full_seq[:-len(FRB)]

        data = {
            "name": f"{name}_plus_RAP",
            "modelSeeds": [1],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": binder,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": []
                    }
                },
                {
                    "protein": {
                        "id": "B",
                        "sequence": FRB,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": []
                    }
                },
                {
                    "ligand": {
                        "id": "C",
                        "ccdCodes": ["RAP"]
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 4
        }

        path = OUT / f"{name}_plus_RAP.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        count += 1

print(f"Generated {count} AF3 inputs")