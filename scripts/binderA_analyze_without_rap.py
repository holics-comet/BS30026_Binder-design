#!/usr/bin/env python3
import gzip
import json
import shlex
import re
from pathlib import Path

import numpy as np
import csv

OUTROOT = Path("/scratch/a2149a01/RFdiffusion3_data/outputs")
AF3_ROOT = OUTROOT / "af3_binderA_noRAP_a100"
OUT_CSV = OUTROOT / "af3_binderA_noRAP_metrics_samples.csv"

CONTACT = 4.0
INTENDED_FRB_PATCH = {21, 22, 25, 77, 78, 88, 92}


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse_cif(path):
    with open_text(path) as f:
        lines = f.readlines()

    atoms = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        j = i + 1
        headers = []

        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1

        if not headers or not headers[0].startswith("_atom_site."):
            i = j
            continue

        h = {x: k for k, x in enumerate(headers)}

        def idx(*names):
            return next((h[n] for n in names if n in h), None)

        ia = idx("_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        ir = idx("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
        ic = idx("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        inum = idx("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        ix = idx("_atom_site.Cartn_x")
        iy = idx("_atom_site.Cartn_y")
        iz = idx("_atom_site.Cartn_z")
        ie = idx("_atom_site.type_symbol")

        while j < len(lines):
            s = lines[j].strip()

            if not s or s.startswith("#"):
                j += 1
                if s.startswith("#"):
                    break
                continue

            if s == "loop_" or s.startswith("_"):
                break

            fields = shlex.split(s)

            if len(fields) < len(headers):
                j += 1
                continue

            try:
                element = fields[ie].upper() if ie is not None else fields[ia][0].upper()
                if element != "H":
                    atoms.append({
                        "chain": fields[ic],
                        "resnum": int(float(fields[inum])),
                        "xyz": np.array([
                            float(fields[ix]),
                            float(fields[iy]),
                            float(fields[iz]),
                        ]),
                    })
            except Exception:
                pass

            j += 1

        return atoms

    raise RuntimeError(f"No _atom_site loop found: {path}")

def chain(atoms, name):
    return [a for a in atoms if a["chain"] == name]


def coords(atoms):
    return np.stack([a["xyz"] for a in atoms]) if atoms else np.empty((0, 3))


def contacting_residues(source, target, cutoff=CONTACT):
    if not source or not target:
        return set()

    T = coords(target)
    cutoff2 = cutoff * cutoff
    out = set()

    for a in source:
        if np.any(((T - a["xyz"]) ** 2).sum(axis=1) <= cutoff2):
            out.add(a["resnum"])

    return out


def pair_iptm(summary):
    try:
        return float(summary["chain_pair_iptm"][0][1])
    except Exception:
        return np.nan


def main():
    rows = []

    dirs = sorted(
        p for p in AF3_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_no_RAP")
    )

    for n, cdir in enumerate(dirs, 1):
        candidate = cdir.name.removesuffix("_no_RAP")

        for sample in range(5):
            sdir = cdir / f"seed-1_sample-{sample}"
            cif = list(sdir.glob("*_model.cif"))
            js = list(sdir.glob("*_summary_confidences.json"))

            if len(cif) != 1 or len(js) != 1:
                raise RuntimeError(f"{candidate} sample {sample}: files missing")

            pred = parse_cif(cif[0])

            with open(js[0]) as f:
                summary = json.load(f)

            A = chain(pred, "A")
            B = chain(pred, "B")

            frb_contact_on_B = contacting_residues(B, A)
            intended = frb_contact_on_B & INTENDED_FRB_PATCH

            rows.append({
                "candidate": candidate,
                "sample": sample,
                "A_B_iptm": pair_iptm(summary),
                "intended_FRB_patch_count": len(intended),
            })

        print(f"[{n}/{len(dirs)}] {candidate}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUT_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()