#!/usr/bin/env python3

import gzip
import shlex
from pathlib import Path
import numpy as np

ROOT = Path("/home01/a2149a01/RFdiffusion3")

REF = ROOT / (
    "outputs/frb_rap_raphotspot_pilot/"
    "frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_5_model_5.cif.gz"
)

AF3_BASE = ROOT / (
    "outputs/af3_binderA_5_model_5_b0_d0/"
    "5_model_5_b0_d0_plus_RAP"
)

OUT = ROOT / "outputs/binderA_molstar_review/af3_5_model_5"
OUT.mkdir(parents=True, exist_ok=True)


def read_text(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.readlines()
    with open(path) as f:
        return f.readlines()


def atom_site_info(lines):
    """Find atom_site loop and return column mapping + data-line indices."""

    for i, line in enumerate(lines):
        if line.strip() != "loop_":
            continue

        j = i + 1
        columns = []

        while j < len(lines) and lines[j].startswith("_atom_site."):
            columns.append(lines[j].strip())
            j += 1

        if not columns:
            continue

        col = {name: k for k, name in enumerate(columns)}

        required = {
            "_atom_site.label_atom_id",
            "_atom_site.label_asym_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        }

        if not required.issubset(col):
            continue

        rows = []

        k = j
        while k < len(lines):
            s = lines[k].strip()

            if not s or s == "#" or s == "loop_" or s.startswith("_"):
                break

            fields = shlex.split(s)

            if len(fields) >= len(columns):
                rows.append((k, fields))

            k += 1

        return columns, col, rows

    raise RuntimeError("No usable _atom_site loop found")


def get_frb_ca(lines):
    columns, col, rows = atom_site_info(lines)

    seq_key = (
        "_atom_site.auth_seq_id"
        if "_atom_site.auth_seq_id" in col
        else "_atom_site.label_seq_id"
    )

    result = {}

    for _, f in rows:
        chain = f[col["_atom_site.label_asym_id"]]
        atom = f[col["_atom_site.label_atom_id"]]

        if chain != "B" or atom != "CA":
            continue

        resid_raw = f[col[seq_key]]
        if resid_raw in (".", "?"):
            continue

        resid = int(float(resid_raw))

        result[resid] = np.array([
            float(f[col["_atom_site.Cartn_x"]]),
            float(f[col["_atom_site.Cartn_y"]]),
            float(f[col["_atom_site.Cartn_z"]]),
        ])

    return result


def kabsch(mobile, reference):
    """
    Row-vector convention.

    Find R, t such that:
        aligned = mobile @ R + t
    """

    mob_cent = mobile.mean(axis=0)
    ref_cent = reference.mean(axis=0)

    P = mobile - mob_cent
    Q = reference - ref_cent

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    # Prevent reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = ref_cent - mob_cent @ R

    aligned = mobile @ R + t

    rmsd = np.sqrt(
        np.mean(
            np.sum((aligned - reference) ** 2, axis=1)
        )
    )

    return R, t, rmsd

def transform_cif(lines, R, t, output_path):
    columns, col, rows = atom_site_info(lines)

    xidx = col["_atom_site.Cartn_x"]
    yidx = col["_atom_site.Cartn_y"]
    zidx = col["_atom_site.Cartn_z"]

    row_lookup = {idx: fields for idx, fields in rows}

    output = []

    for i, line in enumerate(lines):

        if i not in row_lookup:
            output.append(line)
            continue

        fields = row_lookup[i]

        p = np.array([
            float(fields[xidx]),
            float(fields[yidx]),
            float(fields[zidx]),
        ])

        q = p @ R + t

        fields[xidx] = f"{q[0]:.3f}"
        fields[yidx] = f"{q[1]:.3f}"
        fields[zidx] = f"{q[2]:.3f}"

        # mmCIF whitespace-separated atom_site row
        output.append(" ".join(fields) + "\n")

    with open(output_path, "w") as f:
        f.writelines(output)


# ------------------------------------------------------------
# Reference
# ------------------------------------------------------------

ref_lines = read_text(REF)
ref_frb = get_frb_ca(ref_lines)

# Also write decompressed reference for Mol*
ref_out = OUT / "RFD3_5_model_5_reference.cif"

with open(ref_out, "w") as f:
    f.writelines(ref_lines)


# ------------------------------------------------------------
# AF3 samples
# ------------------------------------------------------------

for sample in (0, 4):

    pred_path = (
        AF3_BASE
        / f"seed-1_sample-{sample}"
        / f"5_model_5_b0_d0_plus_RAP_seed-1_sample-{sample}_model.cif"
    )

    pred_lines = read_text(pred_path)
    pred_frb = get_frb_ca(pred_lines)

    common = sorted(set(ref_frb) & set(pred_frb))

    mobile = np.array([pred_frb[r] for r in common])
    reference = np.array([ref_frb[r] for r in common])

    R, t, rmsd = kabsch(mobile, reference)

    out_path = OUT / f"AF3_sample{sample}_FRB_aligned.cif"

    transform_cif(pred_lines, R, t, out_path)

    print(
        f"sample {sample}: "
        f"FRB CA atoms={len(common)}, "
        f"FRB alignment RMSD={rmsd:.3f} A"
    )

print()
print("Output:")
print(OUT)
