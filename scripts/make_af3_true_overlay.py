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

OUT = ROOT / "outputs/binderA_molstar_review/af3_true_overlay"
OUT.mkdir(parents=True, exist_ok=True)


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_atoms(path):
    with open_text(path) as f:
        lines = f.readlines()

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

        required = [
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]

        if not all(x in col for x in required):
            continue

        seq_key = (
            "_atom_site.auth_seq_id"
            if "_atom_site.auth_seq_id" in col
            else "_atom_site.label_seq_id"
        )

        atoms = []

        while j < len(lines):
            s = lines[j].strip()

            if (
                not s
                or s == "#"
                or s == "loop_"
                or s.startswith("_")
                or s.startswith("data_")
            ):
                break

            try:
                f = shlex.split(s)

                if len(f) < len(columns):
                    j += 1
                    continue

                raw = f[col[seq_key]]
                resid = None if raw in (".", "?") else int(float(raw))

                atoms.append({
                    "element": f[col["_atom_site.type_symbol"]],
                    "atom": f[col["_atom_site.label_atom_id"]],
                    "resname": f[col["_atom_site.label_comp_id"]],
                    "chain": f[col["_atom_site.label_asym_id"]],
                    "resid": resid,
                    "xyz": np.array([
                        float(f[col["_atom_site.Cartn_x"]]),
                        float(f[col["_atom_site.Cartn_y"]]),
                        float(f[col["_atom_site.Cartn_z"]]),
                    ])
                })

            except (ValueError, IndexError):
                pass

            j += 1

        return atoms

    raise RuntimeError(f"No atom_site loop: {path}")


def ca_map(atoms, chain):
    return {
        a["resid"]: a["xyz"]
        for a in atoms
        if a["chain"] == chain
        and a["atom"] == "CA"
        and a["resid"] is not None
    }


def kabsch(mobile, reference):
    mc = mobile.mean(axis=0)
    rc = reference.mean(axis=0)

    P = mobile - mc
    Q = reference - rc

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = rc - mc @ R

    aligned = mobile @ R + t

    rmsd = np.sqrt(
        np.mean(np.sum((aligned - reference) ** 2, axis=1))
    )

    return R, t, rmsd


def write_pdb(path, ref_atoms, af3_atoms, R, t):
    # RFD3 A/B/C remain A/B/C.
    # AF3 A/B/C become D/E/F.
    af3_chain_map = {
        "A": "D",
        "B": "E",
        "C": "F",
    }

    serial = 1

    with open(path, "w") as out:

        out.write("REMARK RFD3: A=Binder B=FRB C=RAP\n")
        out.write("REMARK AF3 : D=Binder E=FRB F=RAP\n")
        out.write("REMARK AF3 aligned to RFD3 using FRB CA atoms\n")

        # ---------------- RFD3 ----------------
        for a in ref_atoms:

            if a["chain"] not in ("A", "B", "C"):
                continue

            x, y, z = a["xyz"]

            record = "HETATM" if a["chain"] == "C" else "ATOM  "
            resid = a["resid"] if a["resid"] is not None else 1

            out.write(
                f"{record}{serial:5d} "
                f"{a['atom']:^4s} "
                f"{a['resname']:>3s} "
                f"{a['chain']:1s}"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          "
                f"{a['element']:>2s}\n"
            )

            serial += 1

        out.write("TER\n")

        # ---------------- AF3 ----------------
        for a in af3_atoms:

            if a["chain"] not in af3_chain_map:
                continue

            q = a["xyz"] @ R + t
            x, y, z = q

            new_chain = af3_chain_map[a["chain"]]

            record = "HETATM" if a["chain"] == "C" else "ATOM  "
            resid = a["resid"] if a["resid"] is not None else 1

            out.write(
                f"{record}{serial:5d} "
                f"{a['atom']:^4s} "
                f"{a['resname']:>3s} "
                f"{new_chain:1s}"
                f"{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          "
                f"{a['element']:>2s}\n"
            )

            serial += 1

        out.write("TER\nEND\n")


ref = read_atoms(REF)
ref_frb = ca_map(ref, "B")

for sample in (0, 4):

    pred_path = (
        AF3_BASE
        / f"seed-1_sample-{sample}"
        / f"5_model_5_b0_d0_plus_RAP_seed-1_sample-{sample}_model.cif"
    )

    pred = read_atoms(pred_path)
    pred_frb = ca_map(pred, "B")

    common = sorted(set(ref_frb) & set(pred_frb))

    mobile = np.array([pred_frb[i] for i in common])
    reference = np.array([ref_frb[i] for i in common])

    R, t, rmsd = kabsch(mobile, reference)

    out_path = OUT / f"RFD3_vs_AF3_sample{sample}_FRB_aligned.pdb"

    write_pdb(out_path, ref, pred, R, t)

    print(
        f"sample {sample}: "
        f"FRB CA={len(common)}, "
        f"alignment RMSD={rmsd:.3f} A"
    )
    print(f"  {out_path}")
