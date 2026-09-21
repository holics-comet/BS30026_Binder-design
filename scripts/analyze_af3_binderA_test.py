#!/usr/bin/env python3

import gzip
import math
import shlex
from pathlib import Path
import numpy as np


REF = Path(
    "/home01/a2149a01/RFdiffusion3/outputs/"
    "frb_rap_raphotspot_pilot/"
    "frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_5_model_5.cif.gz"
)

AF3_BASE = Path(
    "/home01/a2149a01/RFdiffusion3/outputs/"
    "af3_binderA_5_model_5_b0_d0/"
    "5_model_5_b0_d0_plus_RAP"
)

B25 = 25
B88 = 88
CONTACT = 4.0
CLASH = 2.0


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_atoms(path):
    with open_text(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        j = i + 1
        columns = []

        while j < len(lines) and lines[j].startswith("_atom_site."):
            columns.append(lines[j].strip())
            j += 1

        if not columns:
            i += 1
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
            i = j
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
                fields = shlex.split(s)

                if len(fields) < len(columns):
                    j += 1
                    continue

                raw = fields[col[seq_key]]
                resid = None if raw in (".", "?") else int(float(raw))

                atoms.append({
                    "element": fields[col["_atom_site.type_symbol"]],
                    "atom": fields[col["_atom_site.label_atom_id"]],
                    "resname": fields[col["_atom_site.label_comp_id"]],
                    "chain": fields[col["_atom_site.label_asym_id"]],
                    "resid": resid,
                    "x": float(fields[col["_atom_site.Cartn_x"]]),
                    "y": float(fields[col["_atom_site.Cartn_y"]]),
                    "z": float(fields[col["_atom_site.Cartn_z"]]),
                })

            except (ValueError, IndexError):
                pass

            j += 1

        return atoms

    raise RuntimeError(f"No atom_site loop: {path}")


def coord(a):
    return np.array([a["x"], a["y"], a["z"]], dtype=float)


def dist(a, b):
    return np.linalg.norm(coord(a) - coord(b))


def heavy(atoms):
    return [a for a in atoms if a["element"].upper() != "H"]


def chain(atoms, c):
    return [a for a in atoms if a["chain"] == c]


def ca_map(atoms, c):
    return {
        a["resid"]: coord(a)
        for a in atoms
        if a["chain"] == c
        and a["atom"] == "CA"
        and a["resid"] is not None
    }


def kabsch(P, Q):
    P = np.asarray(P)
    Q = np.asarray(Q)

    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)

    X = P - pc
    Y = Q - qc

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = qc - pc @ R

    aligned = P @ R + t
    rmsd = np.sqrt(np.mean(np.sum((aligned - Q) ** 2, axis=1)))

    return R, t, rmsd


def paired_ca(ref, pred, c):
    r = ca_map(ref, c)
    p = ca_map(pred, c)

    ids = sorted(set(r) & set(p))

    return (
        np.array([p[i] for i in ids]),
        np.array([r[i] for i in ids]),
        ids,
    )


def distance_matrix_rmsd(P, Q):
    DP = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    DQ = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)

    iu = np.triu_indices(len(P), k=1)

    return np.sqrt(np.mean((DP[iu] - DQ[iu]) ** 2))


def transform_atoms(atoms, R, t):
    out = []

    for a in atoms:
        b = dict(a)
        v = coord(a) @ R + t
        b["x"], b["y"], b["z"] = v
        out.append(b)

    return out


def rmsd_matched_atoms(pred, ref):
    ref_map = {
        (a["resid"], a["atom"]): coord(a)
        for a in ref
    }

    P, Q = [], []

    for a in pred:
        key = (a["resid"], a["atom"])
        if key in ref_map:
            P.append(coord(a))
            Q.append(ref_map[key])

    if not P:
        return float("nan")

    P = np.array(P)
    Q = np.array(Q)

    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def min_distance(A, B):
    if not A or not B:
        return float("nan")

    return min(dist(a, b) for a in A for b in B)


def contact_residues(binder, target, cutoff=CONTACT):
    result = set()

    for a in binder:
        if a["resid"] is None:
            continue

        if any(dist(a, b) <= cutoff for b in target):
            result.add(a["resid"])

    return result


def clashes(A, B, cutoff=CLASH):
    return sum(
        dist(a, b) < cutoff
        for a in A
        for b in B
    )


ref = read_atoms(REF)

ref_A = heavy(chain(ref, "A"))
ref_B = heavy(chain(ref, "B"))
ref_C = heavy(chain(ref, "C"))

print(
    "sample FRB_RMSD FRB_dRMSD BinderFold "
    "TargetPose RAPpose A-RAP RAPres FRBres "
    "B25 B88 bridge clash"
)

for s in range(5):

    files = list(
        (AF3_BASE / f"seed-1_sample-{s}").glob("*_model.cif")
    )

    if len(files) != 1:
        raise RuntimeError(f"sample {s}: model CIF not found")

    pred = read_atoms(files[0])

    A = heavy(chain(pred, "A"))
    B = heavy(chain(pred, "B"))
    C = heavy(chain(pred, "C"))

    # --------------------------------------------------------
    # 1. FRB fold recovery
    # --------------------------------------------------------

    PB, QB, idsB = paired_ca(ref, pred, "B")

    Rb, tb, frb_rmsd = kabsch(PB, QB)
    frb_drmsd = distance_matrix_rmsd(PB, QB)

    # --------------------------------------------------------
    # 2. Binder fold recovery
    # --------------------------------------------------------

    PA, QA, idsA = paired_ca(ref, pred, "A")

    Ra, ta, binder_fold = kabsch(PA, QA)

    # --------------------------------------------------------
    # 3. Align entire AF3 complex using FRB
    # --------------------------------------------------------

    pred_aligned = transform_atoms(pred, Rb, tb)

    A_al = heavy(chain(pred_aligned, "A"))
    C_al = heavy(chain(pred_aligned, "C"))

    pred_A_ca = ca_map(pred_aligned, "A")
    ref_A_ca = ca_map(ref, "A")

    commonA = sorted(set(pred_A_ca) & set(ref_A_ca))

    P_pose = np.array([pred_A_ca[i] for i in commonA])
    Q_pose = np.array([ref_A_ca[i] for i in commonA])

    target_pose = np.sqrt(
        np.mean(np.sum((P_pose - Q_pose) ** 2, axis=1))
    )

    # RAP atom names are matched after FRB alignment.
    rap_pose = rmsd_matched_atoms(C_al, ref_C)

    # --------------------------------------------------------
    # 4. Interfaces in AF3 prediction
    # --------------------------------------------------------

    rap_contacts = contact_residues(A, C)
    frb_contacts = contact_residues(A, B)
    bridge = rap_contacts & frb_contacts

    b25 = [
        x for x in B
        if x["resid"] == B25
    ]

    b88 = [
        x for x in B
        if x["resid"] == B88
    ]

    d25 = min_distance(A, b25)
    d88 = min_distance(A, b88)

    nclash = (
        clashes(A, B)
        + clashes(A, C)
        + clashes(B, C)
    )

    print(
        f"{s:>6} "
        f"{frb_rmsd:>8.2f} "
        f"{frb_drmsd:>10.2f} "
        f"{binder_fold:>10.2f} "
        f"{target_pose:>10.2f} "
        f"{rap_pose:>7.2f} "
        f"{min_distance(A,C):>5.2f} "
        f"{len(rap_contacts):>6} "
        f"{len(frb_contacts):>6} "
        f"{d25:>5.1f} "
        f"{d88:>5.1f} "
        f"{len(bridge):>6} "
        f"{nclash:>5}"
    )
