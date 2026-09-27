#!/usr/bin/env python3

import gzip
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

OUTROOT = Path("/scratch/a2149a01/RFdiffusion3_data/outputs")
AF3_ROOT = OUTROOT / "af3_binderA_pilot"
RFD3_ROOT = OUTROOT / "frb_rap_raphotspot_pilot"
OUT_CSV = OUTROOT / "binderA_af3_RAP_metrics_samples.csv"

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
                        "resname": fields[ir],
                        "atom": fields[ia],
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


def rap(atoms):
    return [a for a in atoms if a["resname"].upper() == "RAP"]


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


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def ca_map(atoms, chain_id):
    return {
        a["resnum"]: a["xyz"]
        for a in atoms
        if a["chain"] == chain_id and a["atom"] == "CA"
    }


def matched_ca(ref, pred, chain_id):
    r = ca_map(ref, chain_id)
    p = ca_map(pred, chain_id)
    common = sorted(set(r) & set(p))

    if len(common) < 3:
        return None, None

    return (
        np.stack([r[i] for i in common]),
        np.stack([p[i] for i in common]),
    )


def kabsch(mobile, target):
    mc = mobile.mean(0)
    tc = target.mean(0)

    X = mobile - mc
    Y = target - tc

    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt

    t = tc - mc @ R
    return R, t


def rmsd(A, B):
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def aligned_rmsd(ref, pred, chain_id):
    R0, P0 = matched_ca(ref, pred, chain_id)

    if R0 is None:
        return np.nan

    R, t = kabsch(P0, R0)
    return rmsd(P0 @ R + t, R0)

def target_pose(ref, pred):
    refB, predB = matched_ca(ref, pred, "B")

    if refB is None:
        return np.nan

    R, t = kabsch(predB, refB)

    refA, predA = matched_ca(ref, pred, "A")

    if refA is None:
        return np.nan

    return rmsd(predA @ R + t, refA)


def find_reference(parent):
    hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif.gz"))
    if not hits:
        hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif"))

    pat = re.compile(rf"(^|_){re.escape(parent)}(?=\.|_)")
    hits = [x for x in hits if pat.search(x.name)] or hits

    if len(hits) != 1:
        raise RuntimeError(f"Reference error for {parent}: {hits}")

    return hits[0]


def pair_iptm(summary):
    try:
        return float(summary["chain_pair_iptm"][0][1])
    except Exception:
        return np.nan


def main():
    rows = []
    ref_cache = {}

    dirs = sorted(
        p for p in AF3_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_plus_RAP")
    )

    for n, cdir in enumerate(dirs, 1):
        candidate = cdir.name.removesuffix("_plus_RAP")
        parent = re.sub(r"_b\d+_d\d+$", "", candidate)

        if parent not in ref_cache:
            ref_cache[parent] = parse_cif(find_reference(parent))

        ref = ref_cache[parent]
        refA, refB = chain(ref, "A"), chain(ref, "B")
        ref_frb_interface = contacting_residues(refB, refA)

        for sample in range(5):
            sdir = cdir / f"seed-1_sample-{sample}"
            cif = list(sdir.glob("*_model.cif"))
            js = list(sdir.glob("*_summary_confidences.json"))

            if len(cif) != 1 or len(js) != 1:
                raise RuntimeError(f"{candidate} sample {sample}: files missing")

            pred = parse_cif(cif[0])

            with open(js[0]) as f:
                summary = json.load(f)

            A, B, R = chain(pred, "A"), chain(pred, "B"), rap(pred)

            rap_contact = contacting_residues(A, R)
            frb_contact_on_B = contacting_residues(B, A)
            intended = frb_contact_on_B & INTENDED_FRB_PATCH

            obvious_decoy = int(
                len(frb_contact_on_B) > 0
                and len(rap_contact) == 0
                and len(intended) == 0
            )

            rows.append({
                "candidate": candidate,
                "sample": sample,
                "A_B_iptm": pair_iptm(summary),
                "RAP_contact_res_count": len(rap_contact),
                "intended_FRB_patch_count": len(intended),
                "obvious_wrong_FRB_face_decoy": obvious_decoy,
                "FRB_interface_Jaccard": jaccard(
                    frb_contact_on_B,
                    ref_frb_interface,
                ),
                "BinderFold": aligned_rmsd(ref, pred, "A"),
                "TargetPose": target_pose(ref, pred),
            })

'''
Samples:    440
Candidates: 88
Sample CSV: /home01/a2149a01/RFdiffusion3/outputs/af3_binderA_pilot_metrics_samples.csv
Cand CSV:   /home01/a2149a01/RFdiffusion3/outputs/af3_binderA_pilot_metrics_candidates.csv

=== geometry overview ===
Samples with direct RAP contact: 218 / 440
Samples with intended FRB-patch contact: 167 / 440
Samples with BOTH: 83 / 440
Obvious wrong-face decoys: 138 / 440
'''