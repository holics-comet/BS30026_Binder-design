#!/usr/bin/env python3

import csv
import gzip
import json
import math
import re
import shlex
from pathlib import Path

import numpy as np


PROJECT = Path("/home01/a2149a01/RFdiffusion3")
AF3_ROOT = PROJECT / "outputs/af3_binderA_noRAP_a100"
RFD3_ROOT = PROJECT / "outputs/frb_rap_raphotspot_pilot"

OUT_SAMPLE = PROJECT / "outputs/af3_binderA_noRAP_metrics_samples.csv"
OUT_CAND = PROJECT / "outputs/af3_binderA_noRAP_metrics_candidates.csv"

CONTACT = 4.0
CLASH = 2.0

# RFD3 output FRB numbering
# native: 2038,2039,2042,2094,2095,2105,2109
# output:   21,  22,  25,  77,  78,  88,  92
INTENDED_FRB_PATCH = {21, 22, 25, 77, 78, 88, 92}


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_cif_atoms(path):
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

        hmap = {h: k for k, h in enumerate(headers)}

        def idx(*names):
            for n in names:
                if n in hmap:
                    return hmap[n]
            return None

        ix_group = idx("_atom_site.group_PDB")
        ix_atom = idx("_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        ix_res = idx("_atom_site.auth_comp_id", "_atom_site.label_comp_id")
        ix_chain = idx("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        ix_num = idx("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        ix_x = idx("_atom_site.Cartn_x")
        ix_y = idx("_atom_site.Cartn_y")
        ix_z = idx("_atom_site.Cartn_z")
        ix_elem = idx("_atom_site.type_symbol")

        while j < len(lines):
            s = lines[j].strip()

            if not s or s.startswith("#"):
                j += 1
                if s.startswith("#"):
                    break
                continue

            if s == "loop_" or s.startswith("_"):
                break

            try:
                fields = shlex.split(s)
            except ValueError:
                j += 1
                continue

            if len(fields) < len(headers):
                j += 1
                continue

            try:
                group = fields[ix_group] if ix_group is not None else "ATOM"
                atom = fields[ix_atom]
                resname = fields[ix_res]
                chain = fields[ix_chain]
                resnum_raw = fields[ix_num]

                x = float(fields[ix_x])
                y = float(fields[ix_y])
                z = float(fields[ix_z])

                elem = (
                    fields[ix_elem].upper()
                    if ix_elem is not None
                    else atom[0].upper()
                )

                if resnum_raw in (".", "?"):
                    resnum = None
                else:
                    resnum = int(float(resnum_raw))

                if elem != "H":
                    atoms.append({
                        "group": group,
                        "chain": chain,
                        "resnum": resnum,
                        "resname": resname,
                        "atom": atom,
                        "element": elem,
                        "xyz": np.array([x, y, z], dtype=float),
                    })

            except Exception:
                pass

            j += 1

        return atoms

    raise RuntimeError(f"No _atom_site loop found: {path}")


def chain_atoms(atoms, chain):
    return [a for a in atoms if a["chain"] == chain]


def coords(atoms):
    if not atoms:
        return np.empty((0, 3))
    return np.stack([a["xyz"] for a in atoms])


def min_dist(A, B):
    A = coords(A)
    B = coords(B)

    if len(A) == 0 or len(B) == 0:
        return float("nan")

    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def contacting_residues(source, target, cutoff=CONTACT):
    if not source or not target:
        return set()

    T = coords(target)
    cutoff2 = cutoff * cutoff
    out = set()

    for a in source:
        d2 = ((T - a["xyz"]) ** 2).sum(axis=1)
        if np.any(d2 <= cutoff2):
            if a["resnum"] is not None:
                out.add(a["resnum"])

    return out


def contact_residue_pairs(A, B, cutoff=CONTACT):
    """Unique residue-residue pairs having at least one heavy-atom contact."""
    if not A or not B:
        return set()

    out = set()
    cutoff2 = cutoff * cutoff

    B_xyz = coords(B)

    for a in A:
        if a["resnum"] is None:
            continue

        d2 = ((B_xyz - a["xyz"]) ** 2).sum(axis=1)
        hits = np.where(d2 <= cutoff2)[0]

        for i in hits:
            b = B[i]
            if b["resnum"] is not None:
                out.add((a["resnum"], b["resnum"]))

    return out


def count_intercomponent_clashes(A, B, cutoff=CLASH):
    if not A or not B:
        return 0

    X = coords(A)
    Y = coords(B)

    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return int(np.sum(d2 < cutoff * cutoff))


def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a and not b:
        return float("nan")

    return len(a & b) / len(a | b)


def ca_by_res(atoms, chain):
    d = {}

    for a in atoms:
        if (
            a["chain"] == chain
            and a["atom"] == "CA"
            and a["resnum"] is not None
        ):
            d[a["resnum"]] = a["xyz"]

    return d


def matched_ca(ref_atoms, pred_atoms, chain):
    R = ca_by_res(ref_atoms, chain)
    P = ca_by_res(pred_atoms, chain)

    common = sorted(set(R) & set(P))

    if not common:
        return np.empty((0, 3)), np.empty((0, 3))

    return (
        np.stack([R[i] for i in common]),
        np.stack([P[i] for i in common]),
    )


def kabsch(mobile, target):
    pc = mobile.mean(axis=0)
    qc = target.mean(axis=0)

    X = mobile - pc
    Y = target - qc

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = qc - pc @ R

    return R, t


def rmsd(A, B):
    if len(A) == 0 or len(A) != len(B):
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                np.sum((A - B) ** 2, axis=1)
            )
        )
    )


def aligned_rmsd(ref_atoms, pred_atoms, chain):
    ref, pred = matched_ca(ref_atoms, pred_atoms, chain)

    if len(ref) < 3:
        return float("nan")

    R, t = kabsch(pred, ref)

    return rmsd(pred @ R + t, ref)


def drmsd(ref_atoms, pred_atoms, chain):
    ref, pred = matched_ca(ref_atoms, pred_atoms, chain)

    if len(ref) < 3:
        return float("nan")

    D1 = np.sqrt(
        ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    )

    D2 = np.sqrt(
        ((pred[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2)
    )

    return float(np.sqrt(np.mean((D1 - D2) ** 2)))


def transform_atoms(atoms, R, t):
    out = []

    for a in atoms:
        b = dict(a)
        b["xyz"] = a["xyz"] @ R + t
        out.append(b)

    return out


def target_pose_metric(ref_atoms, pred_atoms):
    """
    Align predicted FRB onto RFD3 FRB, then measure Binder A CA RMSD
    without independently aligning Binder A.
    """
    refB, predB = matched_ca(ref_atoms, pred_atoms, "B")

    if len(refB) < 3:
        return float("nan")

    R, t = kabsch(predB, refB)
    pred_aligned = transform_atoms(pred_atoms, R, t)

    refA, predA = matched_ca(ref_atoms, pred_aligned, "A")

    if len(refA) < 3:
        return float("nan")

    return rmsd(predA, refA)


def find_rfd3_reference(parent):
    hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif.gz"))

    if not hits:
        hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif"))

    if len(hits) > 1:
        exactish = [p for p in hits if parent in p.name]
        hits = exactish or hits

    if not hits:
        raise FileNotFoundError(
            f"RFD3 reference not found for {parent}"
        )

    pat = re.compile(
        rf"(^|_){re.escape(parent)}(?=\.|_)"
    )

    better = [
        p for p in hits
        if pat.search(p.name)
    ]

    if better:
        hits = better

    if len(hits) != 1:
        raise RuntimeError(
            f"Ambiguous RFD3 reference for {parent}: {hits}"
        )

    return hits[0]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def get_pair(matrix, i, j):
    try:
        return float(matrix[i][j])
    except Exception:
        return float("nan")


def fmt_set(x):
    return ";".join(map(str, sorted(x)))


def safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def analyze_sample(candidate, sample, cif, summary, ref_atoms):
    pred = parse_cif_atoms(cif)

    A = chain_atoms(pred, "A")
    B = chain_atoms(pred, "B")

    refA = chain_atoms(ref_atoms, "A")
    refB = chain_atoms(ref_atoms, "B")

    # Current A-FRB interface
    a_contact_res = contacting_residues(A, B)
    frb_contact_res = contacting_residues(B, A)
    residue_pairs = contact_residue_pairs(A, B)

    intended_contacts = frb_contact_res & INTENDED_FRB_PATCH

    # Designed RFD3 interface
    ref_a_contact_res = contacting_residues(refA, refB)
    ref_frb_contact_res = contacting_residues(refB, refA)

    # Interface recovery
    a_interface_j = jaccard(a_contact_res, ref_a_contact_res)
    frb_interface_j = jaccard(
        frb_contact_res,
        ref_frb_contact_res,
    )

    # Structure
    frb_rmsd = aligned_rmsd(ref_atoms, pred, "B")
    frb_drmsd = drmsd(ref_atoms, pred, "B")
    binder_fold = aligned_rmsd(ref_atoms, pred, "A")
    target_pose = target_pose_metric(ref_atoms, pred)

    # Geometry
    ab_min = min_dist(A, B)
    clashes = count_intercomponent_clashes(A, B)

    # Confidence
    ranking = summary.get("ranking_score", float("nan"))
    iptm = summary.get("iptm", float("nan"))
    ptm = summary.get("ptm", float("nan"))
    disorder = summary.get("fraction_disordered", float("nan"))
    has_clash = summary.get("has_clash", float("nan"))

    cp = summary.get("chain_pair_iptm", [])
    ab_iptm = get_pair(cp, 0, 1)

    return {
        "candidate": candidate,
        "parent": re.sub(r"_b\d+_d\d+$", "", candidate),
        "sample": sample,

        "ranking_score": ranking,
        "iptm": iptm,
        "ptm": ptm,
        "A_B_iptm": ab_iptm,
        "fraction_disordered": disorder,
        "AF3_has_clash": has_clash,

        "FRB_RMSD": frb_rmsd,
        "FRB_dRMSD": frb_drmsd,
        "BinderFold": binder_fold,
        "TargetPose": target_pose,

        "A_B_min_dist": ab_min,

        "A_contact_res_count": len(a_contact_res),
        "FRB_contact_res_count": len(frb_contact_res),
        "AB_contact_residue_pair_count": len(residue_pairs),

        "intended_FRB_patch_count": len(intended_contacts),
        "intended_FRB_patch": fmt_set(intended_contacts),

        "A_contact_residues": fmt_set(a_contact_res),
        "FRB_contact_residues": fmt_set(frb_contact_res),

        "RFD3_A_contact_residues": fmt_set(ref_a_contact_res),
        "RFD3_FRB_contact_residues": fmt_set(ref_frb_contact_res),

        "A_interface_Jaccard": a_interface_j,
        "FRB_interface_Jaccard": frb_interface_j,

        "intercomponent_clashes_lt2A": clashes,
    }


def summarize_candidate(candidate, rows):
    def vals(key):
        return [
            v for r in rows
            if (v := safe_float(r[key])) is not None
        ]

    def minv(key):
        x = vals(key)
        return min(x) if x else float("nan")

    def maxv(key):
        x = vals(key)
        return max(x) if x else float("nan")

    def meanv(key):
        x = vals(key)
        return float(np.mean(x)) if x else float("nan")

    # Purely descriptive counts.
    # No binding/non-binding cutoff is imposed here.
    contact_samples = sum(
        r["A_contact_res_count"] > 0
        and r["FRB_contact_res_count"] > 0
        for r in rows
    )

    intended_samples = sum(
        r["intended_FRB_patch_count"] > 0
        for r in rows
    )

    return {
        "candidate": candidate,
        "parent": rows[0]["parent"],
        "n_samples": len(rows),

        "samples_with_AB_contact": contact_samples,
        "samples_with_intended_FRB_patch": intended_samples,

        "min_A_B_min_dist": minv("A_B_min_dist"),
        "mean_A_B_min_dist": meanv("A_B_min_dist"),

        "max_A_contact_res_count": maxv("A_contact_res_count"),
        "mean_A_contact_res_count": meanv("A_contact_res_count"),

        "max_FRB_contact_res_count": maxv("FRB_contact_res_count"),
        "mean_FRB_contact_res_count": meanv("FRB_contact_res_count"),

        "max_AB_contact_residue_pair_count":
            maxv("AB_contact_residue_pair_count"),
        "mean_AB_contact_residue_pair_count":
            meanv("AB_contact_residue_pair_count"),

        "max_intended_FRB_patch_count":
            maxv("intended_FRB_patch_count"),
        "mean_intended_FRB_patch_count":
            meanv("intended_FRB_patch_count"),

        "max_A_interface_Jaccard":
            maxv("A_interface_Jaccard"),
        "mean_A_interface_Jaccard":
            meanv("A_interface_Jaccard"),

        "max_FRB_interface_Jaccard":
            maxv("FRB_interface_Jaccard"),
        "mean_FRB_interface_Jaccard":
            meanv("FRB_interface_Jaccard"),

        "best_BinderFold": minv("BinderFold"),
        "mean_BinderFold": meanv("BinderFold"),

        "best_TargetPose": minv("TargetPose"),
        "mean_TargetPose": meanv("TargetPose"),

        "max_ranking_score": maxv("ranking_score"),
        "mean_ranking_score": meanv("ranking_score"),

        "max_iptm": maxv("iptm"),
        "mean_iptm": meanv("iptm"),

        "max_A_B_iptm": maxv("A_B_iptm"),
        "mean_A_B_iptm": meanv("A_B_iptm"),
    }


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


def main():
    candidate_dirs = sorted(
        p for p in AF3_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_no_RAP")
    )

    if len(candidate_dirs) != 88:
        print(
            f"WARNING: expected 88 candidate dirs, "
            f"found {len(candidate_dirs)}"
        )

    all_rows = []
    cand_rows = []

    ref_cache = {}

    for n, cdir in enumerate(candidate_dirs, 1):
        candidate = cdir.name.removesuffix("_no_RAP")
        parent = re.sub(r"_b\d+_d\d+$", "", candidate)

        if parent not in ref_cache:
            ref_path = find_rfd3_reference(parent)
            ref_cache[parent] = parse_cif_atoms(ref_path)

        ref_atoms = ref_cache[parent]
        rows = []

        for s in range(5):
            sdir = cdir / f"seed-1_sample-{s}"

            cif_hits = list(sdir.glob("*_model.cif"))
            summary_hits = list(
                sdir.glob("*_summary_confidences.json")
            )

            if len(cif_hits) != 1 or len(summary_hits) != 1:
                raise RuntimeError(
                    f"{candidate} sample {s}: "
                    f"CIF={len(cif_hits)}, "
                    f"summary={len(summary_hits)}"
                )

            summary = load_summary(summary_hits[0])

            row = analyze_sample(
                candidate,
                s,
                cif_hits[0],
                summary,
                ref_atoms,
            )

            rows.append(row)
            all_rows.append(row)

        cand_rows.append(
            summarize_candidate(candidate, rows)
        )

        if n % 10 == 0 or n == len(candidate_dirs):
            print(f"[{n}/{len(candidate_dirs)}]")

    if len(all_rows) != 440:
        raise RuntimeError(
            f"Expected 440 samples, got {len(all_rows)}"
        )

    write_csv(OUT_SAMPLE, all_rows)
    write_csv(OUT_CAND, cand_rows)

    print()
    print(f"Samples:    {len(all_rows)}")
    print(f"Candidates: {len(cand_rows)}")
    print(f"Sample CSV: {OUT_SAMPLE}")
    print(f"Cand CSV:   {OUT_CAND}")

    print()
    print("=== no-RAP overview ===")

    print(
        "Samples with any A-FRB contact:",
        sum(
            r["A_contact_res_count"] > 0
            for r in all_rows
        ),
        "/ 440",
    )

    print(
        "Samples contacting intended FRB patch:",
        sum(
            r["intended_FRB_patch_count"] > 0
            for r in all_rows
        ),
        "/ 440",
    )

    print()
    print("A-B pair ipTM:")
    x = [
        r["A_B_iptm"]
        for r in all_rows
        if safe_float(r["A_B_iptm"]) is not None
    ]

    if x:
        print(f"  mean   = {np.mean(x):.3f}")
        print(f"  median = {np.median(x):.3f}")
        print(f"  min    = {np.min(x):.3f}")
        print(f"  max    = {np.max(x):.3f}")


if __name__ == "__main__":
    main()