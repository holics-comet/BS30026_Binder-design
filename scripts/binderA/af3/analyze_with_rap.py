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
AF3_ROOT = PROJECT / "outputs/af3_binderA_pilot"
RFD3_ROOT = PROJECT / "outputs/frb_rap_raphotspot_pilot"
OUT_SAMPLE = PROJECT / "outputs/af3_binderA_pilot_metrics_samples.csv"
OUT_CAND = PROJECT / "outputs/af3_binderA_pilot_metrics_candidates.csv"

CONTACT = 4.0
CLASH = 2.0

# RFD3 output numbering
# native FRB: 2038,2039,2042,2094,2095,2105,2109
# output FRB: 21,22,25,77,78,88,92
INTENDED_FRB_PATCH = {21, 22, 25, 77, 78, 88, 92}

VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
    "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75,
}
PROBE = 1.4
N_SASA_POINTS = 96


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_cif_atoms(path):
    """
    Minimal mmCIF _atom_site parser.
    Returns list of dicts:
    chain, resnum, resname, atom, element, xyz
    """
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
                elem = fields[ix_elem].upper() if ix_elem is not None else atom[0].upper()

                if resnum_raw in (".", "?"):
                    resnum = None
                else:
                    resnum = int(float(resnum_raw))

                # Ignore hydrogens
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


def rap_atoms(atoms):
    # Current AF3/RFD3 convention: RAP = chain C.
    x = [a for a in atoms if a["chain"] == "C" and a["resname"].upper() == "RAP"]
    if not x:
        x = [a for a in atoms if a["resname"].upper() == "RAP"]
    return x


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


def contacting_residues(source, target, cutoff=4.0):
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


def contacting_atom_names(source, target, cutoff=4.0):
    """
    Atom names in source that contact target.
    Used with source=RAP.
    """
    if not source or not target:
        return set()

    T = coords(target)
    cutoff2 = cutoff * cutoff
    out = set()

    for a in source:
        d2 = ((T - a["xyz"]) ** 2).sum(axis=1)
        if np.any(d2 <= cutoff2):
            out.add(a["atom"])

    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def count_intercomponent_clashes(A, B, cutoff=2.0):
    if not A or not B:
        return 0
    X, Y = coords(A), coords(B)
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    return int(np.sum(d2 < cutoff * cutoff))


def ca_by_res(atoms, chain):
    d = {}
    for a in atoms:
        if a["chain"] == chain and a["atom"] == "CA" and a["resnum"] is not None:
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
    """
    Find R,t such that mobile @ R + t ~= target.
    """
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
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


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

    D1 = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    D2 = np.sqrt(((pred[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2))
    return float(np.sqrt(np.mean((D1 - D2) ** 2)))


def transform_atoms(atoms, R, t):
    out = []
    for a in atoms:
        b = dict(a)
        b["xyz"] = a["xyz"] @ R + t
        out.append(b)
    return out


def target_frame_metrics(ref_atoms, pred_atoms):
    refB, predB = matched_ca(ref_atoms, pred_atoms, "B")
    if len(refB) < 3:
        return float("nan"), float("nan")

    R, t = kabsch(predB, refB)
    pred_aligned = transform_atoms(pred_atoms, R, t)

    # Binder pose after aligning only FRB
    refA, predA = matched_ca(ref_atoms, pred_aligned, "A")
    target_pose = rmsd(predA, refA) if len(refA) >= 3 else float("nan")

    # RAP atom-name matched RMSD
    rr = {a["atom"]: a["xyz"] for a in rap_atoms(ref_atoms)}
    rp = {a["atom"]: a["xyz"] for a in rap_atoms(pred_aligned)}
    common = sorted(set(rr) & set(rp))

    if common:
        X = np.stack([rp[x] for x in common])
        Y = np.stack([rr[x] for x in common])
        rap_pose = rmsd(X, Y)
    else:
        rap_pose = float("nan")

    return target_pose, rap_pose


def atom_named(atoms, name):
    return [a for a in atoms if a["atom"] == name]


def fibonacci_sphere(n):
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        pts.append([math.cos(theta) * r, y, math.sin(theta) * r])
    return np.array(pts)


SPHERE = fibonacci_sphere(N_SASA_POINTS)


def element_radius(e):
    e = e.upper()
    return VDW.get(e, 1.70)


def sasa_of_selected(selected, environment):
    """
    Shrake-Rupley approximation.
    SASA of selected atoms in presence of environment.
    """
    if not selected:
        return float("nan")

    env_xyz = coords(environment)
    env_r = np.array([element_radius(a["element"]) + PROBE for a in environment])

    total = 0.0

    for atom in selected:
        r = element_radius(atom["element"]) + PROBE
        surface = atom["xyz"] + SPHERE * r

        # Environment excludes the atom itself by coordinate/identity approximation.
        d2 = ((surface[:, None, :] - env_xyz[None, :, :]) ** 2).sum(axis=2)
        buried = d2 < (env_r[None, :] ** 2)

        # Ignore self occlusion: points inside sphere centered on this exact atom.
        same = np.linalg.norm(env_xyz - atom["xyz"], axis=1) < 1e-5
        if np.any(same):
            buried[:, same] = False

        accessible = ~np.any(buried, axis=1)
        frac = accessible.mean()
        total += 4.0 * math.pi * r * r * frac

    return total


def rap_exposure_metrics(atoms):
    rap = rap_atoms(atoms)
    if not rap:
        return float("nan"), float("nan"), 0

    # RAP alone
    sasa_free = sasa_of_selected(rap, rap)

    # RAP in full ternary complex
    sasa_complex = sasa_of_selected(rap, atoms)

    rel = sasa_complex / sasa_free if sasa_free > 0 else float("nan")

    proteins = chain_atoms(atoms, "A") + chain_atoms(atoms, "B")
    P = coords(proteins)

    exposed_atoms = 0
    if len(P):
        for a in rap:
            d = np.sqrt(((P - a["xyz"]) ** 2).sum(axis=1))
            if d.min() > CONTACT:
                exposed_atoms += 1

    return sasa_complex, rel, exposed_atoms


def find_rfd3_reference(parent):
    hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif.gz"))
    if not hits:
        hits = list(RFD3_ROOT.rglob(f"*{parent}*.cif"))

    # Prefer exact ending around parent if multiple names happen to match.
    if len(hits) > 1:
        exactish = [p for p in hits if parent in p.name]
        hits = exactish or hits

    if not hits:
        raise FileNotFoundError(f"RFD3 reference not found for {parent}")

    # Avoid 1_model_1 accidentally matching 1_model_10-like names.
    pat = re.compile(rf"(^|_){re.escape(parent)}(?=\.|_)")
    better = [p for p in hits if pat.search(p.name)]
    if better:
        hits = better

    if len(hits) != 1:
        raise RuntimeError(f"Ambiguous RFD3 reference for {parent}: {hits}")

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


def fmt_atoms(x):
    return ";".join(sorted(x))


def analyze_sample(candidate, sample, cif, summary, ref_atoms):
    pred = parse_cif_atoms(cif)

    A = chain_atoms(pred, "A")
    B = chain_atoms(pred, "B")
    RAP = rap_atoms(pred)

    refA = chain_atoms(ref_atoms, "A")
    refB = chain_atoms(ref_atoms, "B")
    refR = rap_atoms(ref_atoms)

    # Current AF3 contacts
    rap_res = contacting_residues(A, RAP)
    frb_res = contacting_residues(A, B)

    rap_atom_contacts = contacting_atom_names(RAP, A)
    intended_contacts = frb_res & INTENDED_FRB_PATCH

    # Designed RFD3 interfaces
    ref_rap_res = contacting_residues(refA, refR)
    ref_frb_res = contacting_residues(refA, refB)
    ref_rap_atoms = contacting_atom_names(refR, refA)

    # Interface recovery
    frb_j = jaccard(frb_res, ref_frb_res)
    rap_atom_j = jaccard(rap_atom_contacts, ref_rap_atoms)

    # O3/O8
    o3 = min_dist(A, atom_named(RAP, "O3"))
    o8 = min_dist(A, atom_named(RAP, "O8"))

    # FRB B25/B88
    B25 = [x for x in B if x["resnum"] == 25]
    B88 = [x for x in B if x["resnum"] == 88]
    d25 = min_dist(A, B25)
    d88 = min_dist(A, B88)

    # Bridge residues = same Binder residue contacts RAP and FRB
    bridge = rap_res & frb_res

    # clashes
    clashes = (
        count_intercomponent_clashes(A, B)
        + count_intercomponent_clashes(A, RAP)
        + count_intercomponent_clashes(B, RAP)
    )

    # Structural recovery
    frb_rmsd = aligned_rmsd(ref_atoms, pred, "B")
    frb_drmsd = drmsd(ref_atoms, pred, "B")
    binder_fold = aligned_rmsd(ref_atoms, pred, "A")
    target_pose, rap_pose = target_frame_metrics(ref_atoms, pred)

    # Exposure
    rap_sasa, rap_rel_sasa, rap_exposed_atoms = rap_exposure_metrics(pred)

    # confidence
    ranking = summary.get("ranking_score", float("nan"))
    iptm = summary.get("iptm", float("nan"))
    ptm = summary.get("ptm", float("nan"))
    disorder = summary.get("fraction_disordered", float("nan"))
    has_clash = summary.get("has_clash", float("nan"))

    cp = summary.get("chain_pair_iptm", [])
    ab_iptm = get_pair(cp, 0, 1)
    arap_iptm = get_pair(cp, 0, 2)
    brap_iptm = get_pair(cp, 1, 2)

    # Decoy indicator deliberately based on geometry, not AF3 confidence.
    #
    # Strongest obvious case:
    # Binder contacts FRB but essentially does not contact RAP and also misses
    # the intended FRB patch.
    obvious_wrong_face = int(
        len(frb_res) > 0
        and len(rap_res) == 0
        and len(intended_contacts) == 0
    )

    return {
        "candidate": candidate,
        "parent": re.sub(r"_b\d+_d\d+$", "", candidate),
        "sample": sample,

        "ranking_score": ranking,
        "iptm": iptm,
        "ptm": ptm,
        "A_B_iptm": ab_iptm,
        "A_RAP_iptm": arap_iptm,
        "B_RAP_iptm": brap_iptm,
        "fraction_disordered": disorder,
        "AF3_has_clash": has_clash,

        "FRB_RMSD": frb_rmsd,
        "FRB_dRMSD": frb_drmsd,
        "BinderFold": binder_fold,
        "TargetPose": target_pose,
        "RAPpose": rap_pose,

        "A_RAP_min_dist": min_dist(A, RAP),
        "RAP_contact_res_count": len(rap_res),
        "FRB_contact_res_count": len(frb_res),
        "bridge_count": len(bridge),

        "O3_dist": o3,
        "O8_dist": o8,
        "B25_dist": d25,
        "B88_dist": d88,

        "FRB_interface_Jaccard": frb_j,
        "RAP_atom_interface_Jaccard": rap_atom_j,

        "intended_FRB_patch_count": len(intended_contacts),
        "intended_FRB_patch": fmt_set(intended_contacts),
        "FRB_contact_residues": fmt_set(frb_res),
        "RFD3_FRB_contact_residues": fmt_set(ref_frb_res),

        "RAP_contact_atoms": fmt_atoms(rap_atom_contacts),
        "RFD3_RAP_contact_atoms": fmt_atoms(ref_rap_atoms),

        "RAP_SASA_A2": rap_sasa,
        "RAP_relative_SASA": rap_rel_sasa,
        "RAP_exposed_atom_count": rap_exposed_atoms,

        "intercomponent_clashes_lt2A": clashes,
        "obvious_wrong_FRB_face_decoy": obvious_wrong_face,
    }


def safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def summarize_candidate(candidate, rows):
    """
    No arbitrary final score.
    Summarize reproducibility and best observed designed-state geometry.
    """
    def vals(key):
        return [v for r in rows if (v := safe_float(r[key])) is not None]

    def minv(key):
        x = vals(key)
        return min(x) if x else float("nan")

    def maxv(key):
        x = vals(key)
        return max(x) if x else float("nan")

    def meanv(key):
        x = vals(key)
        return float(np.mean(x)) if x else float("nan")

    # Geometry-only descriptive counts, not final selection thresholds.
    rap_contact_samples = sum(r["RAP_contact_res_count"] > 0 for r in rows)
    intended_patch_samples = sum(r["intended_FRB_patch_count"] > 0 for r in rows)
    both_samples = sum(
        r["RAP_contact_res_count"] > 0 and r["intended_FRB_patch_count"] > 0
        for r in rows
    )
    decoy_samples = sum(r["obvious_wrong_FRB_face_decoy"] for r in rows)

    return {
        "candidate": candidate,
        "parent": rows[0]["parent"],
        "n_samples": len(rows),

        "samples_with_RAP_contact": rap_contact_samples,
        "samples_with_intended_FRB_patch": intended_patch_samples,
        "samples_with_both": both_samples,
        "obvious_decoy_samples": decoy_samples,

        "best_BinderFold": minv("BinderFold"),
        "mean_BinderFold": meanv("BinderFold"),

        "best_TargetPose": minv("TargetPose"),
        "best_RAPpose": minv("RAPpose"),

        "best_A_RAP_min_dist": minv("A_RAP_min_dist"),
        "max_RAP_contact_res_count": maxv("RAP_contact_res_count"),
        "max_FRB_contact_res_count": maxv("FRB_contact_res_count"),
        "max_bridge_count": maxv("bridge_count"),

        "max_FRB_interface_Jaccard": maxv("FRB_interface_Jaccard"),
        "max_RAP_atom_interface_Jaccard": maxv("RAP_atom_interface_Jaccard"),
        "max_intended_FRB_patch_count": maxv("intended_FRB_patch_count"),

        "max_RAP_relative_SASA": maxv("RAP_relative_SASA"),
        "mean_RAP_relative_SASA": meanv("RAP_relative_SASA"),

        "max_ranking_score": maxv("ranking_score"),
        "max_iptm": maxv("iptm"),
        "max_A_B_iptm": maxv("A_B_iptm"),
        "max_A_RAP_iptm": maxv("A_RAP_iptm"),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    candidate_dirs = sorted(
        p for p in AF3_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_plus_RAP")
    )

    if len(candidate_dirs) != 88:
        print(f"WARNING: expected 88 candidate dirs, found {len(candidate_dirs)}")

    all_rows = []
    cand_rows = []

    ref_cache = {}

    for n, cdir in enumerate(candidate_dirs, 1):
        candidate = cdir.name.removesuffix("_plus_RAP")
        parent = re.sub(r"_b\d+_d\d+$", "", candidate)

        if parent not in ref_cache:
            ref_path = find_rfd3_reference(parent)
            ref_cache[parent] = parse_cif_atoms(ref_path)

        ref_atoms = ref_cache[parent]
        rows = []

        for s in range(5):
            sdir = cdir / f"seed-1_sample-{s}"

            cif_hits = list(sdir.glob("*_model.cif"))
            summary_hits = list(sdir.glob("*_summary_confidences.json"))

            if len(cif_hits) != 1 or len(summary_hits) != 1:
                raise RuntimeError(
                    f"{candidate} sample {s}: "
                    f"CIF={len(cif_hits)}, summary={len(summary_hits)}"
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

        cand_rows.append(summarize_candidate(candidate, rows))

        if n % 10 == 0 or n == len(candidate_dirs):
            print(f"[{n}/{len(candidate_dirs)}]")

    if len(all_rows) != 440:
        raise RuntimeError(f"Expected 440 samples, got {len(all_rows)}")

    write_csv(OUT_SAMPLE, all_rows)
    write_csv(OUT_CAND, cand_rows)

    print()
    print(f"Samples:    {len(all_rows)}")
    print(f"Candidates: {len(cand_rows)}")
    print(f"Sample CSV: {OUT_SAMPLE}")
    print(f"Cand CSV:   {OUT_CAND}")

    print()
    print("=== geometry overview ===")
    print(
        "Samples with direct RAP contact:",
        sum(r["RAP_contact_res_count"] > 0 for r in all_rows),
        "/ 440"
    )
    print(
        "Samples with intended FRB-patch contact:",
        sum(r["intended_FRB_patch_count"] > 0 for r in all_rows),
        "/ 440"
    )
    print(
        "Samples with BOTH:",
        sum(
            r["RAP_contact_res_count"] > 0
            and r["intended_FRB_patch_count"] > 0
            for r in all_rows
        ),
        "/ 440"
    )
    print(
        "Obvious wrong-face decoys:",
        sum(r["obvious_wrong_FRB_face_decoy"] for r in all_rows),
        "/ 440"
    )


if __name__ == "__main__":
    main()
