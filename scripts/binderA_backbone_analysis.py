#!/usr/bin/env python3

import argparse
import csv
import gzip
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Project / I/O
# -----------------------------------------------------------------------------
# Expected location:
#   RFdiffusion3/scripts/binderA/backbone/analyze_backbone_quality.py
PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = Path(
    "/scratch/a2149a01/RFdiffusion3_data/outputs/frb_rap_raphotspot_pilot"
)

DEFAULT_OUTPUT_CSV = Path(
    "/scratch/a2149a01/RFdiffusion3_data/outputs/binderA_backbone_quality.csv"
)

DEFAULT_SELECTED_DIR = Path(
    "/scratch/a2149a01/RFdiffusion3_data/outputs/binderA_ligandmpnn_pilot2"
)

# -----------------------------------------------------------------------------
# Analysis thresholds
# -----------------------------------------------------------------------------
CONTACT_CUTOFF = 4.0
CLASH_CUTOFF = 2.0
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Backbone sanity checks. These are QC windows, not energy functions.
CA_CA_IDEAL = 3.80
CA_CA_TOL = 0.25
PEPTIDE_CN_IDEAL = 1.33
PEPTIDE_CN_TOL = 0.15

# Selection settings inherited from select_candidates.py
MAX_PER_CLASS = 3

# RFD3 output numbering
# Original FRB B2042 ARG -> output B25
# Original FRB B2105 TYR -> output B88
TARGET_DEFS = {
    "RAP_O3": {"chain": "C", "resname": "RAP", "resid": None, "atoms": {"O3"}},
    "RAP_O8": {"chain": "C", "resname": "RAP", "resid": None, "atoms": {"O8"}},
    "B25_ARG": {"chain": "B", "resname": "ARG", "resid": 25, "atoms": {"NH1", "NH2"}},
    "B88_TYR": {"chain": "B", "resname": "TYR", "resid": 88, "atoms": {"OH"}},
}


# -----------------------------------------------------------------------------
# CIF parsing / basic helpers
# -----------------------------------------------------------------------------
def read_cif(path):
    """Read heavy atoms from the RFD3 mmCIF output."""
    atoms = []
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(("ATOM ", "HETATM ")):
                continue

            fields = line.split()
            if len(fields) < 21:
                continue

            element = fields[1]
            atom_name = fields[2]
            resname = fields[4]
            chain = fields[5]
            resid = int(fields[9])

            if element == "H" or atom_name.startswith("H"):
                continue

            atoms.append({
                "element": element,
                "atom": atom_name,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "xyz": np.array(
                    [float(fields[18]), float(fields[19]), float(fields[20])],
                    dtype=float,
                ),
            })

    return atoms


def get_atoms(atoms, chain=None, resname=None, resid=None, atom_names=None):
    selected = []
    for a in atoms:
        if chain is not None and a["chain"] != chain:
            continue
        if resname is not None and a["resname"] != resname:
            continue
        if resid is not None and a["resid"] != resid:
            continue
        if atom_names is not None and a["atom"] not in atom_names:
            continue
        selected.append(a)
    return selected


def min_distance(group1, group2):
    if not group1 or not group2:
        return np.nan
    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])
    return float(np.linalg.norm(xyz1[:, None, :] - xyz2[None, :, :], axis=2).min())


def count_atom_contacts(group1, group2, cutoff):
    if not group1 or not group2:
        return 0
    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])
    dist = np.linalg.norm(xyz1[:, None, :] - xyz2[None, :, :], axis=2)
    return int(np.sum(dist <= cutoff))


def residue_list_string(residues):
    return ";".join(str(x) for x in residues)


def float_or_blank(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return float(x)

# -----------------------------------------------------------------------------
# Interface geometry metrics (merged from analyze_geometry.py)
# -----------------------------------------------------------------------------
def contacting_binder_residues(binder_bb, target, cutoff):
    if not binder_bb or not target:
        return []

    target_xyz = np.array([a["xyz"] for a in target])
    residues = defaultdict(list)
    for atom in binder_bb:
        residues[atom["resid"]].append(atom)

    contacted = []
    for resid, residue_atoms in residues.items():
        xyz = np.array([a["xyz"] for a in residue_atoms])
        dist = np.linalg.norm(xyz[:, None, :] - target_xyz[None, :, :], axis=2)
        if np.any(dist <= cutoff):
            contacted.append(resid)
    return sorted(contacted)


def contacted_target_residues(target, binder_bb, cutoff):
    if not target or not binder_bb:
        return []

    binder_xyz = np.array([a["xyz"] for a in binder_bb])
    residues = defaultdict(list)
    for atom in target:
        residues[(atom["resid"], atom["resname"])].append(atom)

    contacted = []
    for key, residue_atoms in residues.items():
        xyz = np.array([a["xyz"] for a in residue_atoms])
        dist = np.linalg.norm(xyz[:, None, :] - binder_xyz[None, :, :], axis=2)
        if np.any(dist <= cutoff):
            contacted.append(key)
    return sorted(contacted)


def contacted_target_atoms(target, binder_bb, cutoff):
    if not target or not binder_bb:
        return []

    binder_xyz = np.array([a["xyz"] for a in binder_bb])
    contacted = []
    for atom in target:
        d = np.linalg.norm(binder_xyz - atom["xyz"], axis=1)
        if np.any(d <= cutoff):
            contacted.append(atom["atom"])
    return sorted(set(contacted))


# -----------------------------------------------------------------------------
# Sidechain-reach descriptors (merged from analyze_sidechain_reach.py)
# -----------------------------------------------------------------------------
def group_binder_residues(binder_all):
    residues = {}
    for atom in binder_all:
        resid = atom["resid"]
        if resid not in residues:
            residues[resid] = {"resname": atom["resname"], "atoms": []}
        residues[resid]["atoms"].append(atom)
    return residues


def find_atom(residue_atoms, atom_name):
    for atom in residue_atoms:
        if atom["atom"] == atom_name:
            return atom
    return None


def distance_to_target(point_atom, target_atoms):
    if point_atom is None or not target_atoms:
        return np.nan
    xyz = np.array([a["xyz"] for a in target_atoms])
    return float(np.linalg.norm(xyz - point_atom["xyz"], axis=1).min())


def closest_residue(residues, target_atoms, mode="CA"):
    best = None

    for resid, data in residues.items():
        if mode == "CA":
            reference_atom = find_atom(data["atoms"], "CA")
        elif mode == "CBpref":
            reference_atom = find_atom(data["atoms"], "CB")
            if reference_atom is None:
                reference_atom = find_atom(data["atoms"], "CA")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if reference_atom is None:
            continue

        d = distance_to_target(reference_atom, target_atoms)
        if np.isnan(d):
            continue

        if best is None or d < best["distance"]:
            best = {
                "resid": resid,
                "resname": data["resname"],
                "atom": reference_atom["atom"],
                "distance": d,
            }

    return best

def add_reach_metrics(result, residues, target_name, target_atoms):
    ca = closest_residue(residues, target_atoms, mode="CA")
    cb = closest_residue(residues, target_atoms, mode="CBpref")

    result[f"{target_name}_CA_min_distance"] = "" if ca is None else ca["distance"]
    result[f"{target_name}_CA_closest_resid"] = "" if ca is None else ca["resid"]
    result[f"{target_name}_CA_closest_resname"] = "" if ca is None else ca["resname"]

    result[f"{target_name}_CBpref_min_distance"] = "" if cb is None else cb["distance"]
    result[f"{target_name}_CBpref_closest_resid"] = "" if cb is None else cb["resid"]
    result[f"{target_name}_CBpref_closest_resname"] = "" if cb is None else cb["resname"]
    result[f"{target_name}_CBpref_reference_atom"] = "" if cb is None else cb["atom"]


# -----------------------------------------------------------------------------
# Binder backbone integrity metrics
# -----------------------------------------------------------------------------
def backbone_integrity_metrics(residues):
    """
    Simple chain sanity metrics for generated Binder A.

    These do NOT measure fold confidence. They only flag local backbone geometry
    problems such as abnormal consecutive CA spacing or peptide C-N spacing.
    """
    ordered = sorted(residues.items())

    ca_distances = []
    cn_distances = []
    ca_bad = 0
    cn_bad = 0
    chain_breaks = 0

    for (_, data_i), (_, data_j) in zip(ordered[:-1], ordered[1:]):
        ca_i = find_atom(data_i["atoms"], "CA")
        ca_j = find_atom(data_j["atoms"], "CA")
        c_i = find_atom(data_i["atoms"], "C")
        n_j = find_atom(data_j["atoms"], "N")

        if ca_i is not None and ca_j is not None:
            d_ca = float(np.linalg.norm(ca_i["xyz"] - ca_j["xyz"]))
            ca_distances.append(d_ca)
            if abs(d_ca - CA_CA_IDEAL) > CA_CA_TOL:
                ca_bad += 1
            if d_ca > 4.5:
                chain_breaks += 1

        if c_i is not None and n_j is not None:
            d_cn = float(np.linalg.norm(c_i["xyz"] - n_j["xyz"]))
            cn_distances.append(d_cn)
            if abs(d_cn - PEPTIDE_CN_IDEAL) > PEPTIDE_CN_TOL:
                cn_bad += 1

    def summary(values):
        if not values:
            return (np.nan, np.nan, np.nan, np.nan)
        arr = np.asarray(values, dtype=float)
        return (float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()))

    ca_mean, ca_std, ca_min, ca_max = summary(ca_distances)
    cn_mean, cn_std, cn_min, cn_max = summary(cn_distances)

    n_pairs = max(len(ordered) - 1, 1)

    return {
        "bb_CA_CA_mean": float_or_blank(ca_mean),
        "bb_CA_CA_std": float_or_blank(ca_std),
        "bb_CA_CA_min": float_or_blank(ca_min),
        "bb_CA_CA_max": float_or_blank(ca_max),
        "bb_CA_CA_outlier_count": ca_bad,
        "bb_CA_CA_outlier_fraction": ca_bad / n_pairs,
        "bb_peptide_CN_mean": float_or_blank(cn_mean),
        "bb_peptide_CN_std": float_or_blank(cn_std),
        "bb_peptide_CN_min": float_or_blank(cn_min),
        "bb_peptide_CN_max": float_or_blank(cn_max),
        "bb_peptide_CN_outlier_count": cn_bad,
        "bb_peptide_CN_outlier_fraction": cn_bad / n_pairs,
        "bb_chain_break_count": chain_breaks,
    }

# -----------------------------------------------------------------------------
# Per-structure analysis
# -----------------------------------------------------------------------------
def analyze_structure(path):
    atoms = read_cif(path)

    binder_all = get_atoms(atoms, chain="A")
    binder_bb = [a for a in binder_all if a["atom"] in BACKBONE_ATOMS]
    frb = get_atoms(atoms, chain="B")
    rap = get_atoms(atoms, chain="C", resname="RAP")
    residues = group_binder_residues(binder_all)

    binder_resids = sorted(residues)
    binder_length = len(binder_resids)

    rap_binder_res = contacting_binder_residues(binder_bb, rap, CONTACT_CUTOFF)
    frb_binder_res = contacting_binder_residues(binder_bb, frb, CONTACT_CUTOFF)
    rap_set = set(rap_binder_res)
    frb_set = set(frb_binder_res)
    bridge = sorted(rap_set & frb_set)
    interface = sorted(rap_set | frb_set)

    rap_target_atoms = contacted_target_atoms(rap, binder_bb, CONTACT_CUTOFF)
    frb_target_res = contacted_target_residues(frb, binder_bb, CONTACT_CUTOFF)

    target_atoms = {}
    for name, spec in TARGET_DEFS.items():
        target_atoms[name] = get_atoms(
            atoms,
            chain=spec["chain"],
            resname=spec["resname"],
            resid=spec["resid"],
            atom_names=spec["atoms"],
        )

    rap_clashes = count_atom_contacts(binder_bb, rap, CLASH_CUTOFF)
    frb_clashes = count_atom_contacts(binder_bb, frb, CLASH_CUTOFF)

    result = {
        "design": path.name.replace(".cif.gz", "").replace(".cif", ""),
        "binder_length": binder_length,

        # Binder backbone integrity
        **backbone_integrity_metrics(residues),

        # Binder backbone <-> RAP
        "bb_rap_min_distance": min_distance(binder_bb, rap),
        "bb_rap_atom_contacts_4A": count_atom_contacts(binder_bb, rap, CONTACT_CUTOFF),
        "bb_rap_contact_residues": residue_list_string(rap_binder_res),
        "bb_rap_contact_residue_count": len(rap_binder_res),
        "rap_atoms_contacted_by_bb": ";".join(rap_target_atoms),
        "rap_atoms_contacted_by_bb_count": len(rap_target_atoms),

        # Binder backbone <-> FRB
        "bb_frb_min_distance": min_distance(binder_bb, frb),
        "bb_frb_atom_contacts_4A": count_atom_contacts(binder_bb, frb, CONTACT_CUTOFF),
        "bb_frb_contact_residues": residue_list_string(frb_binder_res),
        "bb_frb_contact_residue_count": len(frb_binder_res),
        "frb_residues_contacted_by_bb": ";".join(f"{r}:{n}" for r, n in frb_target_res),
        "frb_residues_contacted_by_bb_count": len(frb_target_res),

        # Composite interface
        "bb_bridge_residues": residue_list_string(bridge),
        "bb_bridge_residue_count": len(bridge),
        "bb_interface_residues": residue_list_string(interface),
        "bb_interface_residue_count": len(interface),
        "bb_interface_fraction": (len(interface) / binder_length) if binder_length else 0.0,
        "bb_contact_balance": min(len(rap_binder_res), len(frb_binder_res)),

        # Target clashes
        "bb_rap_clashes_lt2A": rap_clashes,
        "bb_frb_clashes_lt2A": frb_clashes,
        "bb_total_clashes_lt2A": rap_clashes + frb_clashes,
    }

    # Direct backbone hotspot distances/contact flags + reach metrics
    for name, atoms_group in target_atoms.items():
        bb_min = min_distance(binder_bb, atoms_group)
        result[f"bb_{name}_min_distance"] = bb_min
        result[f"bb_{name}_contact_4A"] = bool(not np.isnan(bb_min) and bb_min <= CONTACT_CUTOFF)
        add_reach_metrics(result, residues, name, atoms_group)

    return result

    return "far"


def len_class(n):
    if n <= 59:
        return "50-59"
    if n <= 69:
        return "60-69"
    return "70-80"

def geom_class(d):
    if d <= 5:
        return "near"
    if d <= 8:
        return "mid"
    return "far"

def add_selection_fields(records):
    for r in records:
        r["length_class"] = len_class(r["binder_length"])

        o8 = float(r["RAP_O8_CBpref_min_distance"])
        b88 = float(r["B88_TYR_CBpref_min_distance"])
        r["O8_class"] = geom_class(o8)
        r["B88_class"] = geom_class(b88)

        # Preserve the original select_candidates.py hard filter exactly.
        r["hard_pass"] = (
            r["bb_total_clashes_lt2A"] == 0
            and r["bb_rap_contact_residue_count"] >= 1
            and r["bb_frb_contact_residue_count"] >= 1
        )

        reasons = []
        if r["bb_total_clashes_lt2A"] != 0:
            reasons.append("target_clash")
        if r["bb_rap_contact_residue_count"] < 1:
            reasons.append("no_RAP_contact")
        if r["bb_frb_contact_residue_count"] < 1:
            reasons.append("no_FRB_contact")
        r["hard_fail_reason"] = ";".join(reasons)

        # New backbone-local geometry QC. Kept separate so existing selection
        # behavior is not changed silently.
        r["backbone_integrity_pass"] = (
            r["bb_chain_break_count"] == 0
            and r["bb_CA_CA_outlier_count"] == 0
            and r["bb_peptide_CN_outlier_count"] == 0
        )
        r["overall_qc_pass"] = r["hard_pass"] and r["backbone_integrity_pass"]

        r["selected"] = False
        r["selection_reason"] = ""
        r["diversity_group"] = f"{r['length_class']}|O8:{r['O8_class']}|B88:{r['B88_class']}"

    passed = [r for r in records if r["hard_pass"]]
    selected_names = set()

    # 1) Preserve every bridge-containing structure.
    for r in passed:
        if r["bb_bridge_residue_count"] >= 1:
            r["selected"] = True
            r["selection_reason"] = "bridge"
            selected_names.add(r["design"])

    # 2) Diversity selection: length x O8 x B88.
    groups = defaultdict(list)
    for r in passed:
        groups[(r["length_class"], r["O8_class"], r["B88_class"])].append(r)

    # Soft rank is intentionally lexicographic rather than a weighted black-box score.
    # Priority: bridge > interface fraction > balanced RAP/FRB contact > hotspot reach.
    global_sorted = sorted(
        passed,
        key=lambda r: (
            -r["bb_bridge_residue_count"],
            -r["bb_interface_fraction"],
            -r["bb_contact_balance"],
            float(r["RAP_O8_CBpref_min_distance"]) + float(r["B88_TYR_CBpref_min_distance"]),
        ),
    )
    for rank, r in enumerate(global_sorted, start=1):
        r["soft_rank"] = rank

    for r in records:
        if not r["hard_pass"]:
            r["soft_rank"] = ""

    for _, members in groups.items():
        members.sort(
            key=lambda r: (
                -r["bb_bridge_residue_count"],
                -r["bb_interface_fraction"],
                -r["bb_contact_balance"],
                float(r["RAP_O8_CBpref_min_distance"]) + float(r["B88_TYR_CBpref_min_distance"]),
            )
        )

        group_rank = 0
        added = 0
        for r in members:
            group_rank += 1
            r["group_rank"] = group_rank

            if r["design"] in selected_names:
                continue
            r["selected"] = True
            r["selection_reason"] = "diversity"
            selected_names.add(r["design"])
            added += 1
            if added >= MAX_PER_CLASS:
                break

        # assign group ranks to any entries after early break
        for r in members:
            if "group_rank" not in r:
                # rank based on final sorted order
                r["group_rank"] = members.index(r) + 1

    for r in records:
        if "group_rank" not in r:
            r["group_rank"] = ""

    return passed, [r for r in records if r["selected"]]
# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def write_csv(records, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def copy_selected(selected, input_dir, selected_dir):
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for r in selected:
        for ext in (".cif.gz", ".cif"):
            src = input_dir / f"{r['design']}{ext}"
            if src.is_file():
                shutil.copy2(src, selected_dir)
                copied += 1
                break
        else:
            print(f"WARNING: file not found: {r['design']}")
    return copied

def print_summary(records, passed, selected, output_csv, selected_dir, copied):
    print("\n=== BACKBONE QC SUMMARY ===")
    print(f"Analyzed:       {len(records)}")
    print(f"Hard PASS:      {len(passed)}")
    print(f"Hard FAIL:      {len(records) - len(passed)}")
    print(f"Selected:       {len(selected)}")

    print("\nHard filter:")
    print("  target clashes <2A == 0")
    print("  RAP contact residues >= 1")
    print("  FRB contact residues >= 1")
    print("  (selection hard filter is unchanged from the original code)")

    integrity_passed = sum(bool(r["backbone_integrity_pass"]) for r in records)
    overall_passed = sum(bool(r["overall_qc_pass"]) for r in records)
    print("\nAdditional backbone integrity QC:")
    print(f"  integrity PASS: {integrity_passed}/{len(records)}")
    print(f"  overall QC PASS: {overall_passed}/{len(records)}")

    print("\nSoft ranking priority:")
    print("  1. bridge residues (higher is better)")
    print("  2. interface fraction (higher is better)")
    print("  3. min(RAP-contact residues, FRB-contact residues) (higher is better)")
    print("  4. O8 CBpref distance + B88 CBpref distance (lower is better)")

    print("\n=== TOP HARD-PASS DESIGNS ===")
    top = sorted(
        passed,
        key=lambda r: int(r["soft_rank"]),
    )[:20]

    print(
        f"{'rank':>4s} {'design':55s} {'len':>3s} {'RAP':>3s} {'FRB':>3s} "
        f"{'br':>2s} {'intF':>5s} {'O8':>5s} {'B88':>5s} {'CAout':>5s} {'sel':>3s}"
    )
    for r in top:
        print(
            f"{r['soft_rank']:>4} {r['design'][:55]:55s} {r['binder_length']:3d} "
            f"{r['bb_rap_contact_residue_count']:3d} {r['bb_frb_contact_residue_count']:3d} "
            f"{r['bb_bridge_residue_count']:2d} {r['bb_interface_fraction']:5.3f} "
            f"{float(r['RAP_O8_CBpref_min_distance']):5.2f} "
            f"{float(r['B88_TYR_CBpref_min_distance']):5.2f} "
            f"{r['bb_CA_CA_outlier_count']:5d} {str(r['selected']):>3s}"
        )

    print(f"\nCSV written to:\n{output_csv}")
    print(f"\nCopied {copied} selected structures to:\n{selected_dir}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Binder A backbone geometry, reach, QC, ranking, and selection analysis."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy selected CIF files into the selected output directory.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    files = sorted(args.input_dir.glob("*.cif.gz")) + sorted(args.input_dir.glob("*.cif"))
    # Avoid duplicate .cif when a .cif.gz with the same design exists.
    unique = {}
    for path in files:
        design = path.name.replace(".cif.gz", "").replace(".cif", "")
        unique.setdefault(design, path)
    files = [unique[k] for k in sorted(unique)]

    if not files:
        raise RuntimeError(f"No .cif/.cif.gz files found in: {args.input_dir}")

    print(f"Found {len(files)} structures in {args.input_dir}")

    records = []
    for i, path in enumerate(files, start=1):
        r = analyze_structure(path)
        records.append(r)
        print(
            f"[{i:3d}/{len(files)}] {r['design']} | "
            f"RAP={r['bb_rap_contact_residue_count']} "
            f"FRB={r['bb_frb_contact_residue_count']} "
            f"bridge={r['bb_bridge_residue_count']} "
            f"clash={r['bb_total_clashes_lt2A']} "
            f"break={r['bb_chain_break_count']} "
            f"O8reach={float(r['RAP_O8_CBpref_min_distance']):.2f} "
            f"B88reach={float(r['B88_TYR_CBpref_min_distance']):.2f}"
        )

    passed, selected = add_selection_fields(records)
    write_csv(records, args.output_csv)

    copied = 0
    if not args.no_copy:
        copied = copy_selected(selected, args.input_dir, args.selected_dir)

    print_summary(records, passed, selected, args.output_csv, args.selected_dir, copied)


if __name__ == "__main__":
    main()


'''
=== BACKBONE QC SUMMARY ===
Analyzed:       80
Hard PASS:      69
Hard FAIL:      11
Selected:       22

Hard filter:
  target clashes <2A == 0
  RAP contact residues >= 1
  FRB contact residues >= 1
  (selection hard filter is unchanged from the original code)

Additional backbone integrity QC:
  integrity PASS: 80/80
  overall QC PASS: 69/80

Soft ranking priority:
  1. bridge residues (higher is better)
  2. interface fraction (higher is better)
  3. min(RAP-contact residues, FRB-contact residues) (higher is better)
  4. O8 CBpref distance + B88 CBpref distance (lower is better)

=== TOP HARD-PASS DESIGNS ===
rank design                                                  len RAP FRB br  intF    O8   B88 CAout sel
   1 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_3_m  75   5   6  2 0.120  8.10  4.05     0 True
   2 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_4_m  54   7   6  1 0.222  9.76 12.61     0 True
   3 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_9_m  55   4   4  1 0.127  4.93 10.84     0 True
   4 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_7_m  72   7   3  1 0.125  4.65  8.31     0 True
   5 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_6_m  67   6   3  1 0.119  7.23 12.28     0 True
   6 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_7_m  72   9   4  0 0.181  6.38  9.35     0 True
   7 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_0_m  57   5   5  0 0.175  5.22  2.84     0 True
   8 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_5_m  62   5   5  0 0.161  4.33  4.21     0 True
   9 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_9_m  55   5   3  0 0.145  4.75  7.50     0 True
  10 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_5_m  62   7   2  0 0.145  3.28  6.11     0 True
  11 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_0_m  57   5   3  0 0.140  5.36  4.64     0 True
  12 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_1_m  51   4   3  0 0.137  4.75 10.10     0 True
  13 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_8_m  67   2   7  0 0.134  8.71  4.01     0 True
  14 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_5_m  62   3   5  0 0.129  3.43  4.58     0 True
  15 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_9_m  55   3   4  0 0.127  7.88  7.39     0 True
  16 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_7_m  72   5   4  0 0.125  6.54  9.13     0 True
  17 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_0_m  57   4   3  0 0.123  6.94  3.65     0 True
  18 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_0_m  57   4   3  0 0.123  9.96 12.45     0 True
  19 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_0_m  57   6   1  0 0.123  5.85  4.09     0 False
  20 frb_rap_binder_raphotspot_frb_rap_binder_raphotspot_3_m  75   2   7  0 0.120  5.21  2.20     0 True
'''