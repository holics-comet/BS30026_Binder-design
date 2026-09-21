import gzip
import glob
import os
import csv
import numpy as np


PROJECT = "/home01/a2149a01/RFdiffusion3"
INPUT_ROOT = f"{PROJECT}/outputs/binderB_rfd3"
OUTPUT_CSV = f"{PROJECT}/outputs/binderB_backbone_metrics.csv"

CONTACT_CUTOFF = 4.0
CLASH_CUTOFF = 2.0

# target별 RFD3 hotspot
HOTSPOTS = {
    "1_model_7_b0_d0": {
        "A": {44: ["SD", "CE"]},
        "RAP": ["O3", "N7"],
    },
    "3_model_3_b2_d0": {
        "A": {72: ["NE", "NH1", "NH2"]},
        "RAP": ["O8", "O9"],
    },
    "5_model_1_b0_d0": {
        "A": {52: ["OH"]},
        "RAP": ["O10", "O2"],
    },
    "6_model_5_b3_d0": {
        "A": {54: ["NE", "NH1", "NH2"]},
        "RAP": ["O10", "O13"],
    },
    "9_model_4_b0_d0": {
        "A": {24: ["OD1", "OD2"]},
        "RAP": ["O2", "O3"],
    },
}


def read_cif(path):
    atoms = []
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(("ATOM ", "HETATM ")):
                continue

            fields = line.split()

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
                    [float(fields[18]),
                     float(fields[19]),
                     float(fields[20])],
                    dtype=float,
                ),
            })

    return atoms


def get_atoms(atoms, chain=None, resname=None, resid=None):
    selected = []

    for a in atoms:
        if chain is not None and a["chain"] != chain:
            continue
        if resname is not None and a["resname"] != resname:
            continue
        if resid is not None and a["resid"] != resid:
            continue
        selected.append(a)

    return selected


def min_distance(group1, group2):
    if not group1 or not group2:
        return np.nan

    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])

    dist = np.linalg.norm(
        xyz1[:, None, :] - xyz2[None, :, :],
        axis=2,
    )

    return float(dist.min())


def count_atom_contacts(group1, group2, cutoff):
    if not group1 or not group2:
        return 0

    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])

    dist = np.linalg.norm(
        xyz1[:, None, :] - xyz2[None, :, :],
        axis=2,
    )

    return int(np.sum(dist <= cutoff))


def contacted_residues(group1, group2, cutoff):
    if not group1 or not group2:
        return []

    xyz2 = np.array([a["xyz"] for a in group2])

    residues = {}

    for atom in group1:
        key = (atom["resid"], atom["resname"])
        residues.setdefault(key, []).append(atom)

    contacted = []

    for (resid, resname), residue_atoms in residues.items():
        xyz1 = np.array([a["xyz"] for a in residue_atoms])

        dist = np.linalg.norm(
            xyz1[:, None, :] - xyz2[None, :, :],
            axis=2,
        )

        if np.any(dist <= cutoff):
            contacted.append((resid, resname))

    return sorted(contacted)


def contacted_atom_names(group1, group2, cutoff):
    if not group1 or not group2:
        return []

    xyz2 = np.array([a["xyz"] for a in group2])

    contacted = []

    for atom in group1:
        d = np.linalg.norm(xyz2 - atom["xyz"], axis=1)

        if np.any(d <= cutoff):
            contacted.append(atom["atom"])

    return sorted(set(contacted))


def selected_atom_min_distance(
    binder_b,
    target_atoms,
    resid,
    atom_names,
):
    selected = [
        a for a in target_atoms
        if a["resid"] == resid
        and a["atom"] in atom_names
    ]

    return min_distance(binder_b, selected)


def analyze_structure(path, target):
    atoms = read_cif(path)

    # RFD3 output
    # A = fixed Binder A
    # B = generated Binder B
    # C = RAP
    binder_a = get_atoms(atoms, chain="A")
    binder_b = get_atoms(atoms, chain="B")
    rap = get_atoms(atoms, chain="C", resname="RAP")

    if not binder_a or not binder_b or not rap:
        raise RuntimeError(
            f"Missing A/B/RAP in {path}: "
            f"A={len(binder_a)}, "
            f"B={len(binder_b)}, "
            f"RAP={len(rap)}"
        )

    binder_b_length = len(
        set(a["resid"] for a in binder_b)
    )

    # Binder B ↔ RAP
    b_rap_min = min_distance(binder_b, rap)

    b_rap_contacts = count_atom_contacts(
        binder_b,
        rap,
        CONTACT_CUTOFF,
    )

    rap_contact_atoms = contacted_atom_names(
        rap,
        binder_b,
        CONTACT_CUTOFF,
    )

    selected_rap = HOTSPOTS[target]["RAP"]

    selected_rap_contacted = sorted(
        set(rap_contact_atoms) & set(selected_rap)
    )

    # Binder B ↔ Binder A
    b_a_min = min_distance(binder_b, binder_a)

    b_a_contacts = count_atom_contacts(
        binder_b,
        binder_a,
        CONTACT_CUTOFF,
    )

    a_contact_residues = contacted_residues(
        binder_a,
        binder_b,
        CONTACT_CUTOFF,
    )

    # Binder A hotspot
    a_hotspot_distances = []
    a_hotspot_contacts = []

    for resid, atom_names in HOTSPOTS[target]["A"].items():
        d = selected_atom_min_distance(
            binder_b,
            binder_a,
            resid,
            atom_names,
        )

        a_hotspot_distances.append(d)
        a_hotspot_contacts.append(
            bool(d <= CONTACT_CUTOFF)
        )

    a_hotspot_min = min(a_hotspot_distances)
    a_hotspot_contact = any(a_hotspot_contacts)

    # RAP hotspot distances
    rap_hotspot_atoms = [
        a for a in rap
        if a["atom"] in selected_rap
    ]

    rap_hotspot_min = min_distance(
        binder_b,
        rap_hotspot_atoms,
    )

    rap_hotspot_contact = bool(
        rap_hotspot_min <= CONTACT_CUTOFF
    )

    # Intermolecular clashes
    b_a_clashes = count_atom_contacts(
        binder_b,
        binder_a,
        CLASH_CUTOFF,
    )

    b_rap_clashes = count_atom_contacts(
        binder_b,
        rap,
        CLASH_CUTOFF,
    )

    rap_atoms_string = ";".join(rap_contact_atoms)

    selected_rap_string = ";".join(
        selected_rap_contacted
    )

    a_res_string = ";".join(
        f"{resid}:{resname}"
        for resid, resname in a_contact_residues
    )

    composite_contact = (
        b_a_contacts > 0
        and b_rap_contacts > 0
    )

    return {
        "target": target,
        "design": os.path.basename(path).replace(
            ".cif.gz", ""
        ),

        "binderB_length": binder_b_length,

        "B_RAP_min_distance": b_rap_min,
        "B_RAP_atom_contacts_4A": b_rap_contacts,
        "RAP_atoms_contacted_4A": rap_atoms_string,
        "RAP_atom_types_contacted_4A":
            len(rap_contact_atoms),

        "selected_RAP_atoms":
            ";".join(selected_rap),
        "selected_RAP_atoms_contacted":
            selected_rap_string,
        "selected_RAP_atoms_contacted_count":
            len(selected_rap_contacted),
        "RAP_hotspot_min_distance":
            rap_hotspot_min,
        "RAP_hotspot_contact_4A":
            rap_hotspot_contact,

        "B_A_min_distance": b_a_min,
        "B_A_atom_contacts_4A": b_a_contacts,
        "A_residues_contacted_4A":
            a_res_string,
        "A_residues_contacted_count":
            len(a_contact_residues),

        "A_hotspot_min_distance":
            a_hotspot_min,
        "A_hotspot_contact_4A":
            a_hotspot_contact,

        "composite_contact_4A":
            composite_contact,

        "B_A_clashes_lt2A":
            b_a_clashes,
        "B_RAP_clashes_lt2A":
            b_rap_clashes,
        "total_inter_clashes_lt2A":
            b_a_clashes + b_rap_clashes,
    }


def main():
    results = []

    for target in HOTSPOTS:
        pattern = os.path.join(
            INPUT_ROOT,
            target,
            "*.cif.gz",
        )

        files = sorted(glob.glob(pattern))

        print(f"{target}: {len(files)} structures")

        if len(files) != 80:
            print(
                f"WARNING: expected 80, found {len(files)}"
            )

        for path in files:
            results.append(
                analyze_structure(path, target)
            )

    if not results:
        raise RuntimeError("No structures found.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Total structures: {len(results)}")
    print(
        "Composite contact:",
        sum(r["composite_contact_4A"]
            for r in results),
    )
    print(
        "No clashes:",
        sum(r["total_inter_clashes_lt2A"] == 0
            for r in results),
    )
    print(
        "Composite + no clashes:",
        sum(
            r["composite_contact_4A"]
            and r["total_inter_clashes_lt2A"] == 0
            for r in results
        ),
    )
    print()
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()