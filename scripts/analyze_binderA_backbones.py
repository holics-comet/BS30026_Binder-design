import gzip
import glob
import os
import csv
import numpy as np


INPUT_DIR = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_pilot"
OUTPUT_CSV = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_pilot_metrics.csv"

CONTACT_CUTOFF = 4.0
CLASH_CUTOFF = 2.0

# RFD3 output numbering
# Original FRB B2042 -> output B25
# Original FRB B2105 -> output B88
HOTSPOTS = {
    25: {"resname": "ARG", "atoms": ["NH1", "NH2"]},
    88: {"resname": "TYR", "atoms": ["OH"]},
}

SELECTED_RAP_ATOMS = {"O13", "O3", "O6", "O2", "O10", "O4"}


def read_cif(path):
    """RFD3 mmCIF에서 heavy atoms를 읽는다."""

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

            # 수소 제외
            if element == "H" or atom_name.startswith("H"):
                continue

            x = float(fields[18])
            y = float(fields[19])
            z = float(fields[20])

            atoms.append({
                "element": element,
                "atom": atom_name,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "xyz": np.array([x, y, z], dtype=float),
            })

    return atoms


def get_atoms(atoms, chain=None, resname=None, resid=None):
    """조건에 맞는 atom만 선택한다."""

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
    """두 atom group 사이 최소 거리."""

    if not group1 or not group2:
        return np.nan

    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])

    diff = xyz1[:, None, :] - xyz2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return float(dist.min())


def count_atom_contacts(group1, group2, cutoff=4.0):
    """
    cutoff 이하인 heavy-atom pair 개수.
    주의: residue contact 수가 아니라 atom-pair contact 수이다.
    """

    if not group1 or not group2:
        return 0

    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])

    diff = xyz1[:, None, :] - xyz2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return int(np.sum(dist <= cutoff))


def contacted_residues(group1, group2, cutoff=4.0):
    """
    group1의 각 residue가 group2와 cutoff 이하 contact를 가지는지 확인.
    반환값은 residue 번호 목록.
    """

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

        diff = xyz1[:, None, :] - xyz2[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        if np.any(dist <= cutoff):
            contacted.append((resid, resname))

    return sorted(contacted)


def contacted_atom_names(group1, group2, cutoff=4.0):
    """
    group1에서 group2와 contact하는 atom 이름을 반환.
    RAP의 어떤 atom이 binder와 접촉하는지 볼 때 사용.
    """

    if not group1 or not group2:
        return []

    xyz2 = np.array([a["xyz"] for a in group2])

    contacted = []

    for atom in group1:
        d = np.linalg.norm(xyz2 - atom["xyz"], axis=1)

        if np.any(d <= cutoff):
            contacted.append(atom["atom"])

    return sorted(set(contacted))


def hotspot_min_distance(binder, frb, resid, atom_names):
    """
    Binder와 지정한 FRB hotspot atom 사이 최소 거리.
    """

    hotspot_atoms = [
        a for a in frb
        if a["resid"] == resid and a["atom"] in atom_names
    ]

    return min_distance(binder, hotspot_atoms)


def analyze_structure(path):
    atoms = read_cif(path)

    # RFD3 output:
    # A = generated Binder A
    # B = FRB
    # C = RAP
    binder = get_atoms(atoms, chain="A")
    frb = get_atoms(atoms, chain="B")
    rap = get_atoms(atoms, chain="C", resname="RAP")

    binder_residues = sorted(set(a["resid"] for a in binder))
    binder_length = len(binder_residues)

    # Binder ↔ RAP
    binder_rap_min = min_distance(binder, rap)
    binder_rap_contacts = count_atom_contacts(
        binder, rap, CONTACT_CUTOFF
    )

    rap_contact_atoms = contacted_atom_names(
        rap, binder, CONTACT_CUTOFF
    )

    selected_contacted = sorted(
        set(rap_contact_atoms) & SELECTED_RAP_ATOMS
    )

    # Binder ↔ FRB
    binder_frb_min = min_distance(binder, frb)
    binder_frb_contacts = count_atom_contacts(
        binder, frb, CONTACT_CUTOFF
    )

    frb_contact_residues = contacted_residues(
        frb, binder, CONTACT_CUTOFF
    )

    # Hotspots
    b25_min = hotspot_min_distance(
        binder,
        frb,
        25,
        ["NH1", "NH2"]
    )

    b88_min = hotspot_min_distance(
        binder,
        frb,
        88,
        ["OH"]
    )

    b25_contact = bool(b25_min <= CONTACT_CUTOFF)
    b88_contact = bool(b88_min <= CONTACT_CUTOFF)

    # Intermolecular clashes
    binder_frb_clashes = count_atom_contacts(
        binder, frb, CLASH_CUTOFF
    )

    binder_rap_clashes = count_atom_contacts(
        binder, rap, CLASH_CUTOFF
    )

    # 문자열 출력용
    rap_atoms_string = ";".join(rap_contact_atoms)
    selected_string = ";".join(selected_contacted)

    frb_res_string = ";".join(
        f"{resid}:{resname}"
        for resid, resname in frb_contact_residues
    )

    return {
        "design": os.path.basename(path).replace(".cif.gz", ""),

        "binder_length": binder_length,

        "binder_rap_min_distance": binder_rap_min,
        "binder_rap_atom_contacts_4A": binder_rap_contacts,
        "rap_atoms_contacted_4A": rap_atoms_string,
        "rap_atom_types_contacted_4A": len(rap_contact_atoms),

        "selected_rap_atoms_contacted": selected_string,
        "selected_rap_atoms_contacted_count": len(selected_contacted),

        "binder_frb_min_distance": binder_frb_min,
        "binder_frb_atom_contacts_4A": binder_frb_contacts,
        "frb_residues_contacted_4A": frb_res_string,
        "frb_residues_contacted_count": len(frb_contact_residues),

        "B25_ARG_NH1_NH2_min_distance": b25_min,
        "B25_ARG_contact_4A": b25_contact,

        "B88_TYR_OH_min_distance": b88_min,
        "B88_TYR_contact_4A": b88_contact,

        "binder_frb_clashes_lt2A": binder_frb_clashes,
        "binder_rap_clashes_lt2A": binder_rap_clashes,
        "total_inter_clashes_lt2A":
            binder_frb_clashes + binder_rap_clashes,
    }


def main():
    files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.cif.gz"))
    )

    print(f"Found {len(files)} structures.")

    if not files:
        raise RuntimeError("No .cif.gz files found.")

    results = []

    for i, path in enumerate(files, start=1):
        result = analyze_structure(path)
        results.append(result)

        print(
            f"[{i:3d}/{len(files)}] "
            f"{result['design']} "
            f"length={result['binder_length']} "
            f"RAP_contacts={result['binder_rap_atom_contacts_4A']} "
            f"FRB_contacts={result['binder_frb_atom_contacts_4A']} "
            f"clashes={result['total_inter_clashes_lt2A']}"
        )

    fieldnames = list(results[0].keys())

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Analysis complete.")
    print(f"CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()