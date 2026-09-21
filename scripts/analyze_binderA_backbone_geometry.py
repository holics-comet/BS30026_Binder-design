import gzip
import glob
import os
import csv
import numpy as np


INPUT_DIR = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_raphotspot_pilot"
OUTPUT_CSV = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_raphotspot_backbone_geometry.csv"

CONTACT_CUTOFF = 4.0
CLASH_CUTOFF = 2.0

# Generated Binder A에서 sidechain을 완전히 제외한다.
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# RFD3 output numbering
# Original FRB B2042 ARG -> output B25
# Original FRB B2105 TYR -> output B88
HOTSPOTS = {
    25: {"resname": "ARG", "atoms": ["NH1", "NH2"]},
    88: {"resname": "TYR", "atoms": ["OH"]},
}

# 현재 RFD3 conditioning에 사용한 RAP hotspot
RAP_HOTSPOT_ATOMS = {"O3", "O8"}


def read_cif(path):
    """RFD3 mmCIF에서 heavy atom을 읽는다."""

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
    """조건에 맞는 atom을 선택한다."""

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


def count_atom_contacts(group1, group2, cutoff):
    """cutoff 이하의 atom-pair 수."""

    if not group1 or not group2:
        return 0

    xyz1 = np.array([a["xyz"] for a in group1])
    xyz2 = np.array([a["xyz"] for a in group2])

    diff = xyz1[:, None, :] - xyz2[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    return int(np.sum(dist <= cutoff))


def contacting_binder_residues(binder_bb, target, cutoff):
    """
    target과 cutoff 이내에 있는 Binder A residue 번호를 반환한다.

    Binder 쪽은 N/CA/C/O만 사용한다.
    """

    if not binder_bb or not target:
        return []

    target_xyz = np.array([a["xyz"] for a in target])

    residues = {}

    for atom in binder_bb:
        residues.setdefault(atom["resid"], []).append(atom)

    contacted = []

    for resid, residue_atoms in residues.items():

        xyz = np.array([a["xyz"] for a in residue_atoms])

        diff = xyz[:, None, :] - target_xyz[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        if np.any(dist <= cutoff):
            contacted.append(resid)

    return sorted(contacted)


def contacted_target_residues(target, binder_bb, cutoff):
    """
    Binder backbone과 접촉하는 target residue들을 반환한다.
    FRB interface 크기를 residue 단위로 보기 위함.
    """

    if not target or not binder_bb:
        return []

    binder_xyz = np.array([a["xyz"] for a in binder_bb])

    residues = {}

    for atom in target:
        key = (atom["resid"], atom["resname"])
        residues.setdefault(key, []).append(atom)

    contacted = []

    for key, residue_atoms in residues.items():

        xyz = np.array([a["xyz"] for a in residue_atoms])

        diff = xyz[:, None, :] - binder_xyz[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        if np.any(dist <= cutoff):
            contacted.append(key)

    return sorted(contacted)


def contacted_target_atoms(target, binder_bb, cutoff):
    """
    Binder backbone과 접촉하는 target atom 이름을 반환한다.
    RAP의 어떤 atom들이 backbone 가까이에 있는지 확인한다.
    """

    if not target or not binder_bb:
        return []

    binder_xyz = np.array([a["xyz"] for a in binder_bb])

    contacted = []

    for atom in target:

        d = np.linalg.norm(
            binder_xyz - atom["xyz"],
            axis=1
        )

        if np.any(d <= cutoff):
            contacted.append(atom["atom"])

    return sorted(set(contacted))


def hotspot_min_distance(binder_bb, frb, resid, atom_names):
    """Binder backbone과 FRB hotspot atom 사이 최소 거리."""

    hotspot_atoms = [
        a for a in frb
        if a["resid"] == resid
        and a["atom"] in atom_names
    ]

    return min_distance(binder_bb, hotspot_atoms)


def rap_hotspot_min_distance(binder_bb, rap, atom_name):
    """Binder backbone과 특정 RAP hotspot atom 사이 최소 거리."""

    atoms = [
        a for a in rap
        if a["atom"] == atom_name
    ]

    return min_distance(binder_bb, atoms)


def residue_list_string(residues):
    """Residue 번호 목록을 CSV용 문자열로 변환."""

    return ";".join(str(x) for x in residues)


def analyze_structure(path):

    atoms = read_cif(path)

    # RFD3 output
    # A = generated Binder A
    # B = FRB
    # C = RAP
    binder_all = get_atoms(atoms, chain="A")
    frb = get_atoms(atoms, chain="B")
    rap = get_atoms(atoms, chain="C", resname="RAP")

    # 핵심:
    # Binder A에서는 N, CA, C, O만 남긴다.
    binder_bb = [
        a for a in binder_all
        if a["atom"] in BACKBONE_ATOMS
    ]

    binder_residues = sorted(
        set(a["resid"] for a in binder_bb)
    )

    binder_length = len(binder_residues)

    # --------------------------------------------------
    # Binder backbone ↔ RAP
    # --------------------------------------------------

    rap_min = min_distance(
        binder_bb,
        rap
    )

    rap_atom_contacts = count_atom_contacts(
        binder_bb,
        rap,
        CONTACT_CUTOFF
    )

    rap_binder_residues = contacting_binder_residues(
        binder_bb,
        rap,
        CONTACT_CUTOFF
    )

    rap_atoms_contacted = contacted_target_atoms(
        rap,
        binder_bb,
        CONTACT_CUTOFF
    )

    # RAP O3/O8 각각의 거리
    o3_min = rap_hotspot_min_distance(
        binder_bb,
        rap,
        "O3"
    )

    o8_min = rap_hotspot_min_distance(
        binder_bb,
        rap,
        "O8"
    )

    # --------------------------------------------------
    # Binder backbone ↔ FRB
    # --------------------------------------------------

    frb_min = min_distance(
        binder_bb,
        frb
    )

    frb_atom_contacts = count_atom_contacts(
        binder_bb,
        frb,
        CONTACT_CUTOFF
    )

    frb_binder_residues = contacting_binder_residues(
        binder_bb,
        frb,
        CONTACT_CUTOFF
    )

    frb_target_residues = contacted_target_residues(
        frb,
        binder_bb,
        CONTACT_CUTOFF
    )

    # --------------------------------------------------
    # FRB hotspots
    # --------------------------------------------------

    b25_min = hotspot_min_distance(
        binder_bb,
        frb,
        25,
        ["NH1", "NH2"]
    )

    b88_min = hotspot_min_distance(
        binder_bb,
        frb,
        88,
        ["OH"]
    )

    b25_contact = bool(
        b25_min <= CONTACT_CUTOFF
    )

    b88_contact = bool(
        b88_min <= CONTACT_CUTOFF
    )

    # --------------------------------------------------
    # Composite interface
    # --------------------------------------------------

    rap_set = set(rap_binder_residues)
    frb_set = set(frb_binder_residues)

    # 같은 Binder residue의 backbone이
    # RAP과 FRB 양쪽 모두 4 Å 이내인 경우
    bridge_residues = sorted(
        rap_set & frb_set
    )

    # RAP 또는 FRB와 접촉하는 전체 Binder residues
    interface_residues = sorted(
        rap_set | frb_set
    )

    # --------------------------------------------------
    # Backbone-only clash
    # --------------------------------------------------

    rap_clashes = count_atom_contacts(
        binder_bb,
        rap,
        CLASH_CUTOFF
    )

    frb_clashes = count_atom_contacts(
        binder_bb,
        frb,
        CLASH_CUTOFF
    )

    total_clashes = (
        rap_clashes +
        frb_clashes
    )

    # --------------------------------------------------
    # CSV 문자열
    # --------------------------------------------------

    rap_res_string = residue_list_string(
        rap_binder_residues
    )

    frb_res_string = residue_list_string(
        frb_binder_residues
    )

    bridge_string = residue_list_string(
        bridge_residues
    )

    interface_string = residue_list_string(
        interface_residues
    )

    rap_atoms_string = ";".join(
        rap_atoms_contacted
    )

    frb_target_string = ";".join(
        f"{resid}:{resname}"
        for resid, resname in frb_target_residues
    )

    return {
        "design":
            os.path.basename(path).replace(".cif.gz", ""),

        "binder_length":
            binder_length,

        # RAP
        "bb_rap_min_distance":
            rap_min,

        "bb_rap_atom_contacts_4A":
            rap_atom_contacts,

        "bb_rap_contact_residues":
            rap_res_string,

        "bb_rap_contact_residue_count":
            len(rap_binder_residues),

        "rap_atoms_contacted_by_bb":
            rap_atoms_string,

        "rap_atoms_contacted_by_bb_count":
            len(rap_atoms_contacted),

        # RAP hotspots
        "bb_RAP_O3_min_distance":
            o3_min,

        "bb_RAP_O3_contact_4A":
            bool(o3_min <= CONTACT_CUTOFF),

        "bb_RAP_O8_min_distance":
            o8_min,

        "bb_RAP_O8_contact_4A":
            bool(o8_min <= CONTACT_CUTOFF),

        # FRB
        "bb_frb_min_distance":
            frb_min,

        "bb_frb_atom_contacts_4A":
            frb_atom_contacts,

        "bb_frb_contact_residues":
            frb_res_string,

        "bb_frb_contact_residue_count":
            len(frb_binder_residues),

        "frb_residues_contacted_by_bb":
            frb_target_string,

        "frb_residues_contacted_by_bb_count":
            len(frb_target_residues),

        # FRB hotspots
        "bb_B25_ARG_min_distance":
            b25_min,

        "bb_B25_ARG_contact_4A":
            b25_contact,

        "bb_B88_TYR_min_distance":
            b88_min,

        "bb_B88_TYR_contact_4A":
            b88_contact,

        # Composite interface
        "bb_bridge_residues":
            bridge_string,

        "bb_bridge_residue_count":
            len(bridge_residues),

        "bb_interface_residues":
            interface_string,

        "bb_interface_residue_count":
            len(interface_residues),

        # Backbone clashes
        "bb_rap_clashes_lt2A":
            rap_clashes,

        "bb_frb_clashes_lt2A":
            frb_clashes,

        "bb_total_clashes_lt2A":
            total_clashes,
    }


def main():

    files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.cif.gz"
            )
        )
    )

    print(
        f"Found {len(files)} structures."
    )

    if not files:
        raise RuntimeError(
            "No .cif.gz files found."
        )

    results = []

    for i, path in enumerate(
        files,
        start=1
    ):

        result = analyze_structure(path)
        results.append(result)

        print(
            f"[{i:3d}/{len(files)}] "
            f"{result['design']} "
            f"length={result['binder_length']} "
            f"RAPres={result['bb_rap_contact_residue_count']} "
            f"FRBres={result['bb_frb_contact_residue_count']} "
            f"bridge={result['bb_bridge_residue_count']} "
            f"B25={result['bb_B25_ARG_min_distance']:.2f} "
            f"B88={result['bb_B88_TYR_min_distance']:.2f} "
            f"clash={result['bb_total_clashes_lt2A']}"
        )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Analysis complete.")
    print(
        f"CSV written to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()