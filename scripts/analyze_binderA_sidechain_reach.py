import gzip
import glob
import os
import csv
import numpy as np


INPUT_DIR = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_raphotspot_pilot"
OUTPUT_CSV = "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_raphotspot_sidechain_reach.csv"


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


def get_target_atoms(atoms):
    """
    우리가 관심 있는 네 target atom group을 만든다.

    RAP:
        O3
        O8

    FRB:
        B25 ARG NH1/NH2
        B88 TYR OH
    """

    rap = get_atoms(
        atoms,
        chain="C",
        resname="RAP"
    )

    frb = get_atoms(
        atoms,
        chain="B"
    )

    rap_o3 = [
        a for a in rap
        if a["atom"] == "O3"
    ]

    rap_o8 = [
        a for a in rap
        if a["atom"] == "O8"
    ]

    b25 = [
        a for a in frb
        if a["resid"] == 25
        and a["resname"] == "ARG"
        and a["atom"] in {"NH1", "NH2"}
    ]

    b88 = [
        a for a in frb
        if a["resid"] == 88
        and a["resname"] == "TYR"
        and a["atom"] == "OH"
    ]

    return {
        "RAP_O3": rap_o3,
        "RAP_O8": rap_o8,
        "B25_ARG": b25,
        "B88_TYR": b88,
    }


def group_binder_residues(atoms):
    """
    Binder A를 residue별로 묶는다.

    반환:
        {
            resid: {
                "resname": ...,
                "atoms": [...]
            }
        }
    """

    binder = get_atoms(
        atoms,
        chain="A"
    )

    residues = {}

    for atom in binder:
        resid = atom["resid"]

        if resid not in residues:
            residues[resid] = {
                "resname": atom["resname"],
                "atoms": []
            }

        residues[resid]["atoms"].append(atom)

    return residues


def find_atom(residue_atoms, atom_name):
    """특정 residue에서 atom 하나를 찾는다."""

    for atom in residue_atoms:
        if atom["atom"] == atom_name:
            return atom

    return None


def distance_to_target(point_atom, target_atoms):
    """하나의 atom에서 target atom group까지 최소 거리."""

    if point_atom is None or not target_atoms:
        return np.nan

    xyz = np.array([
        a["xyz"]
        for a in target_atoms
    ])

    d = np.linalg.norm(
        xyz - point_atom["xyz"],
        axis=1
    )

    return float(d.min())


def closest_residue_CA(residues, target_atoms):
    """
    모든 Binder residue의 CA 중 target에 가장 가까운 residue를 찾는다.
    """

    best = None

    for resid, data in residues.items():

        ca = find_atom(
            data["atoms"],
            "CA"
        )

        if ca is None:
            continue

        d = distance_to_target(
            ca,
            target_atoms
        )

        if np.isnan(d):
            continue

        if best is None or d < best["distance"]:
            best = {
                "resid": resid,
                "resname": data["resname"],
                "atom": "CA",
                "distance": d,
            }

    return best


def closest_residue_CB_preferred(residues, target_atoms):
    """
    각 Binder residue에서 가능하면 CB를 사용한다.

    CB가 없는 경우(Gly 등)는 CA를 fallback으로 사용한다.

    주의:
    이것은 '실제 sidechain 끝 위치'가 아니다.
    단지 sidechain이 시작되는 backbone 근처 위치를 비교하기 위한
    geometry descriptor이다.
    """

    best = None

    for resid, data in residues.items():

        cb = find_atom(
            data["atoms"],
            "CB"
        )

        if cb is not None:
            reference_atom = cb
        else:
            reference_atom = find_atom(
                data["atoms"],
                "CA"
            )

        if reference_atom is None:
            continue

        d = distance_to_target(
            reference_atom,
            target_atoms
        )

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


def safe_value(result, key):
    """result가 None일 때 CSV에 빈 값을 기록한다."""

    if result is None:
        return ""

    return result[key]


def analyze_structure(path):

    atoms = read_cif(path)

    residues = group_binder_residues(
        atoms
    )

    targets = get_target_atoms(
        atoms
    )

    result = {
        "design":
            os.path.basename(path).replace(".cif.gz", ""),

        "binder_length":
            len(residues),
    }

    # 네 target 각각에 대해 CA와 CB-preferred 분석
    for target_name, target_atoms in targets.items():

        ca_best = closest_residue_CA(
            residues,
            target_atoms
        )

        cb_best = closest_residue_CB_preferred(
            residues,
            target_atoms
        )

        # CA
        result[
            f"{target_name}_CA_min_distance"
        ] = safe_value(
            ca_best,
            "distance"
        )

        result[
            f"{target_name}_CA_closest_resid"
        ] = safe_value(
            ca_best,
            "resid"
        )

        result[
            f"{target_name}_CA_closest_resname"
        ] = safe_value(
            ca_best,
            "resname"
        )

        # CB preferred
        result[
            f"{target_name}_CBpref_min_distance"
        ] = safe_value(
            cb_best,
            "distance"
        )

        result[
            f"{target_name}_CBpref_closest_resid"
        ] = safe_value(
            cb_best,
            "resid"
        )

        result[
            f"{target_name}_CBpref_closest_resname"
        ] = safe_value(
            cb_best,
            "resname"
        )

        result[
            f"{target_name}_CBpref_reference_atom"
        ] = safe_value(
            cb_best,
            "atom"
        )

    return result


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

        result = analyze_structure(
            path
        )

        results.append(
            result
        )

        print(
            f"[{i:3d}/{len(files)}] "
            f"{result['design']} | "
            f"O3={float(result['RAP_O3_CBpref_min_distance']):.2f} "
            f"(res {result['RAP_O3_CBpref_closest_resid']}) | "
            f"O8={float(result['RAP_O8_CBpref_min_distance']):.2f} "
            f"(res {result['RAP_O8_CBpref_closest_resid']}) | "
            f"B25={float(result['B25_ARG_CBpref_min_distance']):.2f} "
            f"(res {result['B25_ARG_CBpref_closest_resid']}) | "
            f"B88={float(result['B88_TYR_CBpref_min_distance']):.2f} "
            f"(res {result['B88_TYR_CBpref_closest_resid']})"
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
        writer.writerows(
            results
        )

    print()
    print("Analysis complete.")
    print(
        f"CSV written to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()