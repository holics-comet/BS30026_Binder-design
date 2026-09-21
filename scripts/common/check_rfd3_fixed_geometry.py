import gzip
import numpy as np


INPUT_PDB = "/home01/a2149a01/RFdiffusion3/inputs/reference/1FAP_FRB_RAP.pdb"
OUTPUT_CIF = (
    "/home01/a2149a01/RFdiffusion3/outputs/frb_rap_pilot/"
    "frb_rap_binder_frb_rap_binder_0_model_0.cif.gz"
)


# PDB에서 FRB(B2018-2112)와 RAP(C108) heavy atom 읽기
def read_input_pdb(path):
    frb = {}
    rap = {}

    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resid = int(line[22:26])

            # 수소 제외
            element = line[76:78].strip()
            if element == "H" or atom.startswith("H"):
                continue

            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])

            if chain == "B" and 2018 <= resid <= 2112:
                # output에서는 B2018 -> B1
                new_resid = resid - 2017
                frb[(new_resid, resname, atom)] = xyz

            elif chain == "C" and resid == 108 and resname == "RAP":
                rap[atom] = xyz

    return frb, rap


# RFD3 mmCIF output 읽기
def read_output_cif(path):
    frb = {}
    rap = {}

    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith(("ATOM ", "HETATM ")):
                continue

            fields = line.split()

            # 현재 RFD3 output의 _atom_site column 순서
            atom = fields[2]
            resname = fields[4]
            chain = fields[5]
            resid = int(fields[9])

            x = float(fields[18])
            y = float(fields[19])
            z = float(fields[20])
            xyz = np.array([x, y, z])

            # 수소 제외
            element = fields[1]
            if element == "H" or atom.startswith("H"):
                continue

            if chain == "B":
                frb[(resid, resname, atom)] = xyz

            elif chain == "C" and resname == "RAP":
                rap[atom] = xyz

    return frb, rap


# Kabsch alignment
# mobile을 reference에 맞추는 rotation/translation 계산
def kabsch(mobile, reference):
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)

    P = mobile - mobile_center
    Q = reference - reference_center

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # reflection 방지
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # row-vector 좌표에 적용할 형태
    R_row = R.T

    t = reference_center - mobile_center @ R_row

    return R_row, t


def rmsd(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def main():
    input_frb, input_rap = read_input_pdb(INPUT_PDB)
    output_frb, output_rap = read_output_cif(OUTPUT_CIF)

    print("=== Atom counts ===")
    print("Input FRB heavy atoms :", len(input_frb))
    print("Output FRB heavy atoms:", len(output_frb))
    print("Input RAP heavy atoms :", len(input_rap))
    print("Output RAP heavy atoms:", len(output_rap))

    # FRB에서 양쪽에 모두 존재하는 atom만 사용
    common_frb = sorted(set(input_frb) & set(output_frb))

    input_frb_xyz = np.array([input_frb[k] for k in common_frb])
    output_frb_xyz = np.array([output_frb[k] for k in common_frb])

    print("\nMatched FRB heavy atoms:", len(common_frb))

    # output FRB를 input FRB에 정렬
    R, t = kabsch(output_frb_xyz, input_frb_xyz)
    aligned_output_frb = output_frb_xyz @ R + t

    frb_rmsd = rmsd(aligned_output_frb, input_frb_xyz)

    frb_displacements = np.linalg.norm(
        aligned_output_frb - input_frb_xyz,
        axis=1
    )

    print("\n=== FRB after alignment ===")
    print(f"FRB RMSD             : {frb_rmsd:.4f} A")
    print(f"FRB max displacement : {frb_displacements.max():.4f} A")

    # 같은 FRB alignment transform을 RAP에도 적용
    common_rap = sorted(set(input_rap) & set(output_rap))

    input_rap_xyz = np.array([input_rap[k] for k in common_rap])
    output_rap_xyz = np.array([output_rap[k] for k in common_rap])

    aligned_output_rap = output_rap_xyz @ R + t

    rap_rmsd = rmsd(aligned_output_rap, input_rap_xyz)

    rap_displacements = np.linalg.norm(
        aligned_output_rap - input_rap_xyz,
        axis=1
    )

    print("\nMatched RAP heavy atoms:", len(common_rap))

    print("\n=== RAP using FRB alignment ===")
    print(f"RAP RMSD             : {rap_rmsd:.4f} A")
    print(f"RAP max displacement : {rap_displacements.max():.4f} A")

    # 가장 많이 이동한 RAP atom 확인
    print("\n=== Largest RAP atom displacements ===")

    ranked = sorted(
        zip(common_rap, rap_displacements),
        key=lambda x: x[1],
        reverse=True
    )

    for atom, dist in ranked[:10]:
        print(f"{atom:>4s} : {dist:.4f} A")


if __name__ == "__main__":
    main()