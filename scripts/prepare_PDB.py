from pathlib import Path
from math import dist

# 프로젝트 경로
PROJECT = Path(__file__).resolve().parent.parent

# 입력/출력 파일
INPUT = PROJECT / "inputs" / "1FAP.pdb"
OUTPUT = PROJECT / "inputs" / "1FAP_FRB_RAP.pdb"

CUTOFF = 4.0


# PDB 원자 정보 읽기
def read_atoms(pdb_file):
    atoms = []

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            atoms.append({
                "record": line[:6].strip(),
                "atom": line[12:16].strip(),
                "element": line[76:78].strip(),
                "resn": line[17:20].strip(),
                "chain": line[21],
                "resi": line[22:26].strip(),
                "xyz": (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ),
                "line": line,
            })

    return atoms


# 수소 원자 제외
def is_heavy(atom):
    return atom["element"].upper() not in ("H", "D")


# 한 원자와 원자 그룹 사이 최소 거리
def min_distance(atom, group):
    return min(
        dist(atom["xyz"], other["xyz"])
        for other in group
    )


atoms = read_atoms(INPUT)


# 1FAP 구성 요소
fkbp = [
    a for a in atoms
    if a["record"] == "ATOM"
    and a["chain"] == "A"
]

frb = [
    a for a in atoms
    if a["record"] == "ATOM"
    and a["chain"] == "B"
]

rap = [
    a for a in atoms
    if a["record"] == "HETATM"
    and a["chain"] == "A"
    and a["resn"] == "RAP"
    and a["resi"] == "108"
]


# 거리 계산용 heavy atom
fkbp_heavy = [a for a in fkbp if is_heavy(a)]
frb_heavy = [a for a in frb if is_heavy(a)]
rap_heavy = [a for a in rap if is_heavy(a)]


# 구조 정보
print("=== 1FAP structure ===")
print(f"FKBP atoms       : {len(fkbp)}")
print(f"FKBP heavy atoms : {len(fkbp_heavy)}")
print(f"FRB atoms        : {len(frb)}")
print(f"FRB heavy atoms  : {len(frb_heavy)}")
print(f"RAP atoms        : {len(rap)}")
print(f"RAP heavy atoms  : {len(rap_heavy)}")


# FKBP와 직접 접촉하는 FRB residue
frb_contact_residues = {
    (int(a["resi"]), a["resn"])
    for a in frb_heavy
    if min_distance(a, fkbp_heavy) <= CUTOFF
}


print(f"\n=== FRB residues within {CUTOFF} Å ===")

for resi, resn in sorted(frb_contact_residues):
    print(f"B{resi} {resn}")


# FRB heavy-atom hotspot 후보
print("\n=== FRB hotspot atom candidates ===")

for resi, resn in sorted(frb_contact_residues):

    residue_atoms = [
        a for a in frb_heavy
        if int(a["resi"]) == resi
    ]

    contacts = []

    for atom in residue_atoms:
        d = min_distance(atom, fkbp_heavy)

        if d <= CUTOFF:
            contacts.append((atom["atom"], d))

    print(f"\nB{resi} {resn}")

    for atom_name, d in sorted(
        contacts,
        key=lambda x: x[1]
    ):
        print(f"  {atom_name:4s}  {d:.2f} Å")


# RAP heavy atom의 FKBP 접촉 분석
rap_contacts = []

for atom in rap_heavy:
    d = min_distance(atom, fkbp_heavy)

    if d <= CUTOFF:
        rap_contacts.append((atom["atom"], d))


print("\n=== RAP heavy-atom contact candidates ===")

for atom_name, d in sorted(
    rap_contacts,
    key=lambda x: x[1]
):
    print(f"{atom_name:4s}  {d:.2f} Å")


# RFD3용 FRB + RAP target 저장
with open(OUTPUT, "w") as f:
    # FRB는 chain B 유지
    for atom in frb:
        f.write(atom["line"])

    # Rapamycin은 별도 chain C로 변경
    for atom in rap:
        line = atom["line"]
        line = line[:21] + "C" + line[22:]
        f.write(line)

    f.write("END\n")


print("\n=== Design target ===")
print(f"Created: {OUTPUT}")
print("Contains: WT FRB chain B + RAP")


# ===============================================================
# Outputs
# === 1FAP structure ===
# FKBP atoms       : 1022
# FKBP heavy atoms : 832
# FRB atoms        : 995
# FRB heavy atoms  : 807
# RAP atoms        : 68
# RAP heavy atoms  : 65

# === FRB residues within 4 Å ===
# B2038 TYR
# B2039 PHE
# B2042 ARG
# B2094 VAL
# B2095 LYS
# B2105 TYR
# B2109 ARG

# === FRB hotspot atom candidates ===

# B2038 TYR
#  OH    3.80 Å

# B2039 PHE
#  CD1   3.71 Å

# B2042 ARG
#   NH1   2.79 Å
#   NH2   3.10 Å
#   CZ    3.32 Å
#   CG    3.56 Å
#   CD    3.98 Å

# B2094 VAL
#   CG2   3.84 Å
#   CG1   3.90 Å

# B2095 LYS
#   CE    3.82 Å
#   NZ    3.96 Å

# B2105 TYR
#   OH    2.56 Å
#   CZ    3.80 Å

# B2109 ARG
#  CZ    3.91 Å
#  NH1   3.95 Å

# === RAP heavy-atom contact candidates ===
# O13   2.62 Å
# O3    2.71 Å
# O6    2.76 Å
# O2    2.78 Å
# O10   2.88 Å
# O4    2.95 Å
# O1    3.14 Å
# C1    3.28 Å
# C35   3.29 Å
# C30   3.31 Å
# C8    3.34 Å
# C49   3.41 Å
# C3    3.41 Å
# C2    3.41 Å
# O11   3.42 Å
# C10   3.43 Å
# C41   3.44 Å
# C9    3.49 Å
# C40   3.49 Å
# C4    3.57 Å
# C43   3.58 Å
# C11   3.58 Å
# O5    3.62 Å
# C39   3.63 Å
# N7    3.67 Å
# C5    3.73 Å
# O8    3.75 Å
# C32   3.82 Å
# C29   3.84 Å
# C28   3.86 Å
# C42   3.87 Å
# C34   3.94 Å

# === Design target ===
# Created: 1FAP_FRB_RAP.pdb
# Contains: WT FRB chain B + RAP

# ===============================================================