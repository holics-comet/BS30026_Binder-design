from pathlib import Path
import json

ROOT = Path("/home01/a2149a01/RFdiffusion3")
TARGET = ROOT / "inputs/binderB_targets"
OUT = ROOT / "inputs/binderB_rfd3"

# candidate: (A length, protein hotspot, RAP hotspots)
CONFIG = {
    "1_model_7_b0_d0": (
        51,
        {"A44": "SD,CE"},
        {"RAP": "O3,N7"},
    ),
    "3_model_3_b2_d0": (
        75,
        {"A72": "NE,NH1,NH2"},
        {"RAP": "O8,O9"},
    ),
    "5_model_1_b0_d0": (
        62,
        {"A52": "OH"},
        {"RAP": "O10,O2"},
    ),
    "6_model_5_b3_d0": (
        67,
        {"A54": "NE,NH1,NH2"},
        {"RAP": "O10,O13"},
    ),
    "9_model_4_b0_d0": (
        55,
        {"A24": "OD1,OD2"},
        {"RAP": "O2,O3"},
    ),
}

OUT.mkdir(parents=True, exist_ok=True)

for cand, (length, protein_hs, rap_hs) in CONFIG.items():

    hotspots = {}
    hotspots.update(protein_hs)
    hotspots.update(rap_hs)

    spec = {
        "dialect": 2,
        "input": str(TARGET / f"{cand}_A_RAP.cif"),
        "ligand": "RAP",

        # A = fixed Binder A
        # /0 -> new chain
        # B = generated Binder B (50-80 aa)
        "contig": f"A1-{length},/0,50-80",

        "infer_ori_strategy": "hotspots",
        "select_hotspots": hotspots,
        "is_non_loopy": True,
    }

    path = OUT / f"{cand}_binderB.json"

    with open(path, "w") as f:
        json.dump({f"{cand}_binderB": spec}, f, indent=2)

    print(path)
    print("  contig:", spec["contig"])
    print("  hotspots:", hotspots)
