#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math
import re
import shutil

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure

PROJECT = Path('/home01/a2149a01/RFdiffusion3')
PLUS_ROOT = PROJECT / 'outputs/binderB_af3/with_rap'
MINUS_ROOT = PROJECT / 'outputs/binderB_af3/without_rap'

OUT = Path('/scratch/a2149a01/RFdiffusion3_data/outputs/binderB_final_top10')
OUT_PLUS = OUT / 'plus_RAP'
OUT_MINUS = OUT / 'minus_RAP'
OUT_ALIGN = OUT / 'align'

PASS_CSV = OUT / 'rap_dependence_pass.csv'
RANK_CSV = OUT / 'ranking.csv'

B_RMSD_CUTOFF = 20.0
TOP_N = 10

# +RAP: A=Binder A, B=Binder B, C=RAP
# -RAP: A=Binder A, B=Binder B
ANCHOR_CHAIN = 'A'
BINDER_B_CHAIN = 'B'
RAP_CHAIN_INDEX = 2

# Overlay output chain IDs in align/RankN.cif
# +RAP structure keeps A/B/C
# aligned -RAP structure is renamed to X/Y
OVERLAY_MINUS_CHAIN_MAP = {
    'A': 'X',
    'B': 'Y',
}


def safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else float('nan')
    except Exception:
        return float('nan')


def load_structure(path):
    cif = CIFFile.read(path)
    return get_structure(cif, model=1, include_bonds=True, extra_fields=['atom_id'])


def write_structure(path, structure):
    cif = CIFFile()
    set_structure(cif, structure)
    cif.write(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_one(directory, pattern):
    hits = list(directory.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"{directory}: expected 1 {pattern}, found {len(hits)}")
    return hits[0]


def collect_samples(candidate_dir):
    rows = []
    for sdir in sorted(candidate_dir.glob('seed-*_sample-*')):
        m = re.fullmatch(r'seed-(\d+)_sample-(\d+)', sdir.name)
        if m is None:
            continue
        rows.append({
            'seed': int(m.group(1)),
            'sample': int(m.group(2)),
            'cif': find_one(sdir, '*_model.cif'),
            'summary_path': find_one(sdir, '*_summary_confidences.json'),
        })
    for r in rows:
        r['summary'] = load_json(r['summary_path'])
    return rows

def ca_dict(structure, chain_id):
    x = structure[(structure.chain_id == chain_id) & (structure.atom_name == 'CA')]
    return {int(x.res_id[i]): x.coord[i].copy() for i in range(len(x))}


def matched_ca(s1, s2, chain_id):
    d1, d2 = ca_dict(s1, chain_id), ca_dict(s2, chain_id)
    common = sorted(set(d1) & set(d2))
    if len(common) < 3:
        raise RuntimeError(f'Too few common CA atoms in chain {chain_id}: {len(common)}')
    return (
        np.array([d1[r] for r in common], dtype=float),
        np.array([d2[r] for r in common], dtype=float),
    )


def kabsch(mobile, target):
    mc, tc = mobile.mean(0), target.mean(0)
    m0, t0 = mobile - mc, target - tc
    U, _, Vt = np.linalg.svd(m0.T @ t0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = tc - mc @ R
    return R, t


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def align_minus_to_plus(plus_structure, minus_structure):
    plus_A, minus_A = matched_ca(plus_structure, minus_structure, ANCHOR_CHAIN)
    R, t = kabsch(minus_A, plus_A)

    aligned = minus_structure.copy()
    aligned.coord = aligned.coord @ R + t

    _, aligned_A = matched_ca(plus_structure, aligned, ANCHOR_CHAIN)
    plus_B, aligned_B = matched_ca(plus_structure, aligned, BINDER_B_CHAIN)

    return (
        aligned,
        rmsd(plus_A, aligned_A),
        rmsd(plus_B, aligned_B),
        float(np.linalg.norm(plus_B.mean(0) - aligned_B.mean(0))),
    )


def get_pair(summary, key, i, j):
    try:
        return safe_float(summary[key][i][j])
    except Exception:
        return float('nan')

def get_chain(summary, key, i):
    try:
        return safe_float(summary[key][i])
    except Exception:
        return float('nan')


def get_scalar(summary, key):
    return safe_float(summary.get(key, float('nan')))


def get_clash(summary):
    x = summary.get('has_clash', summary.get('clash', False))
    if isinstance(x, str):
        return x.strip().lower() in {'1', 'true', 'yes'}
    return bool(x)


def finite_max(x):
    x = [v for v in x if math.isfinite(v)]
    return max(x) if x else float('nan')


def finite_min(x):
    x = [v for v in x if math.isfinite(v)]
    return min(x) if x else float('nan')


def plus_metrics(s):
    return {
        'plus_AB_pair_iptm': get_pair(s, 'chain_pair_iptm', 0, 1),
        'plus_B_RAP_pair_iptm': get_pair(s, 'chain_pair_iptm', 1, RAP_CHAIN_INDEX),
        'plus_AB_pair_pae_min': get_pair(s, 'chain_pair_pae_min', 0, 1),
        'plus_B_RAP_pair_pae_min': get_pair(s, 'chain_pair_pae_min', 1, RAP_CHAIN_INDEX),
        'plus_B_chain_iptm': get_chain(s, 'chain_iptm', 1),
        'plus_B_chain_ptm': get_chain(s, 'chain_ptm', 1),
        'plus_ranking_score': get_scalar(s, 'ranking_score'),
        'plus_iptm': get_scalar(s, 'iptm'),
        'plus_fraction_disordered': get_scalar(s, 'fraction_disordered'),
        'plus_clash': int(get_clash(s)),
    }

def minus_worst_case_metrics(samples):
    ss = [x['summary'] for x in samples]
    return {
        # Conservative unwanted -RAP state: strongest A-B confidence seen in any sample
        'minus_max_AB_pair_iptm': finite_max([get_pair(s, 'chain_pair_iptm', 0, 1) for s in ss]),
        'minus_min_AB_pair_pae_min': finite_min([get_pair(s, 'chain_pair_pae_min', 0, 1) for s in ss]),
        'minus_max_iptm': finite_max([get_scalar(s, 'iptm') for s in ss]),
        'minus_max_ranking_score': finite_max([get_scalar(s, 'ranking_score') for s in ss]),
        'minus_any_clash': int(any(get_clash(s) for s in ss)),
    }


def add_percentile(rows, key, high=True):
    vals = [safe_float(r[key]) for r in rows]
    finite = [v for v in vals if math.isfinite(v)]
    for r in rows:
        x = safe_float(r[key])
        if not math.isfinite(x) or not finite:
            r[f'_pct_{key}'] = 0.5
            continue
        if high:
            better_side = sum(v < x for v in finite)
        else:
            better_side = sum(v > x for v in finite)
        equal = sum(v == x for v in finite)
        r[f'_pct_{key}'] = (better_side + 0.5 * equal) / len(finite)


def add_soft_scores(rows):
    # Soft-ranking:
    # 25% global ipTM
    # 25% delta ipTM (+RAP - strongest -RAP)
    # 50% A-B chain-pair ipTM

    for r in rows:
        r['delta_iptm'] = (
            r['plus_iptm']
            - r['minus_max_iptm']
        )

    weights = [
        ('plus_iptm',         0.25, True),
        ('delta_iptm',        0.25, True),
        ('plus_AB_pair_iptm', 0.50, True),
    ]

    for key, _, high in weights:
        add_percentile(rows, key, high)

    for r in rows:
        r['soft_score'] = sum(
            weight * r[f'_pct_{key}']
            for key, weight, _ in weights
        )


def find_minus_dir(candidate):
    options = [
        MINUS_ROOT / f'{candidate}_no_RAP',
        MINUS_ROOT / f'{candidate}_minus_RAP',
        MINUS_ROOT / f'{candidate}_without_RAP',
        MINUS_ROOT / candidate,
    ]
    hits = [p for p in options if p.is_dir()]
    if len(hits) == 1:
        return hits[0]
    fuzzy = [p for p in MINUS_ROOT.iterdir() if p.is_dir() and p.name.startswith(candidate)]
    if len(fuzzy) == 1:
        return fuzzy[0]
    raise RuntimeError(f'{candidate}: cannot uniquely identify -RAP directory')


def remap_chain_ids(structure, chain_map):
    out = structure.copy()
    chain_id = out.chain_id.copy()
    for old, new in chain_map.items():
        chain_id[chain_id == old] = new
    out.chain_id = chain_id
    return out

def concatenate_structures(structures):
    structures = [s for s in structures if len(s) > 0]
    if not structures:
        raise ValueError('No structures to concatenate')

    categories = list(structures[0].get_annotation_categories())
    total_len = sum(len(s) for s in structures)
    out = struc.AtomArray(total_len)
    out.coord = np.concatenate([s.coord for s in structures], axis=0)

    for cat in categories:
        values = np.concatenate([s.get_annotation(cat) for s in structures], axis=0)
        out.set_annotation(cat, values)

    return out


def make_overlay_structure(plus_structure, aligned_minus_structure):
    plus_out = plus_structure.copy()
    minus_out = remap_chain_ids(aligned_minus_structure, OVERLAY_MINUS_CHAIN_MAP)
    return concatenate_structures([plus_out, minus_out])


def clean_row(row):
    out = {}
    for k, v in row.items():
        if k.startswith('_'):
            continue
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def write_csv(path, rows):
    rows = [clean_row(r) for r in rows]
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def prepare_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob('Rank*.cif'):
        p.unlink()

def main():
    if not PLUS_ROOT.is_dir():
        raise FileNotFoundError(PLUS_ROOT)
    if not MINUS_ROOT.is_dir():
        raise FileNotFoundError(MINUS_ROOT)

    OUT.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(OUT_PLUS)
    prepare_output_dir(OUT_MINUS)
    prepare_output_dir(OUT_ALIGN)

    plus_dirs = sorted(
        p for p in PLUS_ROOT.iterdir()
        if p.is_dir() and p.name.endswith('_plus_RAP')
    )

    passed = []

    print('=== Structural RAP-dependence filter ===')
    print(f'Hard filter: min(-RAP Binder B RMSD after Binder A alignment) >= {B_RMSD_CUTOFF:.1f} A')

    for i, plus_dir in enumerate(plus_dirs, 1):
        candidate = plus_dir.name.removesuffix('_plus_RAP')
        minus_dir = find_minus_dir(candidate)

        plus_samples = collect_samples(plus_dir)
        minus_samples = collect_samples(minus_dir)
        if not plus_samples or not minus_samples:
            continue

        for m in minus_samples:
            m['_structure'] = load_structure(m['cif'])

        minus_metrics = minus_worst_case_metrics(minus_samples)

        for p in plus_samples:
            plus_structure = load_structure(p['cif'])
            comparisons = []

            for m in minus_samples:
                aligned, a_rmsd, b_rmsd, b_shift = align_minus_to_plus(
                    plus_structure, m['_structure']
                )
                comparisons.append((b_rmsd, a_rmsd, b_shift, m, aligned))

            # Most conservative comparator = closest -RAP state
            b_rmsd, a_rmsd, b_shift, matched_minus, aligned_minus = min(
                comparisons, key=lambda x: x[0]
            )

            if b_rmsd < B_RMSD_CUTOFF:
                continue

            row = {
                'candidate': candidate,
                'plus_seed': p['seed'],
                'plus_sample': p['sample'],
                'minus_seed': matched_minus['seed'],
                'minus_sample': matched_minus['sample'],
                'A_CA_RMSD_after_align': a_rmsd,
                'B_CA_RMSD_after_A_align': b_rmsd,
                'B_CA_centroid_shift': b_shift,
                'plus_cif': p['cif'],
                'minus_cif': matched_minus['cif'],
                **plus_metrics(p['summary']),
                **minus_metrics,
            }
            row['_overlay_structure'] = make_overlay_structure(plus_structure, aligned_minus)
            passed.append(row)

        if i % 10 == 0 or i == len(plus_dirs):
            print(f'[{i}/{len(plus_dirs)}] passing +RAP samples: {len(passed)}')

    if not passed:
        print('No sample passed the structural RAP-dependence filter.')
        return

    add_soft_scores(passed)
    passed.sort(key=lambda r: r['soft_score'], reverse=True)
    write_csv(PASS_CSV, passed)

    # One AF3 representative per design/candidate
    best_by_candidate = {}
    for r in passed:
        best_by_candidate.setdefault(r['candidate'], r)

    ranked = sorted(best_by_candidate.values(), key=lambda r: r['soft_score'], reverse=True)
    top = ranked[:TOP_N]

    rank_rows = []
    for rank, row in enumerate(top, 1):
        plus_dst = OUT_PLUS / f'Rank{rank}.cif'
        minus_dst = OUT_MINUS / f'Rank{rank}.cif'
        align_dst = OUT_ALIGN / f'Rank{rank}.cif'

        shutil.copy2(row['plus_cif'], plus_dst)
        shutil.copy2(row['minus_cif'], minus_dst)
        write_structure(align_dst, row['_overlay_structure'])

        x = dict(row)
        x['rank'] = rank
        x['plus_saved'] = plus_dst
        x['minus_saved'] = minus_dst
        x['align_saved'] = align_dst
        rank_rows.append(x)

    clean = []
    for r in rank_rows:
        x = clean_row(r)
        rank = x.pop('rank')
        clean.append({'rank': rank, **x})

    with open(RANK_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
        w.writeheader()
        w.writerows(clean)

    print('\n=== DONE ===')
    print(f'Passing samples: {len(passed)}')
    print(f'Passing unique candidates: {len(ranked)}')
    print(f'Saved Top: {len(top)}')
    print(f'Output: {OUT}\n')

    print('align/RankN.cif = merged overlay structure')
    print('  +RAP chains: A/B/C  (color green)')
    print('  -RAP chains: X/Y    (color red)')

    for rank, row in enumerate(top, 1):
        print(
            f"Rank{rank}: {row['candidate']} | "
            f"+sample={row['plus_sample']} -sample={row['minus_sample']} | "
            f"B_RMSD={row['B_CA_RMSD_after_A_align']:.2f} A | "
            f"score={row['soft_score']:.4f}"
        )


if __name__ == '__main__':
    main()