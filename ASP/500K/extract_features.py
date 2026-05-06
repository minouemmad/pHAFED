"""
extract_features.py
====================
Parse all .esv files for a residue across temperatures (PHAFED) and for
the repex reference run (T*=300 K), and compute a rich feature vector from
each ESV's 2-D lambda×zeta histogram.

Usage
-----
    python extract_features.py --base_dir /path/to/ASP \
                               --residue ASP \
                               --temps 500 750 1000 1250 1500 \
                               --repex_dir /path/to/ASP/repex \
                               --out_features features_ASP.json

Directory convention (mirrors your existing layout):
    <base_dir>/
        <temp>K/        e.g. 500K/
            0/          pH-window directories (0-indexed integers)
                ASP_penta.esv
            1/
                ASP_penta.esv
            ...
        repex/
            0/
                ASP_penta.esv
            ...

Each .esv file contains ONE OR MORE ESV blocks.  For ASP/GLU the first ESV
is lambda (protonation) and the second is zeta (tautomer).  Both share the
same 10×10 histogram layout:
    rows  → lambda bins  [0.0-0.1] … [0.9-1.0]
    cols  → X (zeta) bins [0.0-0.1] … [0.9-1.0]

Features extracted per (residue, temperature, pH_window, ESV_index):
    --- Convergence / sampling time ---
    total_counts            : sum of all histogram bins
    min_window_frac         : min(row_sum) / max(row_sum)  (uniformity of lambda)
    --- Lambda marginal (row sums) ---
    frac_deprot             : counts in lambda[0.0-0.1] / edge_counts
    frac_prot               : counts in lambda[0.9-1.0] / edge_counts
    frac_lambda_intermediate: counts in lambda[0.1-0.9] / total
    lambda_bimodality       : (frac_deprot + frac_prot) / total  (how edge-concentrated)
    lambda_asymmetry        : |frac_deprot - frac_prot| / (frac_deprot + frac_prot + 1e-12)
    --- Zeta marginal (column sums) ---
    frac_zeta_left          : counts in zeta[0.0-0.1] / total
    frac_zeta_right         : counts in zeta[0.9-1.0] / total
    frac_zeta_intermediate  : counts in zeta[0.1-0.9] / total
    zeta_symmetry           : 1 - |frac_zeta_left - frac_zeta_right| / (frac_zeta_left + frac_zeta_right + 1e-12)
    --- Joint 2-D structure ---
    frac_interior           : counts where lambda∈(0.1,0.9) AND zeta∈(0.1,0.9) / total
    frac_corner_deprot_left : lambda[0.0-0.1], zeta[0.0-0.1] / total
    frac_corner_deprot_right: lambda[0.0-0.1], zeta[0.9-1.0] / total
    frac_corner_prot_left   : lambda[0.9-1.0], zeta[0.0-0.1] / total
    frac_corner_prot_right  : lambda[0.9-1.0], zeta[0.9-1.0] / total
    --- Temperature encoding ---
    temperature             : T* in Kelvin (300 for repex)
    pH                      : pH of this window (parsed from ESV header)
"""

import re
import os
import json
import glob
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# ESV PARSING
# ---------------------------------------------------------------------------

def parse_esv_file(filepath):
    """
    Parse a single .esv file and return a list of ESV dicts, one per ESV block.

    Each dict contains:
        esv_index    : int
        esv_label    : str  (e.g. "4-ASD")
        pH           : float
        histogram    : np.ndarray shape (10, 10)   rows=lambda, cols=zeta/X
    """
    text = Path(filepath).read_text()

    # Split on "ESV:" markers to get individual ESV blocks
    # We find each histogram block by the header line pattern
    esv_pattern = re.compile(
        r'ESV:\s+(\S+)\s+\((\d+)\)\s+pH:\s+([\d.]+)'
        r'.*?'
        r'(?=ESV:|\Z)',
        re.DOTALL
    )

    # The histogram table follows the column header line (X  [0.0-0.1] ...)
    hist_pattern = re.compile(
        r'\[[\d.]+\-[\d.]+\]\s+((?:\d+\s+){9}\d+)',
    )

    esvs = []
    for m in esv_pattern.finditer(text):
        label     = m.group(1)
        esv_idx   = int(m.group(2))
        pH        = float(m.group(3))
        block     = m.group(0)

        rows = hist_pattern.findall(block)
        if len(rows) != 10:
            # Malformed or truncated; skip
            continue

        hist = np.array([[int(v) for v in row.split()] for row in rows],
                        dtype=np.float64)  # shape (10, 10)

        esvs.append({
            'esv_index': esv_idx,
            'esv_label': label,
            'pH':        pH,
            'histogram': hist,
        })

    return esvs


def compute_features(hist, temperature, pH, esv_index, esv_label):
    """
    Compute the feature vector from a 10×10 lambda×zeta histogram.
    Returns a flat dict of scalar features.
    """
    total = hist.sum()
    if total == 0:
        return None   # empty histogram — skip

    # Row (lambda) marginal
    lambda_marginal = hist.sum(axis=1)   # shape (10,)
    # Col (zeta) marginal
    zeta_marginal   = hist.sum(axis=0)   # shape (10,)

    edge_lambda = lambda_marginal[0] + lambda_marginal[9]
    edge_zeta   = zeta_marginal[0]   + zeta_marginal[9]

    # Safe division helpers
    def safe_div(a, b):
        return float(a) / float(b) if b > 0 else 0.0

    frac_deprot = safe_div(lambda_marginal[0], total)
    frac_prot   = safe_div(lambda_marginal[9], total)
    frac_lambda_intermediate = safe_div(lambda_marginal[1:9].sum(), total)
    lambda_bimodality = safe_div(edge_lambda, total)
    lambda_asymmetry  = abs(frac_deprot - frac_prot) / (frac_deprot + frac_prot + 1e-12)

    frac_zeta_left         = safe_div(zeta_marginal[0], total)
    frac_zeta_right        = safe_div(zeta_marginal[9], total)
    frac_zeta_intermediate = safe_div(zeta_marginal[1:9].sum(), total)
    zeta_symmetry = 1.0 - abs(frac_zeta_left - frac_zeta_right) / \
                    (frac_zeta_left + frac_zeta_right + 1e-12)

    # Interior: lambda ∈ [0.1,0.9) AND zeta ∈ [0.1,0.9)
    interior = hist[1:9, 1:9].sum()
    frac_interior = safe_div(interior, total)

    # Corner occupancies (lambda end-state × zeta end-state)
    frac_corner_deprot_left  = safe_div(hist[0, 0], total)
    frac_corner_deprot_right = safe_div(hist[0, 9], total)
    frac_corner_prot_left    = safe_div(hist[9, 0], total)
    frac_corner_prot_right   = safe_div(hist[9, 9], total)

    # Uniformity across pH windows: measured as min/max row sum ratio
    # (Only meaningful when aggregating across windows; recorded per-window here)
    row_sums = lambda_marginal
    min_window_frac = safe_div(row_sums.min(), row_sums.max())

    return {
        # Metadata
        'temperature':              float(temperature),
        'pH':                       float(pH),
        'esv_index':                int(esv_index),
        'esv_label':                str(esv_label),
        # Convergence
        'total_counts':             float(total),
        'min_window_frac':          min_window_frac,
        # Lambda marginal
        'frac_deprot':              frac_deprot,
        'frac_prot':                frac_prot,
        'frac_lambda_intermediate': frac_lambda_intermediate,
        'lambda_bimodality':        lambda_bimodality,
        'lambda_asymmetry':         lambda_asymmetry,
        # Zeta marginal
        'frac_zeta_left':           frac_zeta_left,
        'frac_zeta_right':          frac_zeta_right,
        'frac_zeta_intermediate':   frac_zeta_intermediate,
        'zeta_symmetry':            zeta_symmetry,
        # Joint 2-D
        'frac_interior':            frac_interior,
        'frac_corner_deprot_left':  frac_corner_deprot_left,
        'frac_corner_deprot_right': frac_corner_deprot_right,
        'frac_corner_prot_left':    frac_corner_prot_left,
        'frac_corner_prot_right':   frac_corner_prot_right,
    }


# ---------------------------------------------------------------------------
# DIRECTORY TRAVERSAL
# ---------------------------------------------------------------------------

ESV_GLOB = "*.esv"

def collect_esv_paths(run_dir):
    """
    Given a run directory (e.g. 500K/ or repex/), find all .esv files
    inside numbered pH-window subdirectories (0, 1, 2, ...).
    Returns list of (window_id: int, filepath: str).
    """
    results = []
    run_dir = Path(run_dir)
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit():
            window_id = int(sub.name)
            for esv_path in sorted(sub.glob(ESV_GLOB)):
                results.append((window_id, str(esv_path)))
    return results


def extract_all_features(base_dir, residue, temps, repex_dir):
    """
    Returns a list of feature dicts — one per (temp, pH_window, ESV_block).
    """
    all_features = []

    # PHAFED runs
    for T in temps:
        run_dir = Path(base_dir) / f"{T}K"
        if not run_dir.is_dir():
            print(f"  [WARN] {run_dir} not found, skipping.")
            continue

        paths = collect_esv_paths(run_dir)
        if not paths:
            print(f"  [WARN] No ESV files found in {run_dir}")
            continue

        for window_id, esv_path in paths:
            esvs = parse_esv_file(esv_path)
            for esv in esvs:
                feats = compute_features(
                    esv['histogram'], T, esv['pH'],
                    esv['esv_index'], esv['esv_label']
                )
                if feats is not None:
                    feats['residue']   = residue
                    feats['run_type']  = 'phafed'
                    feats['window_id'] = window_id
                    feats['esv_file']  = esv_path
                    all_features.append(feats)

        print(f"  [OK] {residue} T={T}K — {len(paths)} windows processed")

    # Repex run (treated as T*=300)
    repex_dir = Path(repex_dir)
    if repex_dir.is_dir():
        paths = collect_esv_paths(repex_dir)
        for window_id, esv_path in paths:
            esvs = parse_esv_file(esv_path)
            for esv in esvs:
                feats = compute_features(
                    esv['histogram'], 300, esv['pH'],
                    esv['esv_index'], esv['esv_label']
                )
                if feats is not None:
                    feats['residue']   = residue
                    feats['run_type']  = 'repex'
                    feats['window_id'] = window_id
                    feats['esv_file']  = esv_path
                    all_features.append(feats)
        print(f"  [OK] {residue} repex — {len(paths)} windows processed")
    else:
        print(f"  [WARN] repex dir {repex_dir} not found.")

    return all_features


# ---------------------------------------------------------------------------
# AGGREGATED WINDOW-LEVEL FEATURES
# ---------------------------------------------------------------------------

def aggregate_per_temperature(features):
    """
    For each (residue, temperature, run_type), aggregate across pH windows
    to compute cross-window convergence diagnostics:
        - count_ratio_min_max : min(total_counts across windows) / max(...)
        - monotonicity_score  : fraction of consecutive pH windows where
                                frac_deprot increases as pH increases
                                (for lambda ESV only)
    Adds these as additional keys to each feature dict (broadcast back).
    """
    from collections import defaultdict
    import math

    # Group by (residue, temperature, run_type, esv_index)
    groups = defaultdict(list)
    for f in features:
        key = (f['residue'], f['temperature'], f['run_type'], f['esv_index'])
        groups[key].append(f)

    for key, group in groups.items():
        # Sort windows by pH
        group_sorted = sorted(group, key=lambda x: x['pH'])
        counts = [g['total_counts'] for g in group_sorted]
        max_c  = max(counts) if counts else 1.0
        min_c  = min(counts) if counts else 0.0
        count_ratio = min_c / max_c if max_c > 0 else 0.0

        # Monotonicity: deprot fraction should INCREASE as pH increases
        deprot_vals = [g['frac_deprot'] for g in group_sorted]
        if len(deprot_vals) > 1:
            n_monotone = sum(
                1 for i in range(len(deprot_vals)-1)
                if deprot_vals[i] <= deprot_vals[i+1]
            )
            monotonicity = n_monotone / (len(deprot_vals) - 1)
        else:
            monotonicity = 1.0

        for f in group:
            f['cross_window_count_ratio']  = count_ratio
            f['cross_window_monotonicity'] = monotonicity

    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract ESV histogram features")
    ap.add_argument('--base_dir',  required=True,
                    help="Root directory containing <temp>K/ subdirs")
    ap.add_argument('--residue',   default='ASP',
                    help="Residue name (for labeling, e.g. ASP or GLU)")
    ap.add_argument('--temps',     nargs='+', type=int,
                    default=[500, 750, 1000, 1250, 1500],
                    help="List of PHAFED temperatures in K")
    ap.add_argument('--repex_dir', required=True,
                    help="Path to repex run directory (treated as T*=300K)")
    ap.add_argument('--out_features', default='features.json',
                    help="Output JSON file path")
    args = ap.parse_args()

    print(f"\n=== Extracting features for {args.residue} ===")
    features = extract_all_features(
        args.base_dir, args.residue, args.temps, args.repex_dir
    )
    features = aggregate_per_temperature(features)

    # Remove non-serialisable keys before saving
    for f in features:
        f.pop('esv_file', None)

    with open(args.out_features, 'w') as fh:
        json.dump(features, fh, indent=2)

    print(f"\nSaved {len(features)} feature records → {args.out_features}")


if __name__ == '__main__':
    main()
