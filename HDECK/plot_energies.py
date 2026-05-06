#!/usr/bin/env python3
"""
parse_md_log.py — Parse FFX/MD log files and analyze energy drift.

Usage:
  python parse_md_log.py /path/to/run.log -o energies.xlsx

What it does:
  • Parses STEP lines: STEP N T=... KE_atom=... KE_esv=... POT=... TOTAL=...
  • Builds a DataFrame with step, T, KE_atom, KE_esv, POT, TOTAL.
  • Computes simple drift diagnostics (global slope, tail slope, rolling mean).
  • Writes everything to an Excel file with an embedded line chart (TOTAL + rolling mean).
  • Emits a short textual assessment to stdout.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

STEP_RE = re.compile(
    r'^STEP\s+(?P<step>\d+)\s+'
    r'T=(?P<T>[-+]?\d+(?:\.\d+)?)\s+'
    r'KE_atom=\s*(?P<ke_atom>[-+]?\d+(?:\.\d+)?)\s+'
    r'KE_esv=\s*(?P<ke_esv>[-+]?\d+(?:\.\d+)?)\s+'
    r'POT=\s*(?P<pot>[-+]?\d+(?:\.\d+)?)\s+'
    r'TOTAL=\s*(?P<total>[-+]?\d+(?:\.\d+)?)\s*$'
)

def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = STEP_RE.match(line.strip())
            if m:
                rows.append({
                    'step': int(m.group('step')),
                    'T': float(m.group('T')),
                    'KE_atom': float(m.group('ke_atom')),
                    'KE_esv': float(m.group('ke_esv')),
                    'POT': float(m.group('pot')),
                    'TOTAL': float(m.group('total')),
                })
    if not rows:
        raise ValueError("No STEP lines found. Make sure the log has lines like: "
                         "'STEP 100 T=263.44 KE_atom= ... KE_esv= ... POT= ... TOTAL= ...'")
    df = pd.DataFrame(rows).sort_values('step').reset_index(drop=True)
    return df

def analyze(df: pd.DataFrame, roll_window: int = 10) -> dict:
    # Linear trend (global and last quartile).
    x = df['step'].values.astype(float)
    y = df['TOTAL'].values.astype(float)
    slope = np.polyfit(x, y, 1)[0] if len(df) >= 2 else 0.0

    # Tail slope: last 25% of points (min 5 points).
    n = len(df)
    start_tail = max(0, int(n*0.75))
    tail_x = x[start_tail:]
    tail_y = y[start_tail:]
    if len(tail_x) >= 5:
        slope_tail = np.polyfit(tail_x, tail_y, 1)[0]
    else:
        slope_tail = np.nan

    # Rolling mean and detrended series for oscillation check.
    df['TOTAL_rolling'] = df['TOTAL'].rolling(roll_window, min_periods=1, center=False).mean()
    # Detrend using global slope and intercept.
    if len(df) >= 2:
        coeffs = np.polyfit(x, y, 1)
        trend = coeffs[0]*x + coeffs[1]
        detrended = y - trend
    else:
        detrended = y - y.mean()
    df['TOTAL_detrended'] = detrended

    # Oscillation heuristic: zero-crossing rate of detrended around its median.
    med = np.median(df['TOTAL_detrended'])
    signs = np.sign(df['TOTAL_detrended'] - med)
    zero_crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    zc_rate = zero_crossings / max(1, len(df)-1)

    # Stability metric: std of detrended vs full std.
    std_detrended = float(np.std(detrended)) if len(df) > 1 else 0.0
    std_total = float(np.std(y)) if len(df) > 1 else 0.0

    # Simple classification
    # Thresholds are heuristic and unit-agnostic;
    # slope per 1000 steps compared to detrended std.
    slope_per_1000 = slope * 1000.0
    slope_tail_per_1000 = slope_tail * 1000.0 if not np.isnan(slope_tail) else np.nan
    ref = std_detrended if std_detrended > 0 else (abs(y.mean()) * 1e-6 + 1e-12)

    if abs(slope_tail) <= 0.1 * (ref / max(1.0, (x[-1]-x[0]))) if not np.isnan(slope_tail) else False:
        verdict = "drift flattens (tail slope ~ 0)"
    elif abs(slope) > 0.3 * (ref / max(1.0, (x[-1]-x[0]))):
        verdict = "persistent drift (nonzero global slope)"
    else:
        verdict = "mixed / inconclusive"

    if zc_rate > 0.25 and std_detrended > 0.05 * max(1e-6, abs(y.mean())):
        behavior = "oscillatory about a trend"
    else:
        behavior = "weakly oscillatory or monotonic"

    return {
        'slope': float(slope),
        'slope_per_1000': float(slope_per_1000),
        'slope_tail': float(slope_tail),
        'slope_tail_per_1000': float(slope_tail_per_1000) if not np.isnan(slope_tail) else np.nan,
        'zero_crossing_rate': float(zc_rate),
        'std_total': float(std_total),
        'std_detrended': float(std_detrended),
        'verdict': verdict,
        'behavior': behavior,
        'df': df,
    }

def to_excel(analysis: dict, out_path: Path):
    df = analysis['df']
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Energies')
        # Write a summary sheet
        summary = pd.DataFrame({
            'metric': [
                'slope (energy/step)',
                'slope per 1000 steps',
                'tail slope (last 25%)',
                'tail slope per 1000 steps',
                'zero-crossing rate (detrended)',
                'std(TOTAL)',
                'std(TOTAL_detrended)',
                'verdict',
                'behavior',
            ],
            'value': [
                analysis['slope'],
                analysis['slope_per_1000'],
                analysis['slope_tail'],
                analysis['slope_tail_per_1000'],
                analysis['zero_crossing_rate'],
                analysis['std_total'],
                analysis['std_detrended'],
                analysis['verdict'],
                analysis['behavior'],
            ]
        })
        summary.to_excel(writer, index=False, sheet_name='Summary')

        # Add an Excel chart for TOTAL and rolling mean
        workbook  = writer.book
        worksheet = writer.sheets['Energies']

        # Determine the data range (Excel is 1-indexed for sheets).
        nrows = len(df)
        # Columns: A step, B T, C KE_atom, D KE_esv, E POT, F TOTAL, G TOTAL_rolling, H TOTAL_detrended
        chart = workbook.add_chart({'type': 'line'})
        chart.add_series({
            'name':       'TOTAL',
            'categories': ['Energies', 1, 0, nrows, 0],  # step
            'values':     ['Energies', 1, 5, nrows, 5],  # TOTAL
        })
        chart.add_series({
            'name':       'TOTAL_rolling',
            'categories': ['Energies', 1, 0, nrows, 0],  # step
            'values':     ['Energies', 1, 6, nrows, 6],  # rolling
        })
        chart.set_title({'name': 'Total Energy vs Step'})
        chart.set_x_axis({'name': 'Step'})
        chart.set_y_axis({'name': 'Energy'})
        chart.set_legend({'position': 'bottom'})

        # Insert chart below the table
        worksheet.insert_chart('J2', chart)

def main():
    parser = argparse.ArgumentParser(description="Parse MD log and analyze energy drift.")
    parser.add_argument('logfile', type=Path, help='Path to the MD log file')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output Excel path (default: <logname>_energies.xlsx)')
    parser.add_argument('--roll', type=int, default=10, help='Rolling mean window (steps)')
    args = parser.parse_args()

    out = args.output or args.logfile.with_name(args.logfile.stem + '_energies.xlsx')

    df = parse_log(args.logfile)
    analysis = analyze(df, roll_window=args.roll)
    to_excel(analysis, out)

    print("Parsed {} STEP lines.".format(len(df)))
    print("Global slope (energy/step): {:.6g}".format(analysis['slope']))
    print("Tail slope (last 25%):      {:.6g}".format(analysis['slope_tail']))
    print("Zero-crossing rate:         {:.3f}".format(analysis['zero_crossing_rate']))
    print("Verdict: {} | {}".format(analysis['verdict'], analysis['behavior']))
    print("Wrote Excel to:", out)

if __name__ == '__main__':
    main()
