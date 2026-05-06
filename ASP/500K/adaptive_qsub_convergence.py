#!/usr/bin/env python3
"""
Adaptive 1 ns submission controller for PHAFED windows on SGE/UGE.

Workflow:
1) Submit one chunk job via qsub (job script should represent ~1 ns).
2) Wait for completion.
3) Parse the local .esv and compute convergence metrics.
4) If not converged, submit the same job script again.

Assumptions:
- The job script runs in-place and updates the same .esv path.
- Restart/checkpoint behavior is already configured in the job script.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse project ESV parser and feature extractor.
from extract_features import parse_esv_file, compute_features


TRACKED_KEYS = [
    "frac_deprot",
    "frac_prot",
    "frac_lambda_intermediate",
    "lambda_bimodality",
    "lambda_asymmetry",
    "zeta_symmetry",
    "frac_interior",
]


def run_cmd(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def submit_job(qsub_cmd, job_script, cwd=None):
    # In batch mode we submit from the window directory, so pass local script
    # name to mirror manual `cd <window>; qsub dynamic.job` usage.
    if cwd is not None:
        script_arg = Path(job_script).name
    else:
        script_arg = str(Path(job_script).resolve())
    res = run_cmd([qsub_cmd, script_arg], cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"qsub failed: {res.stderr.strip() or res.stdout.strip()}")

    text = (res.stdout or "") + "\n" + (res.stderr or "")
    # Typical SGE output: "Your job 12345 (\"name\") has been submitted"
    m = re.search(r"\bjob\s+(\d+)\b", text)
    if not m:
        raise RuntimeError(f"Could not parse job id from qsub output: {text.strip()}")
    return m.group(1), text.strip()


def job_is_active(qstat_cmd, job_id):
    # qstat -j <id> returns 0 while present, non-zero when gone on most SGE setups.
    res = run_cmd([qstat_cmd, "-j", str(job_id)])
    if res.returncode == 0:
        return True

    # Fallback: qstat table output may still show the job.
    table = run_cmd([qstat_cmd])
    if table.returncode == 0 and re.search(rf"\b{re.escape(str(job_id))}\b", table.stdout or ""):
        return True

    return False


def wait_for_job(qstat_cmd, job_id, poll_seconds):
    while True:
        if not job_is_active(qstat_cmd, job_id):
            return
        time.sleep(poll_seconds)


def extract_esv_metrics(esv_path, esv_index):
    esv_blocks = parse_esv_file(esv_path)
    if not esv_blocks:
        raise RuntimeError(f"No ESV blocks parsed from {esv_path}")

    block = None
    for b in esv_blocks:
        if int(b.get("esv_index", -999)) == esv_index:
            block = b
            break
    if block is None:
        available = [int(b.get("esv_index", -1)) for b in esv_blocks]
        raise RuntimeError(
            f"ESV index {esv_index} not found in {esv_path}. Available indices: {available}"
        )

    feats = compute_features(
        block["histogram"],
        temperature=0.0,
        pH=float(block["pH"]),
        esv_index=int(block["esv_index"]),
        esv_label=str(block["esv_label"]),
    )
    if feats is None:
        raise RuntimeError(f"Empty histogram for ESV index {esv_index} in {esv_path}")
    return feats


def evaluate_convergence(current, previous, stable_streak, args):
    reasons = []

    if float(current["total_counts"]) < args.min_total_counts:
        reasons.append(
            f"total_counts {current['total_counts']:.0f} < min_total_counts {args.min_total_counts:.0f}"
        )

    deltas = {}
    if previous is not None:
        for key in TRACKED_KEYS:
            deltas[key] = abs(float(current[key]) - float(previous[key]))

        stable = (
            deltas["frac_deprot"] <= args.max_d_frac_deprot
            and deltas["frac_prot"] <= args.max_d_frac_prot
            and deltas["frac_lambda_intermediate"] <= args.max_d_frac_lambda_intermediate
            and deltas["lambda_bimodality"] <= args.max_d_lambda_bimodality
            and deltas["lambda_asymmetry"] <= args.max_d_lambda_asymmetry
            and deltas["zeta_symmetry"] <= args.max_d_zeta_symmetry
            and deltas["frac_interior"] <= args.max_d_frac_interior
        )

        if stable:
            stable_streak += 1
        else:
            stable_streak = 0
            reasons.append("feature deltas exceed threshold")
    else:
        stable_streak = 0
        reasons.append("no previous chunk for stability comparison")

    converged = (
        len(reasons) == 0
        and stable_streak >= args.required_stable_chunks
    )

    if not converged and stable_streak < args.required_stable_chunks:
        reasons.append(
            f"stable_streak {stable_streak} < required {args.required_stable_chunks}"
        )

    return converged, stable_streak, deltas, reasons


def load_history(history_path):
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_history(history_path, history):
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


def find_esv_in_window(window_dir, esv_name=None):
    if esv_name:
        esv_path = window_dir / esv_name
        if esv_path.is_file():
            return esv_path
        raise FileNotFoundError(f"ESV not found in {window_dir}: {esv_name}")

    candidates = sorted(window_dir.glob("*.esv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .esv files found in {window_dir}")
    names = [p.name for p in candidates]
    raise RuntimeError(
        f"Multiple .esv files found in {window_dir}: {names}. Use --window-esv-name"
    )


def discover_window_dirs(windows_root, windows_list=None):
    windows = []
    # Keep numeric window directories in numeric order and all others after.
    def _sort_key(path_obj):
        return (0, int(path_obj.name)) if path_obj.name.isdigit() else (1, path_obj.name)

    for child in sorted(windows_root.iterdir(), key=_sort_key):
        if child.is_dir() and child.name.isdigit():
            windows.append(child)

    if windows_list:
        wanted = {str(w) for w in windows_list}
        windows = [w for w in windows if w.name in wanted]

    return windows


def run_single_window(args, job_script, esv_path, history_path, window_label, submit_cwd=None):
    history = load_history(history_path)
    previous = history[-1]["metrics"] if history else None
    stable_streak = int(history[-1].get("stable_streak", 0)) if history else 0
    chunks_done = len(history)

    if args.dry_run:
        metrics = extract_esv_metrics(esv_path, args.esv_index)
        converged, stable_streak, deltas, reasons = evaluate_convergence(
            metrics, previous, stable_streak, args
        )
        print(f"\n[{window_label}] Dry run metrics:")
        print(json.dumps(metrics, indent=2))
        print(f"[{window_label}] Converged: {converged}")
        print(f"[{window_label}] Reasons: {reasons}")
        print(f"[{window_label}] Deltas:")
        print(json.dumps(deltas, indent=2))
        return converged, chunks_done

    while chunks_done < args.max_chunks:
        print(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] "
            f"[{window_label}] Submitting chunk {chunks_done + 1}/{args.max_chunks}"
        )
        job_id, submit_text = submit_job(args.qsub_cmd, job_script, cwd=submit_cwd)
        print(f"[{window_label}] Submitted job {job_id}: {submit_text}")

        wait_for_job(args.qstat_cmd, job_id, args.poll_seconds)
        print(f"[{window_label}] Job {job_id} completed. Evaluating convergence from {esv_path} ...")

        metrics = extract_esv_metrics(esv_path, args.esv_index)
        converged, stable_streak, deltas, reasons = evaluate_convergence(
            metrics, previous, stable_streak, args
        )

        rec = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "chunk_index": chunks_done + 1,
            "job_id": job_id,
            "metrics": metrics,
            "deltas": deltas,
            "stable_streak": stable_streak,
            "converged": converged,
            "reasons": reasons,
        }
        history.append(rec)
        save_history(history_path, history)

        print(f"[{window_label}] Converged: {converged}")
        print(f"[{window_label}] Stable streak: {stable_streak}")
        if reasons:
            print(f"[{window_label}] Reasons:")
            for reason in reasons:
                print(f"  - {reason}")

        if converged:
            print(f"\n[{window_label}] Convergence criteria met. Stopping submissions.")
            return True, chunks_done + 1

        previous = metrics
        chunks_done += 1

    print(f"\n[{window_label}] Reached max chunks without convergence.")
    return False, chunks_done


def main():
    ap = argparse.ArgumentParser(
        description="Submit 1 ns chunks repeatedly until ESV convergence criteria are met"
    )
    ap.add_argument("--job-script", default=None, help="Path to SGE job script (qsub input)")
    ap.add_argument("--esv", default=None, help="Path to .esv file updated by the job")
    ap.add_argument(
        "--windows-root",
        default=None,
        help="Directory containing numbered window folders (0,1,2,...) for batch mode",
    )
    ap.add_argument(
        "--windows",
        nargs="+",
        default=None,
        help="Optional subset of window ids in batch mode (e.g. --windows 0 3 7)",
    )
    ap.add_argument(
        "--window-job-name",
        default="dynamic.job",
        help="Job script name expected inside each numbered window directory",
    )
    ap.add_argument(
        "--window-esv-name",
        default=None,
        help="Specific .esv filename inside each window directory; if omitted, auto-detect *.esv",
    )
    ap.add_argument(
        "--batch-stop-on-error",
        action="store_true",
        help="In batch mode, stop immediately when any window fails",
    )
    ap.add_argument("--esv-index", type=int, default=0, help="ESV block index to monitor")
    ap.add_argument("--qsub-cmd", default="qsub", help="qsub executable name/path")
    ap.add_argument("--qstat-cmd", default="qstat", help="qstat executable name/path")
    ap.add_argument("--poll-seconds", type=int, default=120, help="qstat polling interval")

    ap.add_argument("--max-chunks", type=int, default=20, help="Maximum 1 ns chunk submissions")
    ap.add_argument(
        "--required-stable-chunks",
        type=int,
        default=2,
        help="Consecutive stable chunk-to-chunk checks required",
    )
    ap.add_argument("--min-total-counts", type=float, default=200000.0)

    ap.add_argument("--max-d-frac-deprot", type=float, default=0.01)
    ap.add_argument("--max-d-frac-prot", type=float, default=0.01)
    ap.add_argument("--max-d-frac-lambda-intermediate", type=float, default=0.01)
    ap.add_argument("--max-d-lambda-bimodality", type=float, default=0.01)
    ap.add_argument("--max-d-lambda-asymmetry", type=float, default=0.02)
    ap.add_argument("--max-d-zeta-symmetry", type=float, default=0.02)
    ap.add_argument("--max-d-frac-interior", type=float, default=0.01)

    ap.add_argument(
        "--history",
        default="adaptive_convergence_history.json",
        help="JSON history file for metrics and decisions",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate current ESV once without submitting jobs",
    )

    args = ap.parse_args()

    if args.windows_root:
        windows_root = Path(args.windows_root)
        if not windows_root.is_dir():
            raise FileNotFoundError(f"windows-root not found: {windows_root}")

        window_dirs = discover_window_dirs(windows_root, windows_list=args.windows)
        if not window_dirs:
            raise RuntimeError(f"No numbered window directories found in {windows_root}")

        results = []
        for window_dir in window_dirs:
            window_id = window_dir.name
            window_label = f"window {window_id}"
            try:
                job_script = window_dir / args.window_job_name
                if not job_script.is_file():
                    raise FileNotFoundError(f"Job script not found: {job_script}")

                esv_path = find_esv_in_window(window_dir, esv_name=args.window_esv_name)
                history_path = window_dir / args.history

                converged, chunks_used = run_single_window(
                    args,
                    job_script=job_script,
                    esv_path=esv_path,
                    history_path=history_path,
                    window_label=window_label,
                    submit_cwd=window_dir,
                )
                results.append((window_id, converged, chunks_used, None))
            except Exception as exc:  # pragma: no cover
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[{window_label}] ERROR: {msg}")
                results.append((window_id, False, 0, msg))
                if args.batch_stop_on_error:
                    break

        print("\nBatch summary:")
        failed = 0
        for window_id, converged, chunks_used, err in results:
            if err:
                failed += 1
                print(f"  window {window_id}: ERROR - {err}")
            else:
                status = "CONVERGED" if converged else "NOT CONVERGED"
                if not converged:
                    failed += 1
                print(f"  window {window_id}: {status} (chunks used: {chunks_used})")

        if failed:
            sys.exit(2)
        return

    if not args.job_script or not args.esv:
        raise RuntimeError(
            "Single-window mode requires --job-script and --esv, or use --windows-root for batch mode"
        )

    job_script = Path(args.job_script)
    esv_path = Path(args.esv)
    history_path = Path(args.history)

    if not job_script.is_file():
        raise FileNotFoundError(f"Job script not found: {job_script}")
    if not esv_path.is_file():
        raise FileNotFoundError(f"ESV file not found: {esv_path}")

    converged, _ = run_single_window(
        args,
        job_script=job_script,
        esv_path=esv_path,
        history_path=history_path,
        window_label="single-window",
        submit_cwd=job_script.parent,
    )
    if not converged:
        sys.exit(2)


if __name__ == "__main__":
    main()
