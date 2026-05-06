import os
import re
from pathlib import Path

# ---------------- USER SETTINGS ----------------
BASE_DIR = "."          # directory containing folders 0–11
RESIDUE = "73-LYS"      # residue bin to extract
COUNTS_TO_NS = 1e6
# -----------------------------------------------

def extract_esv_time(esv_path, residue):
    """
    Sum all counts in the ESV histogram for a specific residue.
    """
    with open(esv_path, "r") as f:
        lines = f.readlines()

    inside_block = False
    total_counts = 0

    for line in lines:
        # Detect start of desired ESV block
        if line.startswith(" ESV:") and residue in line:
            inside_block = True
            continue

        # Detect start of a new ESV block → stop
        if inside_block and line.startswith(" ESV:"):
            break

        if inside_block:
            # Match rows like: [0.5-0.6]   25  0  0 ...
            numbers = re.findall(r"\d+", line)
            if numbers:
                total_counts += sum(map(int, numbers))

    return total_counts / COUNTS_TO_NS


def main():
    print(f"Residue: {RESIDUE}")
    print(f"{'Dir':>3}  {'pH':>4}  {'Time (ns)':>10}")
    print("-" * 24)

    for d in range(12):
        ph = 1 + 0.5 * d
        dir_path = Path(BASE_DIR) / str(d)

        esv_files = list(dir_path.glob("*.esv"))
        if not esv_files:
            print(f"{d:>3}  {ph:>4.1f}  MISSING")
            continue

        time_ns = extract_esv_time(esv_files[0], RESIDUE)
        print(f"{d:>3}  {ph:>4.1f}  {time_ns:10.3f}")


if __name__ == "__main__":
    main()

