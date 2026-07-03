"""
Raw Frame Dead-Check (before sequencing)
==========================================
Scans the ORIGINAL per-timestep .npy files (the ones your sequence
builder reads from) directly -- one row per real frame, no
reconstruction through overlapping sequences needed.

Reports:
  - How many raw frames are dead/near-constant
  - Their actual filenames (so you can inspect a few by hand)
  - Hour-of-day distribution, IF your filenames encode a timestamp
    (edit TIMESTAMP_REGEX below if the auto-detect doesn't match)

Run:
    python check_raw_frames.py
"""

import re
from pathlib import Path
from collections import Counter
import numpy as np

# ==========================================================
# CONFIG -- point this at your RAW per-frame .npy folder,
# NOT the time_sequences_fixed folder.
# ==========================================================
RAW_INPUT_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\processed"
)

OUTPUT_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\final"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "raw_frame_dead_check_report.txt"

# Common EUMETSAT-style patterns: YYYYMMDDHHMM or YYYYMMDD_HHMM etc.
# If none of these match your filenames, paste one example filename
# and I'll give you the exact regex.
TIMESTAMP_PATTERNS = [
    r"(\d{8})[_\-]?(\d{4})",   # YYYYMMDD_HHMM or YYYYMMDDHHMM
    r"(\d{4})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})",  # YYYY-MM-DD-HH-MM
]


def parse_timestamp(filename: str):
    for pat in TIMESTAMP_PATTERNS:
        m = re.search(pat, filename)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                date_part, time_part = groups
                return date_part, time_part[:2], time_part[2:4]  # date, HH, MM
            elif len(groups) == 5:
                yyyy, mm, dd, hh, mi = groups
                return f"{yyyy}{mm}{dd}", hh, mi
    return None, None, None


def main():
    if not RAW_INPUT_DIR.exists():
        print(f"[ERROR] Folder does not exist: {RAW_INPUT_DIR}")
        print("        Edit RAW_INPUT_DIR at the top of this script to point at your")
        print("        folder of raw per-frame .npy files, then re-run.")
        return

    files = sorted(RAW_INPUT_DIR.glob("*.npy"))
    if not files:
        print(f"[ERROR] No .npy files found in {RAW_INPUT_DIR}")
        return

    print(f"Found {len(files)} raw frame files.")
    print(f"Example filename: {files[0].name}")

    date_sample, hh_sample, mm_sample = parse_timestamp(files[0].name)
    if date_sample is None:
        print("\n[WARN] Could not auto-parse a timestamp from the filename.")
        print("       Paste an example filename to me and I'll write the exact regex.")
        print("       Continuing with dead-frame detection only (no time-of-day check).")
    else:
        print(f"Parsed example -> date={date_sample} hour={hh_sample} min={mm_sample}")

    dead_files = []
    dead_hours = []
    dead_dates = []
    all_stds = []

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print(f"  [{i}/{len(files)}] scanning...")
        arr = np.load(f).astype(np.float32)
        std = float(np.nanstd(arr))
        all_stds.append(std)

        if std < 1e-8:
            dead_files.append(f.name)
            date_part, hh, mm = parse_timestamp(f.name)
            if hh is not None:
                dead_hours.append(int(hh))
            if date_part is not None:
                dead_dates.append(date_part)

    all_stds = np.array(all_stds)

    print("\n" + "=" * 70)
    print("RAW FRAME SUMMARY")
    print("=" * 70)
    print(f"Total raw frames        : {len(files)}")
    print(f"Dead/near-constant      : {len(dead_files)}  ({100*len(dead_files)/len(files):.1f}%)")
    print(f"Std dev range           : [{all_stds.min():.6f}, {all_stds.max():.6f}]")
    print(f"Std dev mean            : {all_stds.mean():.6f}")

    if dead_hours:
        print("\n" + "=" * 70)
        print("HOUR-OF-DAY DISTRIBUTION OF DEAD RAW FRAMES")
        print("=" * 70)
        hour_counter = Counter(dead_hours)
        for h in range(24):
            n = hour_counter.get(h, 0)
            print(f"  {h:02d}:00  {n:4d}  {'#' * n}")

        # Timestamps are UTC (note the 'Z' in EUMETSAT filenames).
        # Sri Lanka is UTC+5:30, so convert before calling anything "night".
        # Roughly: Sri Lanka daylight (~06:00-18:00 local) = ~00:30-12:30 UTC.
        night_hours_utc = set(range(13, 24)) | {0}  # ~18:30-06:30 local night
        night_count = sum(hour_counter.get(h, 0) for h in night_hours_utc)
        print(f"\nDead frames in ~night hours (Sri Lanka local, i.e. 13:00-24:00 UTC): "
              f"{night_count} / {len(dead_hours)} ({100*night_count/len(dead_hours):.1f}%)")
        print("If this is high, dead frames are likely genuine nighttime/low-signal")
        print("readings, not a bug. If it's close to random (~40% given the hour")
        print("range chosen), investigate further.")

    if dead_dates:
        print("\n" + "=" * 70)
        print("DATES WITH THE MOST DEAD FRAMES (top 15)")
        print("=" * 70)
        date_counter = Counter(dead_dates)
        for date, n in date_counter.most_common(15):
            print(f"  {date}: {n} dead frames")
        print("\nIf dead frames are concentrated on a handful of specific dates,")
        print("that points to a sensor outage or download/preprocessing gap on")
        print("those dates, not routine clear-sky conditions.")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Total raw frames: {len(files)}\n")
        f.write(f"Dead frames: {len(dead_files)}\n\n")
        f.write("Dead frame filenames:\n")
        for name in dead_files:
            f.write(f"  {name}\n")

    print(f"\n[OK] Full report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()