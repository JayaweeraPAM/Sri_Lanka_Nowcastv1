"""
Trace Dead Frames to Root Cause
=================================
The previous diagnostic showed dead frames rippling across ~8
overlapping sequences per bad frame (sliding-window fingerprint).
This script deduplicates that down to the actual distinct root
timestamps and checks whether they cluster by time-of-day or date
(consistent with genuine clear-sky/nighttime periods) or look
scattered/random (more consistent with a data corruption bug).

Requires that your .npz files store a timestamp/filename list so we
can map "timestep N of sequence M" back to a real datetime. Adjust
TIMESTAMP_KEY below to match whatever key your build script saved
(e.g. 'timestamps', 'filenames', 'times'). If no such key exists,
this script reconstructs an approximate global frame index instead
(still enough to show clustering vs. scattering).

Run:
    python trace_dead_frames.py
"""

from pathlib import Path
from collections import Counter, defaultdict
import re
import numpy as np

SEQ_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\time_sequences_fixed"
)
OUTPUT_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\final"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "dead_frame_root_cause_report.txt"

TIMESTAMP_KEY_CANDIDATES = ["timestamps", "timestamp", "times", "filenames", "files", "dates"]


def find_timestamp_key(keys):
    keys_lower = {k.lower(): k for k in keys}
    for c in TIMESTAMP_KEY_CANDIDATES:
        if c in keys_lower:
            return keys_lower[c]
    return None


def guess_input_key(keys):
    keys_lower = {k.lower(): k for k in keys}
    for c in ["input", "inputs", "x", "in_frames", "input_frames", "past", "context"]:
        if c in keys_lower:
            return keys_lower[c]
    return None


def extract_datetime_from_string(s):
    """Try to pull a YYYYMMDDHHMM-style timestamp out of a filename string."""
    m = re.search(r"(\d{8})[_\-]?(\d{4})", str(s))
    if m:
        return m.group(1) + m.group(2)  # YYYYMMDDHHMM
    m = re.search(r"(\d{12})", str(s))
    if m:
        return m.group(1)
    return None


def main():
    files = sorted(SEQ_DIR.glob("seq_*.npz"))
    sample = np.load(files[0])
    keys = list(sample.files)
    in_key = guess_input_key(keys) or keys[0]
    ts_key = find_timestamp_key(keys)
    sample.close()

    print(f"Using input key: '{in_key}'")
    print(f"Timestamp key  : '{ts_key}'" if ts_key else
          "No timestamp key found -- falling back to (sequence_index, timestep) identity.")

    # Map: root identifier -> list of (seq_file, timestep) that reference it
    root_to_occurrences = defaultdict(list)
    dead_root_values = {}

    for f in files:
        data = np.load(f)
        try:
            arr = data[in_key].astype(np.float32)
            if arr.ndim == 4:
                arr = arr[:, 0]

            ts_array = data[ts_key] if ts_key else None

            for t in range(arr.shape[0]):
                frame = arr[t]
                if np.nanstd(frame) < 1e-8:
                    if ts_array is not None:
                        raw_ts = ts_array[t]
                        root_id = extract_datetime_from_string(raw_ts) or str(raw_ts)
                    else:
                        # Fallback: approximate root by (file_index_in_segment - timestep)
                        # since consecutive sequences overlap by 1 and share frames.
                        seq_num = int(re.search(r"seq_(\d+)", f.name).group(1))
                        root_id = f"approx_{seq_num - t}"

                    root_to_occurrences[root_id].append((f.name, t))
                    dead_root_values[root_id] = float(np.nanmean(frame))
        finally:
            data.close()

    n_occurrences = sum(len(v) for v in root_to_occurrences.values())
    n_roots = len(root_to_occurrences)

    print("\n" + "=" * 70)
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)
    print(f"Total dead-frame occurrences (across all sequences): {n_occurrences}")
    print(f"Distinct root timestamps/frames responsible         : {n_roots}")
    print(f"Average sequences affected per root frame            : {n_occurrences / max(n_roots,1):.1f}")

    occurrence_counts = Counter(len(v) for v in root_to_occurrences.values())
    print("\nHow many sequences each root frame appears in (should cluster near 8):")
    for count, n in sorted(occurrence_counts.items()):
        print(f"  appears in {count} sequences: {n} root frames")

    # If we have real timestamps, check hour-of-day clustering
    hours = []
    for root_id in root_to_occurrences:
        if len(root_id) >= 12 and root_id[:8].isdigit():
            hh = root_id[8:10]
            if hh.isdigit():
                hours.append(int(hh))

    if hours:
        print("\n" + "=" * 70)
        print("HOUR-OF-DAY DISTRIBUTION OF DEAD ROOT FRAMES")
        print("=" * 70)
        hour_counter = Counter(hours)
        for h in range(24):
            bar = "#" * hour_counter.get(h, 0)
            print(f"  {h:02d}:00  {hour_counter.get(h,0):4d}  {bar}")
        print("\nIf dead frames cluster heavily in nighttime hours, this is very")
        print("likely genuine (cloud-top-height product may be low-confidence or")
        print("near-zero overnight depending on the sensor). If scattered evenly")
        print("across all hours, treat it as a data-quality issue to investigate.")
    else:
        print("\n[INFO] Could not resolve real timestamps for hour-of-day check.")
        print("       Root IDs used are approximate positions, not real datetimes.")
        print("       If you want the real hour-of-day breakdown, tell me which")
        print("       key in your .npz files stores per-frame filenames/timestamps")
        print("       (or confirm there isn't one, and I'll adjust the sequence")
        print("       builder to store one going forward).")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Total dead-frame occurrences: {n_occurrences}\n")
        f.write(f"Distinct root frames        : {n_roots}\n")
        f.write(f"Avg sequences per root frame : {n_occurrences / max(n_roots,1):.2f}\n\n")
        f.write("Root frame -> value, # occurrences\n")
        for root_id, val in sorted(dead_root_values.items()):
            f.write(f"  {root_id}: value={val:.4f}, occurrences={len(root_to_occurrences[root_id])}\n")

    print(f"\n[OK] Full root-cause report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()