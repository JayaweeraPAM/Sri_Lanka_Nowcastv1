from pathlib import Path
from datetime import datetime
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ==========================================================
# PATHS
# ==========================================================

INPUT_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\checked_temporal_time_gaps_after_preprocess"
)

OUTPUT_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\time_sequences_fixed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# SETTINGS
# ==========================================================

INPUT_STEPS = 8
OUTPUT_STEPS = 4
EXPECTED_MINUTES = 15
NUM_WORKERS = 7  # leave 1 core free for the OS / other work

# ==========================================================
# PARSE TIMESTAMP
# ==========================================================

def get_timestamp(file):
    ts = file.name.split("-")[-2].split(".")[0]
    return datetime.strptime(ts, "%Y%m%d%H%M%S")


# ==========================================================
# WORKER: build + save ONE sequence
# (must be a top-level function — ProcessPoolExecutor needs to pickle it)
# ==========================================================

def build_one_sequence(args):
    seq_index, x_paths, y_paths, output_dir = args

    try:
        x = np.stack([np.load(p).astype(np.float32) for p in x_paths])
        y = np.stack([np.load(p).astype(np.float32) for p in y_paths])

        out_file = output_dir / f"seq_{seq_index:06d}.npz"
        np.savez_compressed(out_file, x=x, y=y)

        return True, seq_index, None
    except Exception as e:
        return False, seq_index, str(e)


def main():
    # ==========================================================
    # LOAD FILES
    # ==========================================================

    files = sorted(INPUT_DIR.glob("*.npy"))

    print("=" * 70)
    print("TIME SEQUENCE BUILDER (parallel, 7 cores)")
    print("=" * 70)
    print(f"Input Folder : {INPUT_DIR}")
    print(f"Output Folder: {OUTPUT_DIR}")
    print(f"Files Found  : {len(files):,}")
    print("=" * 70)

    if len(files) == 0:
        raise SystemExit("No .npy files found.")

    records = [(get_timestamp(f), f) for f in files]
    records.sort(key=lambda x: x[0])

    # ==========================================================
    # SPLIT INTO CONTINUOUS SEGMENTS
    # ==========================================================

    segments = []
    current = [records[0]]

    for i in range(1, len(records)):
        prev_time = records[i - 1][0]
        curr_time = records[i][0]
        delta = (curr_time - prev_time).total_seconds() / 60

        if delta == EXPECTED_MINUTES:
            current.append(records[i])
        else:
            segments.append(current)
            current = [records[i]]

    segments.append(current)
    print(f"Continuous Segments : {len(segments)}")

    # ==========================================================
    # BUILD FULL TASK LIST UP FRONT
    # (each task is independent — this is what makes parallelizing safe:
    #  no shared state between workers, no cross-segment windows)
    # ==========================================================

    tasks = []
    seq_index = 0

    for seg_index, seg in enumerate(segments, start=1):
        if len(seg) < INPUT_STEPS + OUTPUT_STEPS:
            print(f"Skipping Segment {seg_index}/{len(segments)} (only {len(seg)} frames)")
            continue

        max_windows = len(seg) - INPUT_STEPS - OUTPUT_STEPS + 1
        for window in range(max_windows):
            x_paths = [f[1] for f in seg[window:window + INPUT_STEPS]]
            y_paths = [f[1] for f in seg[window + INPUT_STEPS: window + INPUT_STEPS + OUTPUT_STEPS]]
            tasks.append((seq_index, x_paths, y_paths, OUTPUT_DIR))
            seq_index += 1

    total_sequences = len(tasks)
    print(f"Expected Sequences  : {total_sequences:,}")
    print(f"Using {NUM_WORKERS} CPU workers\n")

    if total_sequences == 0:
        raise SystemExit("No sequences to build — check segment lengths vs INPUT_STEPS+OUTPUT_STEPS.")

    # ==========================================================
    # BUILD SEQUENCES IN PARALLEL
    # ==========================================================

    completed = 0
    failed = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(build_one_sequence, t) for t in tasks]

        for future in as_completed(futures):
            ok, idx, err = future.result()
            completed += 1

            if not ok:
                failed.append((idx, err))
                print(f"FAILED seq_{idx:06d}: {err}")

            if completed % 100 == 0 or completed == total_sequences:
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                percent = (completed / total_sequences) * 100
                remaining = total_sequences - completed
                eta = remaining / speed if speed > 0 else 0
                eta_min, eta_sec = int(eta // 60), int(eta % 60)

                print(
                    f"[{percent:6.2f}%] {completed:,}/{total_sequences:,} "
                    f"| {speed:6.1f} seq/s | ETA {eta_min:02d}:{eta_sec:02d}"
                )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Input Files          : {len(files):,}")
    print(f"Segments             : {len(segments)}")
    print(f"Sequences Created    : {completed - len(failed):,}")
    print(f"Sequences Failed     : {len(failed):,}")
    print()
    print(f"Input Frames         : {INPUT_STEPS}")
    print(f"Prediction Frames    : {OUTPUT_STEPS}")
    print(f"Frame Interval       : {EXPECTED_MINUTES} minutes")
    print()
    print(f"Processing Time      : {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Output Folder        : {OUTPUT_DIR}")
    print("=" * 70)

    # ==========================================================
    # SAVE METADATA
    # ==========================================================

    meta = OUTPUT_DIR / "dataset_info.txt"
    with open(meta, "w") as f:
        f.write("Sri Lanka Nowcast Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input Files          : {len(files)}\n")
        f.write(f"Segments             : {len(segments)}\n")
        f.write(f"Sequences            : {completed - len(failed)}\n")
        f.write(f"Sequences Failed     : {len(failed)}\n\n")
        f.write(f"Input Frames         : {INPUT_STEPS}\n")
        f.write(f"Prediction Frames    : {OUTPUT_STEPS}\n")
        f.write(f"Frame Interval       : {EXPECTED_MINUTES} minutes\n\n")
        f.write(f"Processing Time      : {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        if failed:
            f.write(f"\nFailed sequences:\n")
            for idx, err in failed:
                f.write(f"  seq_{idx:06d}: {err}\n")

    print(f"Metadata saved -> {meta}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required on Windows
    main()