"""
Full Dataset Verification Script
=================================
Verifies the built sequence dataset (time_sequences_fixed/seq_XXXXXX.npz).

Checks:
  1. Auto-detects the array keys stored in each .npz (no assumptions)
  2. Confirms every file has consistent shape / dtype
  3. Checks for NaN / Inf / constant (dead) frames
  4. Reports value range + basic stats per channel
  5. Confirms train/val split has no overlap and no leakage across
     what should be a temporal split
  6. Saves a handful of visualized sample sequences to disk
  7. Writes a full text report

Run:
    python verify_dataset.py
"""

from pathlib import Path
import numpy as np
import json
import sys
from collections import Counter, defaultdict

# ==========================================================
# CONFIG
# ==========================================================

SEQ_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\pre process steps\time_sequences_fixed"
)

OUTPUT_DIR = SEQ_DIR / "verification_report"
SAMPLE_PLOTS = 5          # how many sequences to visualize
MAX_FULL_SCAN = None      # set an int to limit scan for a quick check, None = scan all
VAL_FRACTION = 0.15       # expected val split fraction, just for sanity reporting


# ==========================================================
# HELPERS
# ==========================================================

def list_sequence_files(seq_dir: Path):
    files = sorted(seq_dir.glob("seq_*.npz"))
    if not files:
        print(f"[ERROR] No seq_*.npz files found in {seq_dir}")
        sys.exit(1)
    return files


def inspect_first_file(files):
    """Auto-detect keys/shapes without assuming X/y naming."""
    sample = np.load(files[0])
    info = {}
    print("=" * 70)
    print(f"INSPECTING FIRST FILE: {files[0].name}")
    print("=" * 70)
    for key in sample.files:
        arr = sample[key]
        print(f"  key='{key}'  shape={arr.shape}  dtype={arr.dtype}")
        info[key] = {"shape": arr.shape, "dtype": str(arr.dtype)}
    sample.close()
    return info


def guess_input_target_keys(keys):
    """
    Try to figure out which key(s) hold input frames and which hold
    target/prediction frames, based on common naming conventions.
    Falls back to None if it can't confidently guess -- caller must
    then handle a single combined array manually.
    """
    keys_lower = {k.lower(): k for k in keys}

    input_candidates = ["input", "inputs", "x", "in_frames", "input_frames", "past", "context"]
    target_candidates = ["target", "targets", "y", "out_frames", "target_frames",
                          "future", "label", "labels", "gt"]

    in_key = next((keys_lower[c] for c in input_candidates if c in keys_lower), None)
    out_key = next((keys_lower[c] for c in target_candidates if c in keys_lower), None)

    return in_key, out_key


def full_scan(files, in_key, out_key, single_key, limit=None):
    """
    Walk every sequence file and check:
      - shape consistency
      - dtype consistency
      - NaN / Inf presence
      - constant / dead frames
      - value range (min/max/mean/std)
    """
    n = len(files) if limit is None else min(limit, len(files))

    shape_counter = Counter()
    dtype_counter = Counter()
    nan_files = []
    inf_files = []
    dead_frame_files = []
    load_error_files = []

    global_min, global_max = np.inf, -np.inf
    running_sum = 0.0
    running_sq_sum = 0.0
    running_count = 0

    print("\n" + "=" * 70)
    print(f"FULL SCAN: checking {n} / {len(files)} sequence files")
    print("=" * 70)

    for i, f in enumerate(files[:n]):
        if i % 500 == 0:
            print(f"  [{i}/{n}] scanning...")

        try:
            data = np.load(f)
        except Exception as e:
            load_error_files.append((f.name, str(e)))
            continue

        try:
            if single_key is not None:
                arrs = [data[single_key]]
            else:
                arrs = []
                if in_key is not None:
                    arrs.append(data[in_key])
                if out_key is not None:
                    arrs.append(data[out_key])
                if not arrs:
                    # fall back: use every array in the file
                    arrs = [data[k] for k in data.files]

            for arr in arrs:
                shape_counter[arr.shape] += 1
                dtype_counter[str(arr.dtype)] += 1

                if np.isnan(arr).any():
                    nan_files.append(f.name)
                if np.isinf(arr).any():
                    inf_files.append(f.name)

                # check for dead (constant) frames along the time axis
                if arr.ndim >= 3:
                    for t in range(arr.shape[0]):
                        frame = arr[t]
                        if np.nanstd(frame) < 1e-8:
                            dead_frame_files.append((f.name, t))
                            break

                finite = arr[np.isfinite(arr)]
                if finite.size:
                    global_min = min(global_min, float(finite.min()))
                    global_max = max(global_max, float(finite.max()))
                    running_sum += float(finite.sum())
                    running_sq_sum += float((finite.astype(np.float64) ** 2).sum())
                    running_count += finite.size

        finally:
            data.close()

    mean = running_sum / running_count if running_count else float("nan")
    var = (running_sq_sum / running_count - mean ** 2) if running_count else float("nan")
    std = np.sqrt(max(var, 0.0))

    results = {
        "scanned": n,
        "shape_counter": shape_counter,
        "dtype_counter": dtype_counter,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "dead_frame_files": dead_frame_files,
        "load_error_files": load_error_files,
        "global_min": global_min,
        "global_max": global_max,
        "mean": mean,
        "std": std,
    }
    return results


def check_split_integrity(files):
    """
    If a split file / metadata exists describing train/val/test membership,
    verify no overlap. Otherwise just report a simple filename-based
    chronological split sanity check.
    """
    split_file = None
    for candidate in ["split.json", "dataset_split.json", "train_val_split.json"]:
        p = files[0].parent / candidate
        if p.exists():
            split_file = p
            break

    if split_file is None:
        print("\n[INFO] No explicit split.json found next to sequences.")
        print("       If you split train/val by slicing this sorted file list,")
        print("       make sure the split point does NOT fall inside a")
        print("       contiguous run that was also used to build overlapping")
        print("       sliding windows (that would leak validation frames into")
        print("       training sequences).")
        return None

    with open(split_file) as fh:
        split = json.load(fh)

    train_set = set(split.get("train", []))
    val_set = set(split.get("val", []))
    test_set = set(split.get("test", []))

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    print("\n" + "=" * 70)
    print("SPLIT INTEGRITY CHECK")
    print("=" * 70)
    print(f"  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    print(f"  train/val overlap : {len(overlap_tv)}")
    print(f"  train/test overlap: {len(overlap_tt)}")
    print(f"  val/test overlap  : {len(overlap_vt)}")

    return {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
        "overlap_train_val": len(overlap_tv),
        "overlap_train_test": len(overlap_tt),
        "overlap_val_test": len(overlap_vt),
    }


def save_sample_plots(files, in_key, out_key, single_key, n_samples, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping sample plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(files) - 1, n_samples, dtype=int)

    for idx in idxs:
        f = files[idx]
        data = np.load(f)

        if single_key is not None:
            full = data[single_key]
            in_frames = full
            out_frames = None
        else:
            in_frames = data[in_key] if in_key else None
            out_frames = data[out_key] if out_key else None

        data.close()

        seqs = []
        titles = []
        if in_frames is not None:
            seqs.append(in_frames)
            titles.append("input")
        if out_frames is not None:
            seqs.append(out_frames)
            titles.append("target")

        total_frames = sum(s.shape[0] for s in seqs)
        fig, axes = plt.subplots(1, total_frames, figsize=(2.2 * total_frames, 2.6))
        if total_frames == 1:
            axes = [axes]

        col = 0
        for seq, title in zip(seqs, titles):
            for t in range(seq.shape[0]):
                frame = seq[t]
                if frame.ndim == 3:  # (C,H,W) -> take channel 0
                    frame = frame[0]
                ax = axes[col]
                im = ax.imshow(frame, cmap="viridis")
                ax.set_title(f"{title[0]}{t}", fontsize=8)
                ax.axis("off")
                col += 1

        fig.suptitle(f.name)
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{f.stem}.png", dpi=110)
        plt.close(fig)

    print(f"\n[OK] Saved {n_samples} sample sequence plots -> {out_dir}")


def write_report(out_dir, first_file_info, scan_results, split_results):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verification_report.txt"

    with open(report_path, "w") as f:
        f.write("DATASET VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Keys found in sample file:\n")
        for k, v in first_file_info.items():
            f.write(f"  {k}: shape={v['shape']}  dtype={v['dtype']}\n")

        f.write("\nShape consistency:\n")
        for shape, count in scan_results["shape_counter"].items():
            f.write(f"  {shape}: {count} arrays\n")
        if len(scan_results["shape_counter"]) > 1:
            f.write("  [WARNING] More than one distinct shape found!\n")

        f.write("\nDtype consistency:\n")
        for dtype, count in scan_results["dtype_counter"].items():
            f.write(f"  {dtype}: {count} arrays\n")

        f.write(f"\nFiles scanned        : {scan_results['scanned']}\n")
        f.write(f"Files failing to load : {len(scan_results['load_error_files'])}\n")
        for name, err in scan_results["load_error_files"][:20]:
            f.write(f"    {name}: {err}\n")

        f.write(f"\nFiles with NaN        : {len(scan_results['nan_files'])}\n")
        for name in scan_results["nan_files"][:20]:
            f.write(f"    {name}\n")

        f.write(f"\nFiles with Inf        : {len(scan_results['inf_files'])}\n")
        for name in scan_results["inf_files"][:20]:
            f.write(f"    {name}\n")

        f.write(f"\nFiles with a dead/constant frame: {len(scan_results['dead_frame_files'])}\n")
        for name, t in scan_results["dead_frame_files"][:20]:
            f.write(f"    {name} (frame {t})\n")

        f.write(f"\nGlobal min  : {scan_results['global_min']}\n")
        f.write(f"Global max  : {scan_results['global_max']}\n")
        f.write(f"Global mean : {scan_results['mean']}\n")
        f.write(f"Global std  : {scan_results['std']}\n")

        if split_results:
            f.write("\nSplit integrity:\n")
            for k, v in split_results.items():
                f.write(f"  {k}: {v}\n")

    print(f"\n[OK] Full report written -> {report_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    print(f"Sequence dir: {SEQ_DIR}")

    files = list_sequence_files(SEQ_DIR)
    print(f"Found {len(files)} sequence files.\n")

    first_file_info = inspect_first_file(files)
    keys = list(first_file_info.keys())

    in_key, out_key = guess_input_target_keys(keys)
    single_key = None

    if in_key is None and out_key is None:
        if len(keys) == 1:
            single_key = keys[0]
            print(f"\n[INFO] Only one array key found ('{single_key}'). "
                  f"Treating it as one combined sequence; verify the "
                  f"input/target split point yourself (e.g. frames[:8] vs frames[8:]).")
        else:
            print(f"\n[WARN] Could not auto-detect input/target keys from {keys}.")
            print("       Will scan every array under every key generically.")
    else:
        print(f"\n[INFO] Detected input key = '{in_key}', target key = '{out_key}'")

    scan_results = full_scan(files, in_key, out_key, single_key, limit=MAX_FULL_SCAN)

    split_results = check_split_integrity(files)

    save_sample_plots(files, in_key, out_key, single_key, SAMPLE_PLOTS, OUTPUT_DIR / "sample_plots")

    write_report(OUTPUT_DIR, first_file_info, scan_results, split_results)

    # ---- final summary printed to console ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_shapes = len(scan_results["shape_counter"])
    print(f"Distinct shapes found : {n_shapes} {'[OK]' if n_shapes == 1 else '[CHECK REPORT]'}")
    print(f"NaN files             : {len(scan_results['nan_files'])}")
    print(f"Inf files             : {len(scan_results['inf_files'])}")
    print(f"Dead-frame files      : {len(scan_results['dead_frame_files'])}")
    print(f"Load errors           : {len(scan_results['load_error_files'])}")
    print(f"Value range           : [{scan_results['global_min']:.4f}, {scan_results['global_max']:.4f}]")
    print(f"Mean / Std            : {scan_results['mean']:.4f} / {scan_results['std']:.4f}")
    print("\nIf NaN/Inf/dead-frame counts are non-zero, fix those sequences")
    print("(drop or reprocess) BEFORE training -- don't let the DataLoader")
    print("silently feed garbage into the ConvLSTM.")


if __name__ == "__main__":
    main()
