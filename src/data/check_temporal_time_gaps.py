from pathlib import Path
from datetime import datetime
import shutil

# ==========================================================
# PATHS
# ==========================================================
SOURCE_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\processed"
)

DEST_DIR = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\checked_temporal_time_gaps_after_preprocess"
)

EXPECTED_MINUTES = 15

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================
DEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# GET FILES
# ==========================================================
files = sorted(
    SOURCE_DIR.glob("*.npy")
)

print(f"Files found: {len(files)}")

if len(files) == 0:
    raise SystemExit("No .npy files found.")

# ==========================================================
# PARSE TIMES
# ==========================================================
records = []

for f in files:
    try:
        ts = f.name.split("-")[-2].split(".")[0]

        dt = datetime.strptime(
            ts,
            "%Y%m%d%H%M%S"
        )

        records.append((dt, f))

    except Exception:
        print(f"Cannot parse timestamp: {f.name}")

records.sort(key=lambda x: x[0])

# ==========================================================
# KEEP ONLY CONTINUOUS SEGMENTS
# ==========================================================
kept = []
removed = []

if records:
    kept.append(records[0][1])

for i in range(len(records) - 1):
    t1, f1 = records[i]
    t2, f2 = records[i + 1]

    delta = (
        t2 - t1
    ).total_seconds() / 60

    if delta == EXPECTED_MINUTES:
        kept.append(f2)
    else:
        removed.append((delta, f1.name, f2.name))

        print(
            f"\nGAP: {delta:.0f} min"
        )
        print(f"  {f1.name}")
        print(f"  {f2.name}")

# ==========================================================
# COPY FILES
# ==========================================================
print("\nCopying continuous files...")

for f in kept:
    dst = DEST_DIR / f.name

    if not dst.exists():
        shutil.copy2(
            f,
            dst
        )

# ==========================================================
# SUMMARY
# ==========================================================
print("\n====================================")
print(f"Original files : {len(files)}")
print(f"Continuous files copied : {len(kept)}")
print(f"Temporal gaps : {len(removed)}")
print(f"Saved to :")
print(DEST_DIR)
print("====================================")

# ==========================================================
# SAVE GAP REPORT
# ==========================================================
report = DEST_DIR / "temporal_gap_report.txt"

with open(report, "w") as fp:
    fp.write(
        f"Original files: {len(files)}\n"
    )
    fp.write(
        f"Continuous files: {len(kept)}\n"
    )
    fp.write(
        f"Temporal gaps: {len(removed)}\n\n"
    )

    for delta, a, b in removed:
        fp.write(
            f"GAP {delta:.0f} min\n"
        )
        fp.write(f"{a}\n")
        fp.write(f"{b}\n\n")

print(f"Gap report saved:")
print(report)