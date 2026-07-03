import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from src.utils.config import load_config


# ==========================================================
# PROJECT ROOT DETECTION
# ==========================================================
def find_project_root():
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent

    raise RuntimeError("Could not find project root.")


# ==========================================================
# COMPUTE GLOBAL MIN/MAX
# ==========================================================
def compute_global_stats(files):
    print("Computing global normalization statistics...")

    vals = []

    sample_files = files[:200]

    for f in sample_files:
        try:
            ds = xr.open_dataset(f, engine="cfgrib")

            data = ds["ctoph"].values
            data = data[np.isfinite(data)]

            if len(data):
                vals.append(data)

        except Exception:
            pass

    vals = np.concatenate(vals)

    gmin = float(vals.min())
    gmax = float(vals.max())

    return gmin, gmax


# ==========================================================
# PROCESS SINGLE FILE
# ==========================================================
def process_one(args):
    file_path, cfg, grid_size, gmin, gmax, processed_dir = args

    try:
        ds = xr.open_dataset(file_path, engine="cfgrib")

        data = ds["ctoph"].values
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        region = cfg["region"]

        mask = (
            np.isfinite(data)
            & (lat >= region["lat_min"])
            & (lat <= region["lat_max"])
            & (lon >= region["lon_min"])
            & (lon <= region["lon_max"])
        )

        data = data[mask]
        lat = lat[mask]
        lon = lon[mask]

        if len(data) < 20:
            return False, f"{file_path.name}: too few points"

        # --------------------------------------------------
        # Build target grid
        # --------------------------------------------------
        grid_lon = np.linspace(
            region["lon_min"],
            region["lon_max"],
            grid_size
        )

        grid_lat = np.linspace(
            region["lat_min"],
            region["lat_max"],
            grid_size
        )

        grid_lon, grid_lat = np.meshgrid(
            grid_lon,
            grid_lat
        )

        # --------------------------------------------------
        # Interpolate scattered points
        # --------------------------------------------------
        grid = griddata(
            np.column_stack((lon, lat)),
            data,
            (grid_lon, grid_lat),
            method="linear",
            fill_value=np.nan
        )

        # Fill holes
        if np.isnan(grid).all():
            return False, f"{file_path.name}: all NaN"

        mean_val = np.nanmean(data)

        grid = np.nan_to_num(
            grid,
            nan=mean_val
        )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        grid = (
            (grid - gmin)
            / (gmax - gmin + 1e-6)
        ).astype(np.float32)

        out_file = processed_dir / f"{file_path.stem}.npy"

        np.save(out_file, grid)

        return True, file_path.name

    except Exception as e:
        return False, f"{file_path.name}: {e}"


# ==========================================================
# MAIN
# ==========================================================
def main():
    mp.set_start_method("spawn", force=True)

    root = find_project_root()
    cfg = load_config()

    extracted_dir = root / "data" / "extracted"
    processed_dir = root / "data" / "processed"

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        extracted_dir.glob("*.grb")
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No GRB files found in:\n{extracted_dir}"
        )

    print("\n===================================")
    print("PROJECT ROOT :", root)
    print("EXTRACTED DIR:", extracted_dir)
    print("PROCESSED DIR:", processed_dir)
    print("TOTAL FILES  :", len(files))
    print("===================================\n")

    # ------------------------------------------------------
    # Global normalization
    # ------------------------------------------------------
    gmin, gmax = compute_global_stats(files)

    print(f"Global min: {gmin}")
    print(f"Global max: {gmax}")

    grid_size = (
        cfg.get("grid", {})
        .get("size", 256)
    )

    workers = max(
        1,
        mp.cpu_count() - 1
    )

    print(f"\nUsing {workers} CPU cores")
    print(f"Grid size: {grid_size}x{grid_size}\n")

    tasks = [
        (
            f,
            cfg,
            grid_size,
            gmin,
            gmax,
            processed_dir
        )
        for f in files
    ]

    processed = 0
    failed = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                process_one,
                t
            )
            for t in tasks
        ]

        for future in as_completed(futures):
            ok, msg = future.result()

            if ok:
                processed += 1
            else:
                failed.append(msg)
                print("FAILED:", msg)

            total_done = processed + len(failed)

            if total_done % 50 == 0:
                print(
                    f"Progress: "
                    f"{total_done}/{len(files)} "
                    f"| Success: {processed} "
                    f"| Failed: {len(failed)}"
                )

    # ------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------
    meta = {
        "processed": processed,
        "failed": len(failed),
        "global_min": gmin,
        "global_max": gmax,
        "grid_size": grid_size,
        "failed_files": failed
    }

    meta_file = (
        processed_dir
        / "dataset_meta.json"
    )

    with open(meta_file, "w") as f:
        json.dump(
            meta,
            f,
            indent=2
        )

    print("\n===================================")
    print("PREPROCESS COMPLETE")
    print("Processed :", processed)
    print("Failed    :", len(failed))
    print("Metadata  :", meta_file)
    print("===================================")


if __name__ == "__main__":
    main()