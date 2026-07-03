"""
Verify a single .grb cloud-top-height file: prints variables/attrs, crops
to the Sri Lanka bbox, saves full-scene + cropped scatter previews.

Run (auto-picks the first .grb found in data/extracted):
    python scripts/verify_grb_file.py

Or point at a specific file:
    python scripts/verify_grb_file.py "C:\\path\\to\\file.grb"
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config


def verify_grib_file(file_path: Path, cfg: dict):
    region = cfg["region"]

    print("=" * 85)
    print(f"VERIFYING: {file_path.name}")
    print("=" * 85)

    try:
        ds = xr.open_dataset(file_path, engine="cfgrib")
        print("✅ File opened successfully!\n")

        print("📋 Dataset Summary:")
        print(ds)

        print("\n📏 Dimensions:")
        for dim, size in ds.sizes.items():
            print(f"   {dim:15} : {size}")

        var_list = list(ds.data_vars)
        print("\n📊 Variables found in this file:")
        for var in var_list:
            print(f"   {var:20} shape={ds[var].shape}  dtype={ds[var].dtype}")

        if not var_list:
            print("   (no data variables found — check ds.attrs / coords below)")
            print("\n🧭 Coordinates:")
            print(ds.coords)
            return

        # Confirmed: cloud top height is 'ctoph', quality index is 'ctophqi'.
        # Data is a flattened 1D array (SEVIRI 'space_view' scan geometry) —
        # lat/lon are per-point auxiliary coords, not a grid, so imshow() won't work.
        main_var = "ctoph" if "ctoph" in var_list else var_list[0]
        qi_var = "ctophqi" if "ctophqi" in var_list else None

        data = ds[main_var].squeeze()
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        print(f"\n🎯 Main Variable: '{main_var}'")
        print(f"   Shape          : {data.shape}  (flattened space_view points)")
        print(f"   Min Height     : {float(data.min()):.1f} m")
        print(f"   Max Height     : {float(data.max()):.1f} m")
        print(f"   Mean Height    : {float(data.mean()):.1f} m")
        print(f"   Units (attr)   : {data.attrs.get('units', 'not specified')}")
        print(f"   Long name      : {data.attrs.get('long_name', 'not specified')}")
        print(f"   Lat range      : {np.nanmin(lat):.2f} to {np.nanmax(lat):.2f}")
        print(f"   Lon range      : {np.nanmin(lon):.2f} to {np.nanmax(lon):.2f}")

        # --- Crop to Sri Lanka bbox, using the SAME bbox defined in configs/config.yaml
        #     so this preview always matches what preprocess.py will actually use ---
        LAT_MIN, LAT_MAX = region["lat_min"], region["lat_max"]
        LON_MIN, LON_MAX = region["lon_min"], region["lon_max"]

        mask = (
            (lat >= LAT_MIN) & (lat <= LAT_MAX) &
            (lon >= LON_MIN) & (lon <= LON_MAX) &
            np.isfinite(data.values)
        )
        n_points = mask.sum()
        print(f"\n🇱🇰 Sri Lanka bbox crop (from config.yaml): {n_points} points found "
              f"(lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX})")

        if n_points == 0:
            print("   ⚠️  No points fall inside this bbox — check config.yaml region values.")
        else:
            crop_vals = data.values[mask]
            print(f"   Cropped min/mean/max height: "
                  f"{crop_vals.min():.1f} / {crop_vals.mean():.1f} / {crop_vals.max():.1f} m")

        # --- Full-scene scatter preview ---
        out_dir = cfg["paths"]["outputs_dir"] / "verify_previews"
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        valid = np.isfinite(data.values)
        sc = plt.scatter(lon[valid], lat[valid], c=data.values[valid],
                          cmap="jet", s=1, marker=".")
        plt.colorbar(sc, label=f"{main_var} ({data.attrs.get('units', 'm')})")
        plt.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
                  [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
                  color="white", linewidth=1.5, label="Sri Lanka crop bbox")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="upper right")
        plt.title(f"Cloud Top Height (full scene) - {file_path.name[:50]}...")
        plt.tight_layout()
        full_preview_path = out_dir / f"{file_path.stem}_preview_full.png"
        plt.savefig(full_preview_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"\n📸 Full-scene preview saved → {full_preview_path}")

        # --- Cropped-only preview, zoomed to Sri Lanka ---
        if n_points > 0:
            plt.figure(figsize=(8, 8))
            sc2 = plt.scatter(lon[mask], lat[mask], c=data.values[mask],
                               cmap="jet", s=8, marker="s")
            plt.colorbar(sc2, label=f"{main_var} (m)")
            plt.xlim(LON_MIN, LON_MAX)
            plt.ylim(LAT_MIN, LAT_MAX)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Cloud Top Height (Sri Lanka crop) - {file_path.name[:50]}...")
            plt.tight_layout()
            crop_preview_path = out_dir / f"{file_path.stem}_preview_srilanka_crop.png"
            plt.savefig(crop_preview_path, dpi=250, bbox_inches="tight")
            plt.close()
            print(f"📸 Sri Lanka crop preview saved → {crop_preview_path}")

        if qi_var:
            print(f"\n✔️  Quality index '{qi_var}' also present — use this to mask "
                  f"low-confidence pixels before training (confirm its value range/meaning "
                  f"against product docs; qi_threshold in config.yaml is currently null).")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 If this is an ecCodes/cfgrib install error, try:")
        print("   conda install -c conda-forge cfgrib")


def find_default_file(cfg: dict) -> Path:
    extracted_dir = cfg["paths"]["extracted_dir"]
    grb_files = sorted(extracted_dir.glob("*.grb"))
    if not grb_files:
        raise SystemExit(
            f"No .grb files found in {extracted_dir}.\n"
            f"Run extraction first (python -m src.data.extract), or pass a file path directly:\n"
            f"  python scripts/verify_grb_file.py \"C:\\path\\to\\file.grb\""
        )
    return grb_files[0]


if __name__ == "__main__":
    cfg = load_config()

    if len(sys.argv) >= 2:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")
    else:
        file_path = find_default_file(cfg)
        print(f"No file path given — auto-picked: {file_path}\n")

    verify_grib_file(file_path, cfg)