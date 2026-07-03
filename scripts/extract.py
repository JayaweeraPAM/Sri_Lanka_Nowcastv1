import zipfile
from pathlib import Path
from tqdm import tqdm

# Folders — matching your new structure
INPUT_DIR = Path(r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\raw\cth_prototype_3months")
EXTRACTED_DIR = Path(r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\extracted")
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for .nc files in: {INPUT_DIR}")
print("🔄 Extracting all ZIP files (this may take a few minutes)...\n")

zip_files = list(INPUT_DIR.glob("*.nc"))  # All files ending with .nc are actually zips
print(f"Found {len(zip_files)} files.\n")

if len(zip_files) == 0:
    print("⚠️  No .nc files found — double check INPUT_DIR path above is correct.")

extracted_count = 0
failed_count = 0

for zip_path in tqdm(zip_files):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all contents
            zip_ref.extractall(EXTRACTED_DIR)

        # FIX: real extracted files are .grb, not .nc — checking for .nc here
        # meant this success message never printed even when extraction worked.
        expected_grb = EXTRACTED_DIR / f"{zip_path.stem}.grb"
        if expected_grb.exists():
            extracted_count += 1
        else:
            print(f"⚠️  Extracted {zip_path.name} but no matching .grb found")
            failed_count += 1

    except Exception as e:
        print(f"❌ Failed {zip_path.name}: {e}")
        failed_count += 1

print("\n" + "="*70)
print("🎉 EXTRACTION COMPLETE!")
print(f"Extracted: {extracted_count}  |  Failed: {failed_count}")
print(f"Real .grb files are in: {EXTRACTED_DIR.absolute()}")
print("="*70)
print("Now run the verification script again.")
