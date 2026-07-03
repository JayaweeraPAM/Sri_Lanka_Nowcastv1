import eumdac
import datetime
import shutil
import time
import os
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

CONSUMER_KEY = "N3v4fbq5HzQgVGZNo1pEfeXU_n8a"
CONSUMER_SECRET = "fV8Q61aE83rO2qZinVi8kP_pf18a"
COLLECTION_ID = "EO:EUM:DAT:MSG:CTH-IODC"
OUTPUT_DIR = "./cth_prototype_3months"

os.makedirs(OUTPUT_DIR, exist_ok=True)

token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
datastore = eumdac.DataStore(token)
collection = datastore.get_collection(COLLECTION_ID)


def daterange_chunks(start, end, days=4):   # Smaller chunks = more reliable
    cur = start
    while cur < end:
        nxt = min(cur + datetime.timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def download_with_progress(product, out_path):
    try:
        with product.open() as fsrc:
            total_size = getattr(fsrc, 'content_length', None)
            with open(out_path, "wb") as fdst:
                if total_size:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                              unit_divisor=1024, desc=str(product)[-30:], 
                              leave=False, mininterval=0.5) as pbar:
                        wrapper = CallbackIOWrapper(pbar.update, fdst)
                        shutil.copyfileobj(fsrc, wrapper)
                else:
                    shutil.copyfileobj(fsrc, fdst)
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        return False


# ====================== 3 MONTHS PROTOTYPE ======================

end_date = datetime.datetime.utcnow()                    # Today (July 2026)
start_date = end_date - datetime.timedelta(days=93)     # ~3 months back

print("🚀 Starting 3-Month Prototype Download")
print(f"Period : {start_date.strftime('%Y-%m-%d')}  →  {end_date.strftime('%Y-%m-%d')}")
print(f"Target : ~8,600 files\n")

all_chunks = list(daterange_chunks(start_date, end_date))
total_chunks = len(all_chunks)

downloaded = 0
skipped = 0
failed = 0
start_time = time.time()

for chunk_idx, (chunk_start, chunk_end) in enumerate(all_chunks, 1):
    print(f"\n[{chunk_idx:2d}/{total_chunks}] {chunk_start.date()} → {chunk_end.date()}")
    
    products = list(collection.search(dtstart=chunk_start, dtend=chunk_end))
    print(f"   Found {len(products)} products")
    
    for product in tqdm(products, desc="Progress", unit="file", leave=True):
        out_path = os.path.join(OUTPUT_DIR, f"{product}.nc")
        
        if os.path.exists(out_path):
            skipped += 1
            continue

        success = False
        for attempt in range(3):
            if download_with_progress(product, out_path):
                downloaded += 1
                success = True
                break
            time.sleep(7)   # backoff on failure
        
        if not success:
            failed += 1

    time.sleep(1.5)   # Be respectful to the API

# ====================== SUMMARY ======================
elapsed_hours = (time.time() - start_time) / 3600

print("\n" + "="*75)
print("🎉 3-MONTH PROTOTYPE DOWNLOAD FINISHED!")
print("="*75)
print(f"Period          : {start_date.date()}  →  {end_date.date()}")
print(f"Downloaded      : {downloaded:,} files")
print(f"Skipped         : {skipped:,} files")
print(f"Failed          : {failed:,} files")
print(f"Total time      : {elapsed_hours:.1f} hours")
print(f"Output folder   : {os.path.abspath(OUTPUT_DIR)}")
print("="*75)
print("Ready for preprocessing & model training!")