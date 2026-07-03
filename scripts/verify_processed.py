from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

file = Path(
    r"C:\Users\Methvan\Documents\sri_lanka_nowcast\data\processed\MSG2-SEVI-MSGCLTH-0100-0100-20260331063000.000000000Z-NA.npy"
)

arr = np.load(file)

print("=" * 60)
print("FILE:", file.name)
print("=" * 60)

print("Shape :", arr.shape)
print("Dtype :", arr.dtype)
print("Min   :", float(arr.min()))
print("Max   :", float(arr.max()))
print("Mean  :", float(arr.mean()))
print("Std   :", float(arr.std()))
print("NaN count :", np.isnan(arr).sum())
print("Inf count :", np.isinf(arr).sum())

print("\nCenter sample:")
print(arr[120:136, 120:136])

plt.figure(figsize=(8, 8))
plt.imshow(
    arr,
    origin="lower",
    cmap="jet"
)
plt.colorbar(label="Normalized Cloud Top Height")
plt.title(file.name)
plt.tight_layout()
plt.show()