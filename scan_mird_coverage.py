import os
from scipy.io import loadmat

MIRD_ROOT = r"C:\Users\kirit\OneDrive\Documents\Semester 2\Introduction to AIML\Impulse_MIRD"

def unwrap(x):
    try:
        return x.item()
    except Exception:
        return x

rows = []
for root, _, files in os.walk(MIRD_ROOT):
    for fname in sorted(files):
        if not fname.endswith(".mat"):
            continue
        mat = loadmat(os.path.join(root, fname))
        meta = mat["metapar"][0, 0]
        sim = mat["simpar"][0, 0]

        azimuth = unwrap(meta["azimuth"])
        distance = unwrap(meta["distance"])
        rt60 = unwrap(meta["reverberation"])
        fs = unwrap(sim["fs"])
        n_channels = mat["impulse_response"].shape[1]

        rows.append((fname, azimuth, distance, rt60, fs, n_channels))

print(f"Found {len(rows)} files across all subdirectories\n")

angles = sorted(set(r[1] for r in rows))
distances = sorted(set(r[2] for r in rows))
rt60s = sorted(set(r[3] for r in rows))

print("\n--- Full Database Coverage Summary ---")
print("Angles covered:   ", angles)
print("Distances covered:", distances)
print("RT60s covered:    ", rt60s)
print(f"\nExpected total files: {len(angles)} angles x {len(distances)} distances x {len(rt60s)} RT60 levels = {len(angles) * len(distances) * len(rt60s)}")
print(f"Successfully located {len(rows)} files.")