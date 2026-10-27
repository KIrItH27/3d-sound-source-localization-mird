import os
from scipy.io import loadmat

MIRD_DIR = r"PATH_TO_YOUR_MIRD_FOLDER"

def unwrap(x):
    try:
        return x.item()
    except Exception:
        return x

rows = []
for fname in sorted(os.listdir(MIRD_DIR)):
    if not fname.endswith(".mat"):
        continue
    mat = loadmat(os.path.join(MIRD_DIR, fname))
    meta = mat["metapar"][0, 0]
    sim = mat["simpar"][0, 0]

    azimuth = unwrap(meta["azimuth"])
    distance = unwrap(meta["distance"])
    rt60 = unwrap(meta["reverberation"])
    fs = unwrap(sim["fs"])
    n_channels = mat["impulse_response"].shape[1]

    rows.append((fname, azimuth, distance, rt60, fs, n_channels))

print(f"Found {len(rows)} files\n")
print(f"{'filename':<70} {'azimuth':>8} {'dist':>6} {'rt60':>6} {'fs':>7} {'ch':>3}")
for r in rows:
    print(f"{r[0]:<70} {r[1]:>8} {r[2]:>6} {r[3]:>6} {r[4]:>7} {r[5]:>3}")

# summary: unique angles / distances / rt60s covered
angles = sorted(set(r[1] for r in rows))
distances = sorted(set(r[2] for r in rows))
rt60s = sorted(set(r[3] for r in rows))

print("\n--- Coverage summary ---")
print("Angles covered:   ", angles)
print("Distances covered:", distances)
print("RT60s covered:    ", rt60s)
print(f"\nExpected full grid: 13 angles x 2 distances = 26 points per RT60")
print(f"You have {len(rows)} files across {len(rt60s)} RT60 level(s)")
