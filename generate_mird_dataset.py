import os
import csv
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import fftconvolve, resample_poly

# ================= Paths (EDIT THESE) =================
MIRD_DIR = r"C:\Users\kirit\OneDrive\Documents\Semester 2\Introduction to AIML\ImpulseMIRD"
AUDIO_DIR = r"C:\Users\kirit\OneDrive\Documents\Semester 2\Software defined communication\initialdiagnosis"
OUTPUT_DIR = "dataset_mird"

# ================= Setup (confirmed from the .mat metadata) =================
MIRD_FS = 48000                 # simpar.fs
FS = 16000                      # same rate as the simulation + speech clips
DOWN = MIRD_FS // FS            # 3 -> clean integer downsample

MIC_A_CH, MIC_B_CH = 0, 7       # mic1 and mic8 = widest baseline of the 8-mic array
MIC_SPACING = 0.26              # cumulative of [3,3,3,8,3,3,3] cm = 26 cm

SOUND_SPEED = 343
N_FFT = 2048
HOP = 512
MAX_VALID_FREQ = SOUND_SPEED / (2 * MIC_SPACING)    # ~660 Hz aliasing limit for this array
FREQ_RES = FS / N_FFT
MAX_VALID_BIN = int(MAX_VALID_FREQ / FREQ_RES)       # ~84 bins

SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30]

RIR_KEEP_SEC = 0.5              # RT60 = 0.16 s, so 0.5 s of RIR is more than enough
PRE_PEAK_SEC = 0.01             # keep 10 ms before the direct-path peak


def unwrap(x):
    try:
        return x.item()
    except Exception:
        return x


def mird_to_sim_angle(az):
    """
    MIRD azimuth: 0 = straight ahead, 90 = toward mic1, 270 = toward mic8.
    Simulation convention: 0..180, 90 = straight ahead.
    -> sim_angle = 90 + signed_azimuth, where signed azimuth is in [-90, 90].
    """
    signed = az if az <= 90 else az - 360
    return 90 + signed


def stft(sig, n_fft=N_FFT, hop=HOP):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(sig) - n_fft) // hop
    if n_frames < 1:
        return None
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = sig[i * hop: i * hop + n_fft] * window
        out[:, i] = np.fft.rfft(frame)
    return out


def add_awgn(signal, snr_db):
    sig_var = np.var(signal)
    if sig_var == 0:
        return signal
    noise_var = sig_var / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_var), len(signal))


def extract_ipd(sig_a, sig_b):
    Ya = stft(sig_a)
    Yb = stft(sig_b)
    if Ya is None or Yb is None:
        return None
    m = min(Ya.shape[1], Yb.shape[1])
    Ya, Yb = Ya[:, :m], Yb[:, :m]
    ipd = np.angle(Ya / Yb) * 180 / np.pi              # wrapped phase, degrees, no unwrap
    return ipd[:MAX_VALID_BIN, :].astype(np.float32)   # crop to this array's valid band


def load_audio_clips(audio_dir):
    clips = []
    for fname in sorted(os.listdir(audio_dir)):
        if fname.lower().endswith(".wav"):
            src_fs, audio = wavfile.read(os.path.join(audio_dir, fname))
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if src_fs != FS:
                print(f"Skipping {fname}: sample rate {src_fs} != {FS}")
                continue
            clips.append(audio)
    return clips


def load_rirs():
    """Returns {(azimuth, distance): (n, 2) float64 array at 16 kHz} using channels MIC_A_CH / MIC_B_CH."""
    rirs = {}
    for fname in sorted(os.listdir(MIRD_DIR)):
        if not fname.endswith(".mat"):
            continue
        mat = loadmat(os.path.join(MIRD_DIR, fname))
        meta = mat["metapar"][0, 0]
        az = int(unwrap(meta["azimuth"]))
        dist = int(unwrap(meta["distance"]))

        spacing_m = np.array(meta["mic_spacing"]).flatten() / 100.0
        pos = np.concatenate([[0.0], np.cumsum(spacing_m)])
        actual = abs(pos[MIC_B_CH] - pos[MIC_A_CH])
        if abs(actual - MIC_SPACING) > 1e-3:
            raise ValueError(f"{fname}: mic spacing {actual} m != expected {MIC_SPACING} m")

        rir = mat["impulse_response"][:, [MIC_A_CH, MIC_B_CH]].astype(np.float64)

        # trim the bulk recording latency identically on both channels (preserves inter-mic delay)
        peak = int(np.argmax(np.abs(rir).max(axis=1)))
        start = max(0, peak - int(PRE_PEAK_SEC * MIRD_FS))
        rir = rir[start: start + int(RIR_KEEP_SEC * MIRD_FS)]

        rirs[(az, dist)] = resample_poly(rir, 1, DOWN, axis=0)   # 48 kHz -> 16 kHz
    return rirs


def weighted_mean_ipd(sig_a, sig_b):
    """Magnitude-weighted mean IPD (deg) over low, unambiguous bins -- same check used on the sim data."""
    Ya, Yb = stft(sig_a), stft(sig_b)
    m = min(Ya.shape[1], Yb.shape[1])
    Ya, Yb = Ya[2:40, :m], Yb[2:40, :m]
    w = np.abs(Ya) * np.abs(Yb)
    w = w / w.sum()
    return float(np.sum(np.angle(Ya / Yb) * w) * 180 / np.pi)


def check_orientation(rirs, audio):
    """
    Simulation convention: source on the mic1 side (sim angle > 90) gives POSITIVE IPD.
    MIRD azimuth 90 points toward mic1, so it must come out positive. If not, swap channel order.
    """
    r90, r270 = rirs[(90, 1)], rirs[(270, 1)]
    m90 = weighted_mean_ipd(fftconvolve(audio, r90[:, 0]), fftconvolve(audio, r90[:, 1]))
    m270 = weighted_mean_ipd(fftconvolve(audio, r270[:, 0]), fftconvolve(audio, r270[:, 1]))
    print(f"Orientation check -> az=90: {m90:+.1f} deg | az=270: {m270:+.1f} deg")

    if m90 < 0:
        print("Sign is opposite to the simulation convention -> swapping channel order.")
        rirs = {k: v[:, ::-1].copy() for k, v in rirs.items()}
        m90 = -m90
        m270 = -m270
        print(f"After swap -> az=90: {m90:+.1f} deg | az=270: {m270:+.1f} deg")

    if not (m90 > 0 and m270 < 0):
        print("WARNING: az=90 / az=270 are not cleanly opposite in sign. Inspect before trusting labels.")
    return rirs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clips = load_audio_clips(AUDIO_DIR)
    if not clips:
        raise RuntimeError("No valid 16 kHz .wav clips in AUDIO_DIR")
    rirs = load_rirs()
    print(f"Loaded {len(rirs)} RIR files, {len(clips)} speech clips")
    print(f"Mic spacing {MIC_SPACING} m -> valid band up to {MAX_VALID_FREQ:.0f} Hz ({MAX_VALID_BIN} bins)")

    rirs = check_orientation(rirs, clips[0])

    total = len(rirs) * len(clips) * len(SNR_LEVELS)
    print(f"Generating up to {total} samples...")
    idx = 0

    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle", "distance", "rt60", "snr", "clip_id", "mird_azimuth"])

        for (az, dist), rir in sorted(rirs.items()):
            sim_angle = mird_to_sim_angle(az)
            for clip_id, audio in enumerate(clips):
                clean_a = fftconvolve(audio, rir[:, 0])      # convolve once, reuse across SNRs
                clean_b = fftconvolve(audio, rir[:, 1])
                for snr in SNR_LEVELS:
                    ipd = extract_ipd(add_awgn(clean_a, snr), add_awgn(clean_b, snr))
                    if ipd is None:
                        continue
                    fname = f"mird_{idx:06d}.npy"
                    np.save(os.path.join(OUTPUT_DIR, fname), ipd)
                    writer.writerow([fname, sim_angle, dist, 0.16, snr, clip_id, az])
                    idx += 1
                    if idx % 500 == 0:
                        print(f"  {idx}/{total}")

    print(f"Done. {idx} samples saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
