import os
import csv
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import fftconvolve, resample_poly

# ================= Paths =================
MIRD_ROOT = r"C:\Users\kirit\OneDrive\Documents\Semester 2\Introduction to AIML\Impulse_MIRD"
AUDIO_DIR = r"C:\Users\kirit\OneDrive\Documents\Semester 2\Software defined communication\initialdiagnosis"
OUTPUT_DIR = "dataset_mird"

# ================= Setup =================
MIRD_FS = 48000                 
FS = 16000                      
DOWN = MIRD_FS // FS            

MIC_A_CH, MIC_B_CH = 0, 7       
MIC_SPACING = 0.26              

SOUND_SPEED = 343
N_FFT = 2048
HOP = 512
MAX_VALID_FREQ = SOUND_SPEED / (2 * MIC_SPACING)    
FREQ_RES = FS / N_FFT
MAX_VALID_BIN = int(MAX_VALID_FREQ / FREQ_RES)       

SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30]

RIR_KEEP_SEC = 0.5              
PRE_PEAK_SEC = 0.01             


def unwrap(x):
    try:
        return x.item()
    except Exception:
        return x

def mird_to_sim_angle(az):
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
    ipd = np.angle(Ya / Yb) * 180 / np.pi              
    return ipd[:MAX_VALID_BIN, :].astype(np.float32)   

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
    """Traverses all subfolders in MIRD_ROOT and maps to (azimuth, distance, rt60)."""
    rirs = {}
    for root, _, files in os.walk(MIRD_ROOT):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            mat = loadmat(os.path.join(root, fname))
            meta = mat["metapar"][0, 0]
            
            az = int(unwrap(meta["azimuth"]))
            dist = int(unwrap(meta["distance"]))
            rt60 = float(unwrap(meta["reverberation"]))

            spacing_m = np.array(meta["mic_spacing"]).flatten() / 100.0
            pos = np.concatenate([[0.0], np.cumsum(spacing_m)])
            actual = abs(pos[MIC_B_CH] - pos[MIC_A_CH])
            if abs(actual - MIC_SPACING) > 1e-3:
                continue # Skip arrays that don't match our strict 26cm spacing target

            rir = mat["impulse_response"][:, [MIC_A_CH, MIC_B_CH]].astype(np.float64)

            peak = int(np.argmax(np.abs(rir).max(axis=1)))
            start = max(0, peak - int(PRE_PEAK_SEC * MIRD_FS))
            
            # Dynamically adjust keep time based on the specific RT60 to capture full tail
            keep_samples = int((rt60 + 0.2) * MIRD_FS)
            rir = rir[start: start + keep_samples]

            rirs[(az, dist, rt60)] = resample_poly(rir, 1, DOWN, axis=0)   
    return rirs

def weighted_mean_ipd(sig_a, sig_b):
    Ya, Yb = stft(sig_a), stft(sig_b)
    m = min(Ya.shape[1], Yb.shape[1])
    Ya, Yb = Ya[2:40, :m], Yb[2:40, :m]
    w = np.abs(Ya) * np.abs(Yb)
    w = w / w.sum()
    return float(np.sum(np.angle(Ya / Yb) * w) * 180 / np.pi)

def check_orientation(rirs, audio):
    # Grab the first available 90-degree and 270-degree RIRs across any RT60 level
    r90 = next(v for k, v in rirs.items() if k[0] == 90 and k[1] == 1)
    r270 = next(v for k, v in rirs.items() if k[0] == 270 and k[1] == 1)
    
    m90 = weighted_mean_ipd(fftconvolve(audio, r90[:, 0]), fftconvolve(audio, r90[:, 1]))
    m270 = weighted_mean_ipd(fftconvolve(audio, r270[:, 0]), fftconvolve(audio, r270[:, 1]))
    print(f"Orientation check -> az=90: {m90:+.1f} deg | az=270: {m270:+.1f} deg")

    if m90 < 0:
        print("Sign is opposite to the simulation convention -> swapping channel order.")
        rirs = {k: v[:, ::-1].copy() for k, v in rirs.items()}
        m90 = -m90
        m270 = -m270
        print(f"After swap -> az=90: {m90:+.1f} deg | az=270: {m270:+.1f} deg")
    return rirs

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clips = load_audio_clips(AUDIO_DIR)
    if not clips:
        raise RuntimeError("No valid 16 kHz .wav clips in AUDIO_DIR")
    rirs = load_rirs()
    print(f"Loaded {len(rirs)} RIR files from all folders, {len(clips)} speech clips")

    rirs = check_orientation(rirs, clips[0])

    total = len(rirs) * len(clips) * len(SNR_LEVELS)
    print(f"Generating up to {total} samples...")
    idx = 0

    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle", "distance", "rt60", "snr", "clip_id", "mird_azimuth"])

        for (az, dist, rt60), rir in sorted(rirs.items()):
            sim_angle = mird_to_sim_angle(az)
            for clip_id, audio in enumerate(clips):
                clean_a = fftconvolve(audio, rir[:, 0])      
                clean_b = fftconvolve(audio, rir[:, 1])
                for snr in SNR_LEVELS:
                    ipd = extract_ipd(add_awgn(clean_a, snr), add_awgn(clean_b, snr))
                    if ipd is None:
                        continue
                    fname = f"mird_{idx:06d}.npy"
                    np.save(os.path.join(OUTPUT_DIR, fname), ipd)
                    writer.writerow([fname, sim_angle, dist, rt60, snr, clip_id, az])
                    idx += 1
                    if idx % 500 == 0:
                        print(f"  {idx}/{total}")

    print(f"Done. {idx} samples saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()