import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ================= Config =================
DATASET_DIR = "dataset_mird"
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANGLE_MAX = 180.0
DIST_MAX = 2.0
ANGLE_BASELINE = 7.5   # deg  (paper's threshold: half of the 15 deg grid step)
DIST_BASELINE = 0.5    # m    (half of the 1 m grid step)


def get_max_frames(df):
    max_t = 0
    for fname in df["filename"]:
        max_t = max(max_t, np.load(os.path.join(DATASET_DIR, fname), mmap_mode="r").shape[1])
    return max_t


class IPDDataset(Dataset):
    def __init__(self, df, max_t):
        self.df = df.reset_index(drop=True)
        self.max_t = max_t

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(os.path.join(DATASET_DIR, row["filename"]))   # (F, T)

        f, t = arr.shape
        if t < self.max_t:
            arr = np.concatenate([arr, np.zeros((f, self.max_t - t), dtype=np.float32)], axis=1)
        elif t > self.max_t:
            arr = arr[:, :self.max_t]

        x = torch.from_numpy(arr / 180.0).unsqueeze(0).float()
        y = torch.tensor([row["angle"] / ANGLE_MAX, row["distance"] / DIST_MAX], dtype=torch.float32)
        return x, y


class CNN_R(nn.Module):
    """Identical architecture to the simulation model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(16 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


def evaluate(model, loader, criterion, collect=False):
    model.eval()
    total_loss, a_ok, d_ok, n = 0.0, 0, 0, 0
    a_err, d_err = 0.0, 0.0
    all_true_a, all_pred_a = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)

            pa, pd_ = pred[:, 0] * ANGLE_MAX, pred[:, 1] * DIST_MAX
            ta, td = y[:, 0] * ANGLE_MAX, y[:, 1] * DIST_MAX

            a_ok += (torch.abs(pa - ta) < ANGLE_BASELINE).sum().item()
            d_ok += (torch.abs(pd_ - td) < DIST_BASELINE).sum().item()
            a_err += torch.abs(pa - ta).sum().item()
            d_err += torch.abs(pd_ - td).sum().item()
            n += x.size(0)

            if collect:
                all_true_a.append(ta.cpu().numpy())
                all_pred_a.append(pa.cpu().numpy())

    out = dict(loss=total_loss / n, angle_acc=a_ok / n * 100, dist_acc=d_ok / n * 100,
               angle_mae_deg=a_err / n, dist_mae_m=d_err / n)
    if collect:
        out["true_angle"] = np.concatenate(all_true_a)
        out["pred_angle"] = np.concatenate(all_pred_a)
    return out


def main():
    labels = pd.read_csv(LABELS_CSV)

    # ---- split by SPEECH CLIP: test clips are never seen in training ----
    # (only 26 real positions exist, so a random sample split would just test memorised positions
    #  with the same speech; splitting by clip tests generalisation to new speech + noise)
    clip_ids = labels["clip_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(clip_ids)
    n = len(clip_ids)
    n_tr, n_va = int(0.7 * n), int(0.1 * n)
    tr_c, va_c, te_c = clip_ids[:n_tr], clip_ids[n_tr:n_tr + n_va], clip_ids[n_tr + n_va:]

    train_df = labels[labels["clip_id"].isin(tr_c)]
    val_df = labels[labels["clip_id"].isin(va_c)]
    test_df = labels[labels["clip_id"].isin(te_c)]
    print(f"Clips  -> train {len(tr_c)} / val {len(va_c)} / test {len(te_c)}")
    print(f"Samples-> train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")

    MAX_T = get_max_frames(labels)
    print("Padding time axis to:", MAX_T, "frames")

    pin = DEVICE == "cuda"
    mk = lambda df, shuffle: DataLoader(IPDDataset(df, MAX_T), batch_size=BATCH_SIZE, shuffle=shuffle,
                                        num_workers=NUM_WORKERS, pin_memory=pin)
    train_loader, val_loader, test_loader = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    model = CNN_R().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)

    train_losses, val_losses = [], []
    best_val = float("inf")

    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        tr_loss = running / len(train_loader.dataset)
        v = evaluate(model, val_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(v["loss"])

        print(f"Epoch {epoch:3d} | train MAE {tr_loss:.4f} | val MAE {v['loss']:.4f} "
              f"| val angle acc {v['angle_acc']:.1f}% | val dist acc {v['dist_acc']:.1f}%")

        if v["loss"] < best_val:
            best_val = v["loss"]
            torch.save(model.state_dict(), "best_cnn_r_mird.pt")

    # ---- training curve ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training MAE")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("MIRD (real RIR): Training and Validation MAE")
    plt.legend(); plt.grid(True)
    plt.savefig("mird_training_curve.png", dpi=150)
    plt.show()

    # ---- final test ----
    model.load_state_dict(torch.load("best_cnn_r_mird.pt", weights_only=True))
    t = evaluate(model, test_loader, criterion, collect=True)
    print(f"\nTEST | MAE {t['loss']:.4f} | angle acc {t['angle_acc']:.1f}% | distance acc {t['dist_acc']:.1f}%")
    print(f"     | angle MAE {t['angle_mae_deg']:.2f} deg | distance MAE {t['dist_mae_m']:.3f} m")

    # ---- per-angle breakdown (useful for the report) ----
    print("\nPer-angle test results (sim convention: 90 = straight ahead)")
    print(f"{'angle':>6} {'acc %':>8} {'MAE deg':>9}")
    for a in sorted(np.unique(t["true_angle"].round())):
        mask = t["true_angle"].round() == a
        err = np.abs(t["pred_angle"][mask] - t["true_angle"][mask])
        print(f"{a:6.0f} {(err < ANGLE_BASELINE).mean() * 100:8.1f} {err.mean():9.2f}")


if __name__ == "__main__":
    main()
