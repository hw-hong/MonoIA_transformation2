import os
import pandas as pd
import matplotlib.pyplot as plt

# ==== 여기 두 경로만 직접 수정 ====
CSV_PATH = "/nas2/data/heewon.hong/MonoIA_consistency/outputs/monoia_car/monoia_car444_2026-07-05_17-39-49_488_pid33003/consistency_log.csv"
OUT_DIR = "/nas2/data/heewon.hong/MonoIA_consistency/outputs/monoia_car/monoia_car444_2026-07-05_17-39-49_488_pid33003/plots"
# ================================

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
PURPLE = "#8172B2"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()
epoch = epoch_df["epoch"]

# 1. Detection loss (A vs B)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epoch, epoch_df["loss_det_A"], color=BLUE, linewidth=2, label="loss_det_A")
ax.plot(epoch, epoch_df["loss_det_B"], color=ORANGE, linewidth=2, label="loss_det_B")
ax.set_title("Detection Loss (per-epoch mean)")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/detection_loss.png", dpi=150)
plt.close(fig)

# 2. Consistency loss
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epoch, epoch_df["loss_cons"], color=GREEN, linewidth=2, label="loss_cons")
ax.set_title("Consistency Loss (per-epoch mean)")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/consistency_loss.png", dpi=150)
plt.close(fig)

# 3. Dimension difference
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epoch, epoch_df["dim_diff"], color=PURPLE, linewidth=2, label="dim_diff")
ax.set_title("Dimension Difference (per-epoch mean)")
ax.set_xlabel("epoch")
ax.set_ylabel("dim_diff")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/dim_diff.png", dpi=150)
plt.close(fig)

# 4. Angle difference
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epoch, epoch_df["angle_diff"], color=ORANGE, linewidth=2, label="angle_diff")
ax.set_title("Angle Difference (per-epoch mean)")
ax.set_xlabel("epoch")
ax.set_ylabel("angle_diff")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/angle_diff.png", dpi=150)
plt.close(fig)

print("saved plots to", OUT_DIR)
