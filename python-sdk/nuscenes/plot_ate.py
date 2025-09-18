import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# ===== 入力 =====
raw_path     = Path("/workspace/data.csv")                  # 生データ (class,value)
metrics_path = Path("/workspace/metrics_per_class.csv")     # 集計 (class, ATE, ATE_COV)

# ===== 読み込み =====
df_raw = pd.read_csv(raw_path, header=None, names=["class", "value"])
df_metrics = pd.read_csv(metrics_path).dropna(subset=["ATE", "ATE_STDEV"])
df_metrics["std"]  = np.sqrt(df_metrics["ATE_STDEV"])  # 分散→標準偏差
df_metrics.rename(columns={"ATE": "mean"}, inplace=True)

# クラスの順序と統計量
stats = df_metrics[["class", "mean", "std"]].sort_values("class").reset_index(drop=True)
classes = stats["class"].tolist()
xpos = {c: i for i, c in enumerate(classes)}

# min/max は raw から
ext = df_raw.groupby("class")["value"].agg(["min", "max"]).reindex(classes)
stats = stats.join(ext, on="class")

# ===== 描画 =====
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

box_width = 0.5
face = "#C9B6F2"  # 薄紫
edge = "#6B6B6B"

for _, r in stats.iterrows():
    c  = r["class"];  x = xpos[c]
    mu = float(r["mean"]);  sd = float(r["std"])
    vmin = float(r["min"]);  vmax = float(r["max"])

    # --- whisker: min～max
    ax.vlines(x, vmin, vmax, colors=edge, linewidth=1.5)
    cap = box_width * 0.35
    ax.hlines([vmin, vmax], x - cap/2, x + cap/2, colors=edge, linewidth=1.5)

    # --- box: 平均±σ
    y0 = mu - sd
    h  = 2 * sd
    if h == 0:
        ax.hlines(mu, x - box_width/2, x + box_width/2, colors=edge, linewidth=2)
    else:
        rect = patches.Rectangle(
            (x - box_width/2, y0),
            box_width, h,
            linewidth=1.5, edgecolor=edge, facecolor=face, alpha=0.65
        )
        ax.add_patch(rect)

    # --- 平均の横線
    ax.hlines(mu, x - box_width/2, x + box_width/2, colors=edge, linewidth=2)

    # --- 平均の数値を右側に
    ax.annotate(
        f"{mu:.3f}", xy=(x + box_width/2, mu), xytext=(6, 0),
        textcoords="offset points", ha="left", va="center",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    )

# 右上に σ 一覧
sd_text = "\n".join([f"{r['class']}: σ={r['std']:.4f}" for _, r in stats.iterrows()])
ax.text(0.995, 0.995, sd_text, transform=ax.transAxes, ha="right", va="top",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# 軸など
ax.set_xticks([xpos[c] for c in classes])
ax.set_xticklabels(classes)
ax.set_xlabel("Class")
ax.set_ylabel("Value")
ax.set_title("ATE_STDEV with min–max ")

fig.tight_layout()
out_path = "/workspace/metrics_box_sd.png"
fig.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
