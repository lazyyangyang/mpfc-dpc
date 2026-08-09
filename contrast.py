import pandas as pd
import matplotlib.pyplot as plt

df_ours = pd.read_csv("cifar100_IPC10_my.csv")
df_baseline = pd.read_csv("cifar100_IPC10__ori.csv")

plt.figure(figsize=(6, 4))

plt.plot(
    df_baseline["epoch"],
    df_baseline["val_top1"],
    color="blue",
    marker="s",
    linewidth=2,
    markersize=4,
    label="Baseline"
)

plt.plot(
    df_ours["epoch"],
    df_ours["val_top1"],
    color="orange",
    marker="o",
    linewidth=2,
    markersize=4,
    label="Ours"
)

plt.xlabel("Evaluation Epoch")
plt.ylabel("Validation Accuracy (%)")

min_epoch = min(df_ours["epoch"].min(), df_baseline["epoch"].min())
max_epoch = max(df_ours["epoch"].max(), df_baseline["epoch"].max())

plt.xticks(range(int(min_epoch), int(max_epoch) + 1, 10))

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
# plt.savefig("val_accuracy_comparison.pdf", bbox_inches="tight")
plt.savefig("val_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()