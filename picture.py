# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("test_acc_all.csv")

# plt.figure(figsize=(6, 4))

# plt.plot(
#     df["epoch"],
#     df["val_top1"],
#     color="orange",
#     marker="o",
#     linewidth=2,
#     markersize=4
# )

# plt.xlabel("Evaluation Epoch")
# plt.ylabel("Validation Accuracy (%)")

# plt.xticks(
#     range(
#         int(df["epoch"].min()),
#         int(df["epoch"].max()) + 1,
#         10
#     )
# )

# plt.grid(True, linestyle="--", alpha=0.4)

# plt.tight_layout()
# plt.savefig("val_accuracy_all.pdf", bbox_inches="tight")
# plt.savefig("val_accuracy_all.png", dpi=300, bbox_inches="tight")
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "test_acc_ori.csv",
    header=None,
    names=["epoch", "val_top1"]
)

plt.figure(figsize=(5.5, 4))

plt.plot(
    df["epoch"],
    df["val_top1"],
    color="orange",
    marker="o",
    linewidth=2,
    markersize=4
)

plt.xlabel("Evaluation Epoch")
plt.ylabel("Validation Accuracy (%)")

min_epoch = df["epoch"].min()
max_epoch = df["epoch"].max()

plt.xlim(0, int(max_epoch))

# plt.xticks(range(0, int(max_epoch) + 1, 10))
plt.xticks(range(0, 81, 10))
# plt.xticks(df["epoch"])

plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
# plt.savefig("val_accuracy_all.pdf", bbox_inches="tight")
plt.savefig("val_accuracy_my.png", dpi=300, bbox_inches="tight")
plt.show()