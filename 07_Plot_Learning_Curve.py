# =========================================================
# 6)  Average across folds  &  plot
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt

results_by_fraction = pd.read_csv("./results_by_fraction.csv")
agg = pd.read_csv("./aggregated_results.csv")

df = pd.DataFrame(results_by_fraction)

# -------- plotting code unchanged except use 'agg' ----------

figsize = (11, 7)
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
markers    = ["o", "s", "^", "v", "x", "D", ">", "<", "P"]

topic_translations = {
    "Traurigkeit": "1) Reported sadness",
    "Anspannung": "2) Inner tension",
    "Schlaf": "3) Sleep disturbances",
    "Appetit": "4) Loss of appetite",
    "Konzentration": "5) Difficulties concentrating",
    "Antriebslosigkeit": "6) Lassitude",
    "Gefühlslosigkeit": "7) Emotional numbness",
    "Gedanken": "8) Pessimistic thoughts",
    "Suizid": "9) Suicidal ideations"
}
topic_order = list(topic_translations.keys())

# Strict accuracy curve
plt.figure(figsize=figsize)
for i, t in enumerate(topic_order):
    td = agg[agg["topic"] == t]
    plt.errorbar(
        td["fraction"],
        td["strict_acc"],
        yerr=td["strict_sem"],
        linestyle=linestyles[i % len(linestyles)],
        marker   =markers[i % len(markers)],
        capsize  =3,          # small horizontal bars at tips
        linewidth=1.5,
        markersize=5,
        label=topic_translations[t]
    )
plt.title("Strict Accuracy vs. Training-set Fraction")
plt.xlabel("Training set size (% of full dataset)")
plt.ylabel("Strict Accuracy [%]")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./Strict_Topic_Curve.tiff",
            format="png", dpi=150)
plt.close()

# ---- repeat for flexible as before (replace y-value) -----
# Flexible Accuracy Plot
plt.figure(figsize=figsize)
for i, t in enumerate(topic_order):
    td = agg[agg["topic"] == t]
    plt.errorbar(
        td["fraction"],
        td["flexible_acc"],          
        yerr=td["flexible_sem"],     
        linestyle=linestyles[i % len(linestyles)],
        marker=markers[i % len(markers)],
        capsize=3,
        linewidth=1.5,
        markersize=5,
        label=topic_translations[t]
    )
plt.title("Flexible Accuracy vs. Training-set Fraction")
plt.xlabel("Training set size (% of full dataset)")
plt.ylabel("Flexible Accuracy (±1) [%]")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./Flexible_Topic_Curve.tiff",
            format="tiff", dpi=600)
plt.close()
print("Done plotting learning curves with per-topic lines.")
