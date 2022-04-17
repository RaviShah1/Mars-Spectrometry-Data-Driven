import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import sys
sys.path.append("..")

def plot_instrument_distribution(meta):
    meta_instrument = (
        meta.reset_index()
        .groupby(["split", "instrument_type"])["sample_id"]
        .aggregate("count")
        .reset_index()
    )
    meta_instrument = meta_instrument.pivot(
        index="split", columns="instrument_type", values="sample_id"
        ).reset_index()

    ax = meta_instrument.plot(
            x="split",
            kind="barh",
            stacked=True,
            title="Instrument type by data split",
            mark_right=True
    )

    ax.bar_label(ax.containers[0], label_type="center")
    ax.bar_label(ax.containers[1], label_type="center")
    plt.show()

if __name__ == "__main__":
    meta = pd.read_csv("data/metadata.csv")
    plot_instrument_distribution(meta)
