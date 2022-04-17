import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import sys
#sys.path.append("..")
from preprocess import drop_frac_and_He, remove_background_abundance

def plot_abundance_vs_temp(sample_df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Abundance values across temperature by m/z")
    fig.supxlabel("Temperature")
    fig.supylabel("Abundance levels")

    plt.subplot(1, 2, 1)
    for m in sample_df["m/z"].unique():
        plt.plot(
            sample_df[sample_df["m/z"] == m]["temp"],
            sample_df[sample_df["m/z"] == m]["abundance"],
        )
    plt.title("Before subtracting minimum abundance")

    # After subtracting minimum abundance value
    sample_df = remove_background_abundance(sample_df)

    plt.subplot(1, 2, 2)
    for m in sample_df["m/z"].unique():
        plt.plot(
            sample_df[sample_df["m/z"] == m]["temp"],
            sample_df[sample_df["m/z"] == m]["abundance_minsub"],
        )
    plt.title("After subtracting minimum abundance")
    plt.show()

if __name__ == "__main__":
    sample_df = pd.read_csv("data/train_features/S0000.csv")
    #sample_df = pd.read_csv("data/train_features/S0754.csv")
    df = drop_frac_and_He(sample_df)
    plot_abundance_vs_temp(sample_df)
    