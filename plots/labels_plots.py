import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import sys
sys.path.append("..")

def plot_labels_distribution(labels):
    sumlabs = labels.aggregate("sum").sort_values()
    plt.barh(sumlabs.index, sumlabs, align="center")
    plt.ylabel("Compounds")
    plt.xticks(rotation=45)
    plt.xlabel("Count in Training Set")
    plt.title("Compounds Represented in Training Set")
    plt.show()

def plot_heatmap(labels):
    sns.heatmap(labels.corr())
    plt.show()

if __name__ == "__main__":
    labels = pd.read_csv("data/train_labels.csv", index_col="sample_id")
    plot_labels_distribution(labels)
    plot_heatmap(labels)