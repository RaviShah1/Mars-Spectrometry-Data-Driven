import pandas as pd
import numpy as np
import itertools


def abun_per_tempbin(df: pd.DataFrame, feature: str, interval: int) -> pd.DataFrame:
    """
    Transforms dataset to take the preprocessed feature abundance for each
    temperature range for each m/z value

    Args:
        df: dataframe to transform
        feature: string name of the feature to extract
        interval: temperature interval

    Returns:
        transformed dataframe
    """

    # Create a series of temperature bins
    temprange = pd.interval_range(start=-100, end=1500, freq=interval)

    # Make dataframe with rows that are combinations of all temperature bins
    # and all m/z values
    allcombs = list(itertools.product(temprange, [*range(0, 100)]))
    allcombs_df = pd.DataFrame(allcombs, columns=["temp_bin", "m/z"])

    # Bin temperatures
    df["temp_bin"] = pd.cut(df["temp"], bins=temprange)

    # Combine with a list of all temp bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["temp_bin", "m/z"], how="left")

    # Aggregate to temperature bin level to find max
    if feature == "max":
        df = df.groupby(["temp_bin", "m/z"]).max("abun_minsub_scaled").reset_index()
    elif feature == "mean":
        df = df.groupby(["temp_bin", "m/z"]).mean("abun_minsub_scaled").reset_index()
    else:
        raise Exception("Not Implemented")

    # Fill in 0 for abundance values without information
    df = df.replace(np.nan, 0)

    # Reshape so each row is a single sample
    df = df.pivot_table(columns=["m/z", "temp_bin"], values=["abun_minsub_scaled"])

    return df
