import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pybaselines
from sklearn.preprocessing import minmax_scale

def drop_frac_and_He(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops fractional m/z values, m/z values > 100, and carrier gas m/z

    Args:
        df: a dataframe representing a single sample, containing m/z values

    Returns:
        The dataframe without fractional an carrier gas m/z
    """

    # drop fractional m/z values
    df = df[df["m/z"].transform(round) == df["m/z"]]
    assert df["m/z"].apply(float.is_integer).all(), "not all m/z are integers"

    # drop m/z values greater than 99
    df = df[df["m/z"] < 100]

    # drop carrier gas
    df = df[df["m/z"] != 4]

    return df

def remove_background_abundance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'm/z' and 'abundance' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["abundance_minsub"] = df.groupby(["m/z"])["abundance"].transform(
        lambda x: smooth_baseline_sub(x) 
    )

    return df

def smooth_baseline_sub(x):
    """
    Applies the savgol filter and subtracts the baseline (minimum value) from an signal

    Args:
        x: data signal

    Returns:
        smooth data signal with baseline subtracted
    """
    # savgol filter smoothing
    try:
        x = savgol_filter(x, 21, 4)
    except Exception:
        pass

    # subtraction
    x = x - x.min()
    
    return x 

def scale_abun(df):
    """
    Scale abundance from 0-1 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))

    return df

def preprocess_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a single sample's dataframe

    Args:
        df: datafram to be preprocessed

    Returns:
        fully preprocessed dataframe
    """
    
    df = drop_frac_and_He(df)
    df = remove_background_abundance(df)
    df = scale_abun(df)
    return df