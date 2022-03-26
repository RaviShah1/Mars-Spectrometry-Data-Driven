import pandas as pd
import pickle
import argparse

from models import get_model

TRAIN_LABELS = pd.read_csv("data/train_labels.csv", index_col="sample_id")
VALID_LABELS = pd.read_csv("data/val_labels.csv", index_col="sample_id") # stage 2
LABELS = pd.concat([TRAIN_LABELS, VALID_LABELS]) # stage 2

parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
parser.add_argument("-m",
                    "--model",
                    help="name of model", type=str)
parser.add_argument("-f",
                    "--feature",
                    help="name of feature", type=str)
parser.add_argument("-s",
                    "--save",
                    help="enter no or the location to save the model", default='no', type=str)

args = parser.parse_args()

def train(train_df: pd.DataFrame, model_name: str, path: str) -> pd.DataFrame:
    """
    Trains the model

    Args:
        train_df: the training data
        model_name: string name of the model
        path: save path
    """
    fitted_logreg_dict = {}

    # Split into binary classifier for each class
    for col in LABELS.columns:

        y_train_col = LABELS[col]  # Train on one class at a time

        # output the trained model, bind this to a var, then use as input
        # to prediction function
        model = get_model(model_name)
        fitted_logreg_dict[col] = model.fit(train_df.values, y_train_col)  # Train

    pickle.dump(fitted_logreg_dict, open(path, 'wb'))

if __name__ == "__main__":
    train_df = pd.read_csv(f"data/savgol_features/{args.feature}_train.csv", header=[0], low_memory=False)
    train_df.columns = train_df.iloc[0]
    train_df = train_df.drop([0,1]).set_index('temp_bin', drop=True)

    train(train_df, args.model, args.save)