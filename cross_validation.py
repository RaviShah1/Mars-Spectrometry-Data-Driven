import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models import get_model

def evaluate_cv(model_name: str, train: pd.DataFrame):
    """
    Evaluates the Cross Validation of a model

    Args:
        model name: string name of the model
        train: training data
    """

    model = get_model(model_name)
    res = logloss_cross_val(model, np.array(train, dtype='float32')) # model, train

    print("Log Loss")
    print(res[0])
    print(f"Average Log Loss: {res[1]}")

def logloss_cross_val(clf, X, y=None):
    """
    Check log loss score for model

    Args:
        clf: model
        X: input features
        y: true predictions

    Returns:
        log_loss_cv dictionary containing cross validation information
        avg_log_loss is the average log loss
    """

    if y is None:
        y = pd.read_csv("data/train_labels.csv", index_col="sample_id")
        y2 = pd.read_csv("data/val_labels.csv", index_col="sample_id") # stage 2
        y = pd.concat([y, y2]) # stage 2

    # Define stratified k-fold validation
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True) # 10

    # Define log loss
    log_loss_scorer = make_scorer(log_loss, needs_proba=True)

    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:

        y_col = y[col]  # take one label at a time
        log_loss_cv[col] = np.mean(
            cross_val_score(clf, X, y_col, cv=skf, scoring=log_loss_scorer) #X.values
        )

    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss