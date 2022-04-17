from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

def get_model(model_name: str) -> object:
    """
    Gets a model

    Args:
        model name: string name of the model

    Returns:
        the model selected
    """

    model = None
    if model_name == "log_reg":
        model = LogisticRegression(penalty="l1", 
                                   solver="liblinear", 
                                   C=10, 
                                   random_state=42)
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=10)
    elif model_name == "lgbm":
        model = LGBMClassifier(metric="binary_logloss",
                               reg_alpha=1, 
                               colsample_bytree=0.4,
                               random_state=42)
    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=[128, 32],
                              activation='relu',
                              solver='adam',
                              max_iter=200,
                              learning_rate_init=0.001)
    elif model_name == "gaussian_nb":
        model = GaussianNB()
    else:
        raise Exception("Not Implemented")

    return model