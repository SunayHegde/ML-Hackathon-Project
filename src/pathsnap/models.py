"""Model zoo used in the comparison (one entry per algorithm family)."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from .config import SEED

# name -> (family, factory, max training rows (None = all), short config description)
MODEL_ZOO = {
    "LogisticRegression": (
        "Linear",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
        None, "StandardScaler + L2, C=1.0"),
    "GaussianNB": (
        "Probabilistic",
        lambda: GaussianNB(var_smoothing=1e-9),
        None, "var_smoothing=1e-9"),
    "KNN": (
        "Instance-based",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1)),
        150_000, "k=31, distance-weighted, scaled, 150k-row train subsample"),
    "RandomForest": (
        "Bagging ensemble",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=10,
                                       max_features="sqrt", n_jobs=-1, random_state=SEED),
        600_000, "200 trees, depth 18, leaf 10, 600k-row subsample"),
    "LightGBM": (
        "Gradient boosting",
        lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1, verbose=-1),
        None, "400 trees, lr 0.05, 63 leaves"),
    "MLP": (
        "Neural network",
        lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                              batch_size=1024, learning_rate_init=1e-3,
                                                              max_iter=40, early_stopping=True,
                                                              validation_fraction=0.1, n_iter_no_change=5,
                                                              random_state=SEED)),
        400_000, "(64,32) ReLU, Adam lr 1e-3, early stopping, 400k subsample"),
}
