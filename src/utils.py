"""Shared helpers for the road sign classification project.

Every notebook imports from here so that image size, split files, feature
file names, and timing are identical across the three feature tracks.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------- constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"   # data/processed/<class>/<file>.png
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"         # train.csv, val.csv, test.csv
FEATURES_DIR = PROJECT_ROOT / "features"               # <feature>_<split>.npy
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42
IMAGE_SIZE = (128, 128)                                # (width, height) for every cropped sign
CLASSES = ["speedlimit", "stop", "crosswalk", "trafficlight"]


# --------------------------------------------------------------- data access
def load_split(name: str) -> pd.DataFrame:
    """Return the split CSV (columns: path, label) for 'train', 'val' or 'test'.

    Row order here is the row order of every cached feature file.
    """
    return pd.read_csv(DATA_SPLITS / f"{name}.csv")


def load_image(path, size=IMAGE_SIZE) -> np.ndarray:
    """Load one cropped sign as an RGB uint8 array of shape (H, W, 3)."""
    full = Path(path)
    if not full.is_absolute():
        full = PROJECT_ROOT / full
    return np.asarray(Image.open(full).convert("RGB").resize(size))


def load_split_images(name: str):
    """Return (images, labels) for a split: images (N, H, W, 3), labels (N,) strings."""
    df = load_split(name)
    images = np.stack([load_image(p) for p in df["path"]])
    return images, df["label"].to_numpy()


# -------------------------------------------------------- feature caching
def save_features(X: np.ndarray, feature_name: str, split: str) -> Path:
    """Save features as features/<feature_name>_<split>.npy (rows follow the split CSV)."""
    FEATURES_DIR.mkdir(exist_ok=True)
    out = FEATURES_DIR / f"{feature_name}_{split}.npy"
    np.save(out, X)
    return out


def load_features(feature_name: str, split: str) -> np.ndarray:
    return np.load(FEATURES_DIR / f"{feature_name}_{split}.npy")


# ------------------------------------------------------------------ timing
def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


# -------------------------------------------------------------- evaluation
def evaluate(y_true, y_pred, classes=CLASSES):
    """Return (accuracy, per_class_report_df, confusion_matrix_array)."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return acc, per_class, cm


def plot_confusion_matrix(cm, classes=CLASSES, title="Confusion matrix", ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return ax


# --------------------------------------------------------- results logging
RESULTS_CSV = RESULTS_DIR / "results.csv"


def record_result(feature, classifier, val_acc, test_acc, train_time_s, infer_time_s,
                  best_params=None, per_class_df=None, feature_time_s=None):
    """Append one feature x classifier result row so notebook 05 can compare all six combos.

    Re-running a notebook replaces its old row for the same (feature, classifier).
    feature_time_s = total time spent extracting the features (counts toward cost).
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    row = {
        "feature": feature, "classifier": classifier,
        "val_acc": val_acc, "test_acc": test_acc,
        "feature_time_s": feature_time_s,
        "train_time_s": train_time_s, "infer_time_s": infer_time_s,
        "best_params": str(best_params),
    }
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        df = df[~((df["feature"] == feature) & (df["classifier"] == classifier))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False)
    if per_class_df is not None:
        out_dir = RESULTS_DIR / "per_class"
        out_dir.mkdir(exist_ok=True)
        per_class_df.to_csv(out_dir / f"{feature}_{classifier}.csv")
