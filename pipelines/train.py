import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BATCH_PATH = "data/batches/batch_01.csv"
ARTIFACTS_DIR = "models/artifacts/"
VERSION = "v1.0.0"

def initial_training():

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # load first batch
    df = pd.read_csv(BATCH_PATH)

    # separate features & target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocessing (fit once)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # feature stats for drift detection
    feature_means = list(np.mean(X_train_scaled, axis=0))
    feature_stds = list(np.std(X_train_scaled, axis=0))

    stats = {
        "means": feature_means,
        "stds": feature_stds
    }

    # train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # save artifacts
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"model_{VERSION}.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, f"scaler_{VERSION}.pkl"))
    joblib.dump(stats, os.path.join(ARTIFACTS_DIR, f"stats_{VERSION}.pkl"))

    print(f"Initial training complete. Saved model version {VERSION}")

if __name__ == "__main__":
    initial_training()
