import pandas as pd
import joblib
import numpy as np
import os

ARTIFACTS_DIR = "models/artifacts/"
MODEL_VERSION = "v1.0.0"

def compute_drift(batch_path):
    # load stats saved during initial training
    stats = joblib.load(os.path.join(ARTIFACTS_DIR, f"stats_{MODEL_VERSION}.pkl"))
    mean_ref = np.array(stats["means"])
    std_ref = np.array(stats["stds"])

    # read batch
    df = pd.read_csv(batch_path)
    X = df.drop(columns=["Class"], errors="ignore")

    # use saved scaler to transform batch (must match training)
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, f"scaler_{MODEL_VERSION}.pkl"))
    X_scaled = scaler.transform(X)

    # current batch stats
    mean_batch = np.mean(X_scaled, axis=0)
    std_batch = np.std(X_scaled, axis=0)

    # drift score
    drift_score = np.mean(np.abs(mean_ref - mean_batch)) + np.mean(np.abs(std_ref - std_batch))
    
    return float(drift_score)

if __name__ == "__main__":
    score = compute_drift("data/batches/batch_02.csv")
    print("Drift score for batch_02:", score)
