import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS_DIR = "models/artifacts/"
BATCH_DIR = "data/batches/"
NEW_VERSION = "v1.1.0"  # manual bump for now

def retrain_model():
    all_data = []

    # combine all processed batches
    for batch_file in sorted(os.listdir(BATCH_DIR)):
        if batch_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(BATCH_DIR, batch_file))
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # separate features & labels
    X = df_all.drop(columns=["Class"])
    y = df_all["Class"]

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # retrain model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # save artifacts for new version
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"model_{NEW_VERSION}.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, f"scaler_{NEW_VERSION}.pkl"))

    print(f"Retraining complete → saved new model version: {NEW_VERSION}")

if __name__ == "__main__":
    retrain_model()
