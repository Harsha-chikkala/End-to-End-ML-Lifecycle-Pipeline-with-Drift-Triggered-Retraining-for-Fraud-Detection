import pandas as pd
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS_DIR = "models/artifacts/"
BATCH_DIR = "data/batches/"

def retrain_model(new_version):
    all_data = []

    # collect all existing batches
    for batch_file in sorted(os.listdir(BATCH_DIR)):
        if batch_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(BATCH_DIR, batch_file))
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # features & target
    X = df_all.drop(columns=["Class"])
    y = df_all["Class"]

    # preprocessing
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # save artifacts using new version
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"model_{new_version}.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, f"scaler_{new_version}.pkl"))

    print(f"Retraining complete → saved new model version: {new_version}")


if __name__ == "__main__":
    # take version from argument
    new_version_arg = sys.argv[1]
    retrain_model(new_version_arg)
