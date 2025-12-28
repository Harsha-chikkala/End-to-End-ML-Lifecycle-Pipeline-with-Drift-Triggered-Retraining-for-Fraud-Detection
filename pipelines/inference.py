import pandas as pd
import joblib
import os

ARTIFACTS_DIR = "models/artifacts/"
MODEL_VERSION = "v1.0.0"

def run_inference(batch_path):
    # load artifacts
    model = joblib.load(os.path.join(ARTIFACTS_DIR, f"model_{MODEL_VERSION}.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, f"scaler_{MODEL_VERSION}.pkl"))

    # read batch
    df = pd.read_csv(batch_path)

    # separate features
    X = df.drop(columns=["Class"], errors="ignore")

    # scale using saved scaler
    X_scaled = scaler.transform(X)

    # predictions
    preds = model.predict(X_scaled)

    # return dataframe containing predictions + timestamps/batch info
    df_out = df.copy()
    df_out["prediction"] = preds
    return df_out

if __name__ == "__main__":
    batch_file = "data/batches/batch_02.csv"
    output = run_inference(batch_file)
    print(output.head())
