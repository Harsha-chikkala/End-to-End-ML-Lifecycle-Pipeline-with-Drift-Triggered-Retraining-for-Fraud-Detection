import pandas as pd
from sklearn.metrics import f1_score
from pipelines.inference import run_inference

def evaluate_batch(batch_path):
    # run inference & get predictions + true labels
    df = run_inference(batch_path)

    if "Class" not in df.columns:
        raise ValueError("Ground truth labels not available for this batch")

    y_true = df["Class"]
    y_pred = df["prediction"]

    # compute performance metric
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return float(f1)

if __name__ == "__main__":
    f1 = evaluate_batch("data/batches/batch_02.csv")
    print("F1-score for batch_02:", f1)
