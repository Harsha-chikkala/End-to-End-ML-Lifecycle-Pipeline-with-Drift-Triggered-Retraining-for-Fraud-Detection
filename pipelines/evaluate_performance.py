import sys, os
sys.path.append(os.path.abspath("."))

from sklearn.metrics import f1_score
from pipelines.inference import run_inference

def evaluate_batch(batch_path):
    df = run_inference(batch_path)
    f1 = f1_score(df["Class"], df["prediction"], zero_division=0)
    return float(f1)

if __name__ == "__main__":
    print("F1-score for batch_02:", evaluate_batch("data/batches/batch_02.csv"))
