import os
from pipelines.evaluate_performance import evaluate_batch
from monitoring.drift_simple import compute_drift
from pipelines.log_lifecycle import log_lifecycle

BATCH_DIR = "data/batches/"
MODEL_VERSION = "v1.0.0"

def process_batches():
    batch_files = sorted(os.listdir(BATCH_DIR))

    for batch_file in batch_files:
        if not batch_file.endswith(".csv"):
            continue

        batch_path = os.path.join(BATCH_DIR, batch_file)
        batch_id = os.path.splitext(batch_file)[0]

        # compute drift + f1
        drift_score = compute_drift(batch_path)
        f1 = evaluate_batch(batch_path)

        # log lifecycle result
        log_lifecycle(
            batch_id=batch_id,
            drift_score=drift_score,
            f1_score=f1,
            model_version=MODEL_VERSION
        )

        print(f"{batch_id}: drift={drift_score:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    process_batches()
