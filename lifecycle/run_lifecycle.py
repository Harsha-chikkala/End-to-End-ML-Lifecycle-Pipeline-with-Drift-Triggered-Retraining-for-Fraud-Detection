import os
from pipelines.evaluate_performance import evaluate_batch
from monitoring.drift_simple import compute_drift
from pipelines.log_lifecycle import log_lifecycle
import subprocess

BATCH_DIR = "data/batches/"
REGISTRY_FILE = "models/registry.txt"

DRIFT_THRESHOLD = 0.60
F1_THRESHOLD = 0.70


def load_active_version():
    if not os.path.exists(REGISTRY_FILE):
        return "v1.0.0"
    with open(REGISTRY_FILE, "r") as f:
        return f.read().strip()


def save_active_version(version):
    with open(REGISTRY_FILE, "w") as f:
        f.write(version)


def lifecycle_run():
    MODEL_VERSION = load_active_version()

    for batch_file in sorted(os.listdir(BATCH_DIR)):
        if not batch_file.endswith(".csv"):
            continue

        batch_path = os.path.join(BATCH_DIR, batch_file)
        batch_id = os.path.splitext(batch_file)[0]

        drift_score = compute_drift(batch_path)
        f1 = evaluate_batch(batch_path)

        retrained = "N"
        retrain_reason = "-"
        new_model_version = "-"

        # retrain trigger logic
        if drift_score > DRIFT_THRESHOLD or f1 < F1_THRESHOLD:
            retrained = "Y"
            retrain_reason = "drift" if drift_score > DRIFT_THRESHOLD else "performance"

            # bump version correctly
            minor_version = int(MODEL_VERSION.split(".")[1])
            new_model_version = f"v1.{minor_version + 1}.0"

            print(f"Retraining triggered for {batch_id} → new version: {new_model_version}")

            # run retraining with version argument
            subprocess.run(["python", "-m", "pipelines.retrain", new_model_version])

            # update registry
            save_active_version(new_model_version)
            MODEL_VERSION = new_model_version

        # log
        log_lifecycle(
            batch_id=batch_id,
            drift_score=drift_score,
            f1_score=f1,
            model_version=MODEL_VERSION,
            retrained=retrained,
            retrain_reason=retrain_reason,
            new_model_version=new_model_version
        )

        print(f"{batch_id}: drift={drift_score:.3f}, f1={f1:.3f}, retrained={retrained}")


if __name__ == "__main__":
    lifecycle_run()
