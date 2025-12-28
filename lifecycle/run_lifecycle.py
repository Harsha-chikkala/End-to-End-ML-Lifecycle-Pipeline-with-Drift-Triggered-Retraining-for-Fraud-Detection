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
    """Return current active model version, default to v1.0.0 if registry missing."""
    if not os.path.exists(REGISTRY_FILE):
        return "v1.0.0"
    with open(REGISTRY_FILE, "r") as f:
        return f.read().strip()


def save_active_version(version):
    """Save model version to registry.txt."""
    with open(REGISTRY_FILE, "w") as f:
        f.write(version)


def lifecycle_run():
    MODEL_VERSION = load_active_version()

    for batch_file in sorted(os.listdir(BATCH_DIR)):
        if not batch_file.endswith(".csv"):
            continue

        batch_path = os.path.join(BATCH_DIR, batch_file)
        batch_id = os.path.splitext(batch_file)[0]

        # compute drift + f1
        drift_score = compute_drift(batch_path)
        f1 = evaluate_batch(batch_path)

        retrained = "N"
        retrain_reason = "-"
        new_model_version = "-"

        # retrain logic
        if drift_score > DRIFT_THRESHOLD or f1 < F1_THRESHOLD:
            retrained = "Y"
            retrain_reason = "drift" if drift_score > DRIFT_THRESHOLD else "performance"

            # bump version once
            current_version_number = int(MODEL_VERSION.split(".")[2])
            next_version = f"v1.{current_version_number + 1}.0"
            new_model_version = next_version

            print(f"Retraining triggered for {batch_id} → new version: {new_model_version}")
            subprocess.run(["python", "-m", "pipelines.retrain"])  # runs retrain_model()

            # update registry
            save_active_version(new_model_version)
            MODEL_VERSION = new_model_version

        # log this batch
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
