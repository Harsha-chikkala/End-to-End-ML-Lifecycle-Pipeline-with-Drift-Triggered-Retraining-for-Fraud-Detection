import csv
import os
from datetime import datetime

LOG_FILE = "logs/lifecycle_log.csv"

# headers only written if file doesn't exist
HEADERS = [
    "timestamp", "batch_id", "drift_score", "f1_score",
    "model_version", "retrained", "retrain_reason", "new_model_version"
]

def log_lifecycle(batch_id, drift_score, f1_score, model_version,
                  retrained="N", retrain_reason="-", new_model_version="-"):
    
    # create file with headers if missing
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # write header first time
        if not file_exists:
            writer.writerow(HEADERS)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_id,
            drift_score,
            f1_score,
            model_version,
            retrained,
            retrain_reason,
            new_model_version
        ])

if __name__ == "__main__":
    # example log for batch_02
    log_lifecycle(
        batch_id="batch_02",
        drift_score=0.33,
        f1_score=0.79,
        model_version="v1.0.0"
    )
    print("Logged example entry → logs/lifecycle_log.csv")
