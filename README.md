# End-to-End ML Lifecycle Pipeline with Drift-Triggered Retraining for Fraud Detection

This repository implements a complete machine learning lifecycle system for fraud detection that monitors data drift and model performance over time. The system ingests sequential batches of data, evaluates model reliability, and triggers retraining only when thresholds are violated. Model versions and lifecycle decisions are logged for reproducibility and traceability.

---

## 1. Project Overview

Traditional ML models degrade as data changes. This project simulates a production-style workflow where new batch data is processed over time, and retraining occurs only when necessary. It demonstrates:

* batch ingestion and sequential processing
* preprocessing reuse to avoid preprocessing drift
* drift monitoring using statistical changes in distributions
* performance evaluation using delayed ground truth labels
* conditional retraining based on thresholds
* semantic model versioning
* lifecycle logging

---

## 2. Repository Structure

```
ml-fraud-lifecycle/
├── data/
│   ├── raw/                      # local dataset (ignored by git)
│   └── batches/                  # sequential time-based batches
├── models/
│   ├── artifacts/                # saved model and scaler versions
│   └── registry.txt              # tracks current active model version
├── pipelines/
│   ├── create_batches.py         # split dataset into batches
│   ├── train.py                  # initial model training with preprocessing
│   ├── retrain.py                # retraining logic
│   ├── evaluate_performance.py   # batch-wise performance evaluation
│   └── log_lifecycle.py          # append lifecycle decision logs
├── monitoring/
│   └── drift_simple.py           # simple drift calculation
├── lifecycle/
│   └── run_lifecycle.py          # end-to-end execution across batches
├── logs/
│   └── lifecycle_log.csv         # stored lifecycle results
└── README.md
```

---

## 3. Dataset

Credit Card Fraud Detection Dataset
(Anonymized PCA-transformed features, extremely imbalanced)

* ~284,807 samples
* binary target: fraud (1) vs non-fraud (0)
* class imbalance: ~0.17% fraud
* dataset stored locally under `data/raw/` and excluded from git

---

## 4. Drift Monitoring

Drift is calculated based on changes in feature distribution statistics:

```
drift_score = mean(|mean_ref - mean_batch|) + mean(|std_ref - std_batch|)
```

`mean_ref, std_ref` are computed from initial training data.
Retraining occurs if `drift_score > DRIFT_THRESHOLD` or `f1_score < F1_THRESHOLD`.

---

## 5. Model Versioning

Semantic versioning is used:

* v1.0.0: initial training
* v1.x.x: retraining with same preprocessing
* v2.0.0: preprocessing or pipeline change

`models/registry.txt` stores the current active version.

---

## 6. Execution Flow

```
Initial training → Ingest batch → Evaluate → Drift/performance check
→ Retrain if required → Save new version → Log lifecycle outcome
```

---

## 7. How to Run

Activate virtual environment:

```
source venv/bin/activate
```

Create sequential batches:

```
python -m pipelines.create_batches
```

Initial training:

```
python -m pipelines.train
```

Run lifecycle across batches:

```
python -m lifecycle.run_lifecycle
```

View lifecycle decisions:

```
cat logs/lifecycle_log.csv
```

---

## 8. Example Output

```
batch_01: drift=0.007, f1=0.958, retrained=N
batch_02: drift=0.237, f1=0.712, retrained=N
batch_03: drift=0.301, f1=0.749, retrained=N
batch_04: drift=0.598, f1=0.739, retrained=N
Retraining triggered for batch_05 → new version: v1.3.0
Retraining complete → saved new model version: v1.3.0
batch_05: drift=0.628, f1=0.718, retrained=Y
```

---

## 9. Lifecycle Log Example

```
timestamp,batch_id,drift_score,f1_score,model_version,retrained,retrain_reason,new_model_version
2025-12-29 16:17:56,batch_01,0.0065,0.9579,v1.0.0,N,-,-
2025-12-29 16:18:49,batch_05,0.6278,0.7179,v1.3.0,Y,drift,v1.3.0
```

---

## 10. Notes

* preprocessing is fitted during initial training and reused for all batches
* SMOTE is applied only during training, not inference
* raw dataset is intentionally not tracked in git

---
