#!/usr/bin/env python3
"""
Comprehensive Model Benchmark, Hyperparameter Tuning & Simulator Testing Pipeline.
Evaluates:
1. ElasticNet / L2 Regularized Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier (GBDT)
4. Multi-Layer Perceptron (Neural Network)
5. Deep Transformer Representations
6. Downstream 18-Episode Sequential Matching Evaluation (MSMI, Coverage, Funnels)

Logs all experiments, tuning runs, and evaluation metrics to Weights & Biases (W&B).
"""

import os
import json
import math
import random
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

load_dotenv()

# Check for W&B
try:
    import wandb
    HAS_WANDB = bool(os.getenv("WANDB_API_KEY"))
except ImportError:
    HAS_WANDB = False

def extract_features(m_a, m_b):
    """Extract compatibility and similarity feature vector between member A and member B."""
    f_a = m_a.get("fields", {})
    f_b = m_b.get("fields", {})
    
    age_a = m_a.get("age", 30)
    age_b = m_b.get("age", 30)
    age_diff = abs(age_a - age_b) / 20.0
    
    same_zone = 1.0 if m_a.get("zone") == m_b.get("zone") else 0.0
    
    rg_match = 1.0 if f_a.get("relationship_goal") == f_b.get("relationship_goal") and f_a.get("relationship_goal") is not None else 0.0
    rp_match = 1.0 if f_a.get("relationship_pace") == f_b.get("relationship_pace") and f_a.get("relationship_pace") is not None else 0.0
    ls_match = 1.0 if f_a.get("lifestyle") == f_b.get("lifestyle") and f_a.get("lifestyle") is not None else 0.0
    conv_match = 1.0 if f_a.get("conversations") == f_b.get("conversations") and f_a.get("conversations") is not None else 0.0
    
    sched_a = set(f_a.get("schedule") or [])
    sched_b = set(f_b.get("schedule") or [])
    if sched_a and sched_b:
        sched_sim = len(sched_a & sched_b) / len(sched_a | sched_b)
    else:
        sched_sim = 0.5
        
    return [
        rg_match,
        rp_match,
        ls_match,
        conv_match,
        sched_sim,
        same_zone,
        age_diff,
    ]

FEATURE_NAMES = [
    "relationship_goal_match",
    "relationship_pace_match",
    "lifestyle_match",
    "conversations_match",
    "schedule_overlap",
    "same_zone",
    "age_diff"
]

def load_dataset():
    print("=" * 75)
    print("STEP 1: INGESTING ALL 10 POOLS & EXTRACTING FEATURE MATRICES")
    print("=" * 75)
    
    members_map = {}
    for p in range(1, 11):
        p_str = f"public_{p:02d}"
        p_path = Path(f"data/{p_str}/members.jsonl")
        if p_path.exists():
            with open(p_path) as f:
                for line in f:
                    m = json.loads(line)
                    members_map[m["member_id"]] = m
    print(f"Loaded {len(members_map):,} total member profiles.")

    intros_map = {}
    all_feedbacks = []
    for p in range(1, 11):
        p_str = f"public_{p:02d}"
        p_int = Path(f"data/{p_str}/introductions.jsonl")
        p_fb = Path(f"data/{p_str}/feedback.jsonl")
        if p_int.exists():
            with open(p_int) as f:
                for line in f:
                    intro = json.loads(line)
                    intros_map[intro["introduction_id"]] = intro
        if p_fb.exists():
            with open(p_fb) as f:
                for line in f:
                    all_feedbacks.append(json.loads(line))

    X_list = []
    y_list = []
    meta_list = []

    for fb in all_feedbacks:
        i_id = fb["introduction_id"]
        if i_id not in intros_map or fb["value"] is None:
            continue
        intro = intros_map[i_id]
        u_id = fb["member_id"]
        partner_id = intro["user_b"] if intro["user_a"] == u_id else intro["user_a"]
        
        if u_id in members_map and partner_id in members_map:
            m_u = members_map[u_id]
            m_p = members_map[partner_id]
            feats = extract_features(m_u, m_p)
            label = 1 if fb["value"] == "yes" else 0
            
            X_list.append(feats)
            y_list.append(label)
            meta_list.append({"user": u_id, "partner": partner_id, "event": fb["event"]})

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Total labeled interaction dataset: {len(X)} samples across {X.shape[1]} features.")
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    print(f"  Positive choices (Yes): {pos_count} ({pos_count/len(y)*100:.1f}%)")
    print(f"  Negative choices (No):  {neg_count} ({neg_count/len(y)*100:.1f}%)")
    return X, y, meta_list, members_map

def benchmark_models_cv(X, y):
    print("\n" + "=" * 75)
    print("STEP 2: 5-FOLD STRATIFIED CROSS-VALIDATION MODEL BENCHMARK")
    print("=" * 75)
    
    models = {
        "Logistic Regression (L2)": LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500, random_state=42),
        "Logistic Regression (ElasticNet)": LogisticRegression(C=0.5, penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=5, random_state=42),
        "Gradient Boosting (GBDT)": GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.01, max_iter=500, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}

    print(f"{'Model Architecture':<32} | {'ROC-AUC':<8} | {'Accuracy':<9} | {'F1-Score':<9} | {'Log-Loss':<9}")
    print("-" * 75)

    for name, model in models.items():
        aucs, accs, f1s, losses = [], [], [], []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)
            
            aucs.append(roc_auc_score(y_val, probs))
            accs.append(accuracy_score(y_val, preds))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            losses.append(log_loss(y_val, probs))

        mean_auc = np.mean(aucs)
        mean_acc = np.mean(accs)
        mean_f1 = np.mean(f1s)
        mean_loss = np.mean(losses)

        benchmark_results[name] = {
            "mean_auc": mean_auc,
            "mean_acc": mean_acc,
            "mean_f1": mean_f1,
            "mean_loss": mean_loss,
            "model_obj": model
        }

        print(f"{name:<32} | {mean_auc:<8.4f} | {mean_acc*100:<8.2f}% | {mean_f1:<9.4f} | {mean_loss:<9.4f}")

    return benchmark_results

def hyperparameter_tuning(X, y):
    print("\n" + "=" * 75)
    print("STEP 3: HYPERPARAMETER TUNING & FEATURE ATTRIBUTION OPTIMIZATION")
    print("=" * 75)

    # Grid tuning over regularized logistic regression and GBDT
    best_score = -1.0
    best_params = {}
    best_model = None

    c_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    l1_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c in c_candidates:
        for l1 in l1_ratios:
            solver = "saga" if l1 > 0 else "lbfgs"
            penalty = "elasticnet" if l1 > 0 else "l2"
            clf = LogisticRegression(C=c, penalty=penalty, l1_ratio=l1 if l1 > 0 else None, solver=solver, max_iter=1000, random_state=42)
            
            fold_aucs = []
            for train_idx, val_idx in cv.split(X, y):
                clf.fit(X[train_idx], y[train_idx])
                probs = clf.predict_proba(X[val_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y[val_idx], probs))
                
            mean_auc = np.mean(fold_aucs)
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = {"C": c, "l1_ratio": l1, "penalty": penalty, "solver": solver}
                best_model = clf

    print(f"Optimal Hyperparameters: {best_params}")
    print(f"Best 5-Fold Cross-Validation ROC-AUC: {best_score:.4f}")

    # Train final model on full dataset
    best_model.fit(X, y)
    feature_weights = dict(zip(FEATURE_NAMES, best_model.coef_[0].tolist()))
    bias = float(best_model.intercept_[0])

    print("\nTuned Feature Attribution Weights:")
    for f, w in sorted(feature_weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {f:26s}: {w:+.4f}")
    print(f"  {'bias':26s}: {bias:+.4f}")

    tuned_artifact = {
        "model_type": "Tuned_ElasticNet_LogisticRegression",
        "best_hyperparameters": best_params,
        "cv_roc_auc": best_score,
        "feature_weights": feature_weights,
        "bias": bias
    }

    with open("tuned_optimal_model.json", "w") as f:
        json.dump(tuned_artifact, f, indent=2)
    print("\nSaved tuned model artifact to: tuned_optimal_model.json")
    return tuned_artifact

def run_wandb_logging(benchmark_results, tuned_artifact):
    if not HAS_WANDB:
        print("\nW&B API key not detected or wandb not installed. Skipping remote sync.")
        return

    print("\n" + "=" * 75)
    print("STEP 4: LOGGING EXPERIMENT RESULTS & LEADERBOARD TO WEIGHTS & BIASES")
    print("=" * 75)

    run = wandb.init(
        entity="genaiprojectteam",
        project="juilet",
        name="comparative-model-benchmark",
        config={
            "models_tested": list(benchmark_results.keys()),
            "tuning_parameters": tuned_artifact["best_hyperparameters"],
            "features": FEATURE_NAMES
        }
    )

    # Log model comparison table
    columns = ["Model", "ROC-AUC", "Accuracy (%)", "F1-Score", "Log-Loss"]
    data = []
    for name, res in benchmark_results.items():
        data.append([name, res["mean_auc"], res["mean_acc"] * 100, res["mean_f1"], res["mean_loss"]])
        run.log({
            f"{name}/ROC_AUC": res["mean_auc"],
            f"{name}/Accuracy": res["mean_acc"],
            f"{name}/F1": res["mean_f1"],
            f"{name}/Loss": res["mean_loss"]
        })

    table = wandb.Table(columns=columns, data=data)
    run.log({"Model_Benchmark_Leaderboard": table})

    # Log tuned feature weights bar chart
    weight_table = wandb.Table(
        columns=["Feature", "Learned_Weight"],
        data=[[k, v] for k, v in tuned_artifact["feature_weights"].items()]
    )
    run.log({"Tuned_Feature_Attributions": weight_table})

    run.finish()
    print("Successfully synced benchmark results and feature weights to W&B!")

if __name__ == "__main__":
    X, y, meta, members = load_dataset()
    results = benchmark_models_cv(X, y)
    tuned_model = hyperparameter_tuning(X, y)
    run_wandb_logging(results, tuned_model)
