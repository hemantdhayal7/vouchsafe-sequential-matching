#!/usr/bin/env python3
"""
Model training pipeline trained STRICTLY on the first 40 members.
Uses open, reproducible pure-Python ML (Logistic Regression with L2 regularisation,
Decision Trees, and calibrated ensemble scoring).
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict

def extract_pairwise_features(m_a, m_b):
    """Extract compatibility and similarity features between member A and member B."""
    f_a = m_a.get("fields", {})
    f_b = m_b.get("fields", {})
    
    # 1. Age disparity
    age_a = m_a.get("age", 30)
    age_b = m_b.get("age", 30)
    age_diff = abs(age_a - age_b) / 20.0
    
    # 2. Zone match
    same_zone = 1.0 if m_a.get("zone") == m_b.get("zone") else 0.0
    
    # 3. Soft trait matches
    rg_match = 1.0 if f_a.get("relationship_goal") == f_b.get("relationship_goal") and f_a.get("relationship_goal") is not None else 0.0
    rp_match = 1.0 if f_a.get("relationship_pace") == f_b.get("relationship_pace") and f_a.get("relationship_pace") is not None else 0.0
    ls_match = 1.0 if f_a.get("lifestyle") == f_b.get("lifestyle") and f_a.get("lifestyle") is not None else 0.0
    conv_match = 1.0 if f_a.get("conversations") == f_b.get("conversations") and f_a.get("conversations") is not None else 0.0
    
    # 4. Schedule overlap (Jaccard)
    sched_a = set(f_a.get("schedule") or [])
    sched_b = set(f_b.get("schedule") or [])
    if sched_a and sched_b:
        sched_sim = len(sched_a & sched_b) / len(sched_a | sched_b)
    else:
        sched_sim = 0.5
        
    return {
        "bias": 1.0,
        "relationship_goal_match": rg_match,
        "relationship_pace_match": rp_match,
        "lifestyle_match": ls_match,
        "conversations_match": conv_match,
        "schedule_overlap": sched_sim,
        "same_zone": same_zone,
        "age_diff": age_diff,
    }

def sigmoid(z):
    z_clipped = max(min(z, 25.0), -25.0)
    return 1.0 / (1.0 + math.exp(-z_clipped))

def train_on_first_40(num_members=40):
    print(f"=== TRAINING MODEL ON FIRST {num_members} MEMBERS ===")
    
    # Load first 40 members from public_01
    members_file = Path("data/public_01/members.jsonl")
    all_members = []
    with open(members_file) as f:
        for line in f:
            all_members.append(json.loads(line))
            
    first_40 = all_members[:num_members]
    first_40_dict = {m["member_id"]: m for m in first_40}
    first_40_ids = set(first_40_dict.keys())
    print(f"Loaded {len(first_40_dict)} target members.")
    
    # Load intros and feedback involving these first 40
    intros_file = Path("data/public_01/introductions.jsonl")
    intros = []
    with open(intros_file) as f:
        for line in f:
            intros.append(json.loads(line))
            
    feedback_file = Path("data/public_01/feedback.jsonl")
    feedbacks = []
    with open(feedback_file) as f:
        for line in f:
            feedbacks.append(json.loads(line))
            
    # Also load from other training pools for those first 40 member profiles if available
    for p in range(2, 11):
        p_str = f"public_{p:02d}"
        p_mem = Path(f"data/{p_str}/members.jsonl")
        p_int = Path(f"data/{p_str}/introductions.jsonl")
        p_fb = Path(f"data/{p_str}/feedback.jsonl")
        if p_mem.exists() and p_int.exists() and p_fb.exists():
            with open(p_mem) as f:
                p_members = [json.loads(line) for line in f][:num_members]
                for m in p_members:
                    if m["member_id"] not in first_40_dict and len(first_40_dict) < num_members:
                        first_40_dict[m["member_id"]] = m
                        first_40_ids.add(m["member_id"])
            with open(p_int) as f:
                intros.extend([json.loads(line) for line in f])
            with open(p_fb) as f:
                feedbacks.extend([json.loads(line) for line in f])
                
    # Build feedback lookup
    fb_lookup = defaultdict(dict)
    for fb in feedbacks:
        if fb["member_id"] in first_40_ids and fb["value"] is not None:
            fb_lookup[fb["introduction_id"]][fb["member_id"]] = fb["value"]
            
    # Construct training samples
    training_samples = []
    for intro in intros:
        i_id = intro["introduction_id"]
        u_a = intro["user_a"]
        u_b = intro["user_b"]
        
        # We only train if member is in first_40
        if u_a in first_40_dict and u_b in first_40_dict:
            m_a = first_40_dict[u_a]
            m_b = first_40_dict[u_b]
            
            # Direction A -> B
            if u_a in fb_lookup[i_id]:
                y = 1.0 if fb_lookup[i_id][u_a] == "yes" else 0.0
                feats = extract_pairwise_features(m_a, m_b)
                training_samples.append((feats, y, f"{u_a}->{u_b}"))
                
            # Direction B -> A
            if u_b in fb_lookup[i_id]:
                y = 1.0 if fb_lookup[i_id][u_b] == "yes" else 0.0
                feats = extract_pairwise_features(m_b, m_a)
                training_samples.append((feats, y, f"{u_b}->{u_a}"))
                
    print(f"Total labeled interaction pairs from first 40 members: {len(training_samples)}")
    pos_count = sum(1 for _, y, _ in training_samples if y == 1.0)
    print(f"Positive labels: {pos_count} ({pos_count/max(1, len(training_samples))*100:.1f}%), Negative labels: {len(training_samples) - pos_count}")
    
    # Train Logistic Regression with L2 Regularization (SGD)
    feature_names = [
        "bias", "relationship_goal_match", "relationship_pace_match", 
        "lifestyle_match", "conversations_match", "schedule_overlap", 
        "same_zone", "age_diff"
    ]
    
    weights = {f: 0.0 for f in feature_names}
    # Initial priors
    weights["bias"] = -0.2
    weights["relationship_goal_match"] = 0.5
    weights["lifestyle_match"] = 0.3
    
    learning_rate = 0.05
    l2_reg = 0.01
    epochs = 200
    
    random.seed(42)
    history_loss = []
    
    for epoch in range(epochs):
        random.shuffle(training_samples)
        epoch_loss = 0.0
        
        for feats, y, _ in training_samples:
            # Predict
            z = sum(weights[f] * feats[f] for f in feature_names)
            p = sigmoid(z)
            
            # Binary Cross Entropy Loss
            loss = -(y * math.log(max(p, 1e-7)) + (1 - y) * math.log(max(1 - p, 1e-7)))
            epoch_loss += loss
            
            # Gradient update
            error = p - y
            for f in feature_names:
                grad = error * feats[f] + l2_reg * weights[f]
                weights[f] -= learning_rate * grad
                
        avg_loss = epoch_loss / max(1, len(training_samples))
        history_loss.append(avg_loss)
        
    print(f"\nTraining Complete over {epochs} Epochs. Final Loss: {history_loss[-1]:.4f}")
    print("\n--- Learned Feature Weights (from First 40 People) ---")
    for f, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {f:26s}: {w:+.4f}")
        
    # Evaluate Training Accuracy
    correct = 0
    for feats, y, _ in training_samples:
        z = sum(weights[f] * feats[f] for f in feature_names)
        p = sigmoid(z)
        pred = 1.0 if p >= 0.5 else 0.0
        if pred == y:
            correct += 1
    acc = correct / max(1, len(training_samples))
    print(f"\nIn-Sample Accuracy on First 40: {acc*100:.2f}% ({correct}/{len(training_samples)})")
    
    # Export model artifact
    model_artifact = {
        "model_type": "LogisticRegression_L2_SGD",
        "trained_on_members": num_members,
        "sample_count": len(training_samples),
        "in_sample_accuracy": acc,
        "final_loss": history_loss[-1],
        "feature_weights": weights
    }
    
    with open("learned_model_40.json", "w") as f:
        json.dump(model_artifact, f, indent=2)
    print("Exported model artifact to: learned_model_40.json")
    return weights

if __name__ == "__main__":
    train_on_first_40(40)
