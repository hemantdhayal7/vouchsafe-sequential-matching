#!/usr/bin/env python3
"""Offline machine learning training pipeline using Python standard library.
Trains calibrated pairwise outcome likelihood estimators on training pools (public_01 to public_06).
"""
import csv
import json
import math
import random
from pathlib import Path


def load_dataset():
    """Loads all members, introductions, and feedback events across pools."""
    members_by_id = {}
    with open('data_csv/all_members.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            members_by_id[row['member_id']] = row

    introductions = []
    with open('data_csv/all_introductions.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            introductions.append(row)

    feedback_by_intro = {}
    with open('data_csv/all_feedback.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            intro_id = row['introduction_id']
            feedback_by_intro.setdefault(intro_id, []).append(row)

    return members_by_id, introductions, feedback_by_intro


def extract_pair_features(user_a, user_b):
    """Extracts numerical and indicator features for a directional introduction (user_a -> user_b)."""
    feat = {}
    # Soft trait comparisons
    soft_traits = [
        'relationship_goal', 'relationship_pace', 'lifestyle', 'conversations',
        'emotional_availability', 'space_for_relationship', 'relocate'
    ]
    for t in soft_traits:
        va = user_a.get(t)
        vb = user_b.get(t)
        if va and vb:
            feat[f'{t}_match'] = 1.0 if va == vb else -0.5
            feat[f'{t}_missing'] = 0.0
        elif va or vb:
            feat[f'{t}_match'] = 0.0
            feat[f'{t}_missing'] = 0.5
        else:
            feat[f'{t}_match'] = 0.0
            feat[f'{t}_missing'] = 1.0

    # Specific goal interaction
    ga, gb = user_a.get('relationship_goal'), user_b.get('relationship_goal')
    feat['both_long_term'] = 1.0 if ga == 'long_term' and gb == 'long_term' else 0.0
    feat['both_exploring'] = 1.0 if ga == 'exploring' and gb == 'exploring' else 0.0

    # Demographics
    try:
        age_a = float(user_a.get('age', 30))
        age_b = float(user_b.get('age', 30))
        feat['age_diff'] = abs(age_a - age_b) / 10.0
    except ValueError:
        feat['age_diff'] = 0.5

    # Schedule overlap
    sa = set(user_a.get('schedule', '').split(';')) if user_a.get('schedule') else set()
    sb = set(user_b.get('schedule', '').split(';')) if user_b.get('schedule') else set()
    feat['sched_overlap'] = float(len(sa & sb)) if sa and sb else 0.0

    # Zone match
    feat['same_zone'] = 1.0 if user_a.get('zone') == user_b.get('zone') else 0.0

    return feat


def prepare_samples():
    members, intros, feedback = load_dataset()
    train_samples = []
    val_samples = []

    for intro in intros:
        split = intro['split']
        iid = intro['introduction_id']
        events = feedback.get(iid, [])
        ua_id, ub_id = intro['user_a'], intro['user_b']
        
        if ua_id not in members or ub_id not in members:
            continue
        
        user_a = members[ua_id]
        user_b = members[ub_id]

        # Extract directional responses
        for ev in events:
            if ev['event'] == 'introduction_response' and ev['value'] in ('yes', 'no'):
                actor_id = ev['member_id']
                label = 1.0 if ev['value'] == 'yes' else 0.0
                
                if actor_id == ua_id:
                    x = extract_pair_features(user_a, user_b)
                else:
                    x = extract_pair_features(user_b, user_a)
                
                sample = (x, label)
                if split == 'train':
                    train_samples.append(sample)
                elif split == 'validation':
                    val_samples.append(sample)

    return train_samples, val_samples


def sigmoid(z):
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic_regression(train_data, val_data, epochs=150, lr=0.08, l2=0.005):
    """Trains regularized logistic regression via SGD with validation monitoring."""
    feature_names = sorted(train_data[0][0].keys())
    weights = {k: 0.0 for k in feature_names}
    bias = 0.0

    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples across {len(feature_names)} features...")

    for epoch in range(1, epochs + 1):
        random.shuffle(train_data)
        for x, y in train_data:
            z = bias + sum(weights[k] * x.get(k, 0.0) for k in feature_names)
            p = sigmoid(z)
            err = p - y
            # Gradient update with L2 penalty
            bias -= lr * err
            for k in feature_names:
                weights[k] -= lr * (err * x.get(k, 0.0) + l2 * weights[k])

        if epoch % 30 == 0 or epoch == epochs:
            train_loss = -sum(y * math.log(max(1e-7, sigmoid(bias + sum(weights[k] * x[k] for k in feature_names)))) +
                              (1 - y) * math.log(max(1e-7, 1 - sigmoid(bias + sum(weights[k] * x[k] for k in feature_names))))
                              for x, y in train_data) / len(train_data)
            
            val_loss = -sum(y * math.log(max(1e-7, sigmoid(bias + sum(weights[k] * x[k] for k in feature_names)))) +
                            (1 - y) * math.log(max(1e-7, 1 - sigmoid(bias + sum(weights[k] * x[k] for k in feature_names))))
                            for x, y in val_data) / len(val_data)
            
            val_acc = sum((sigmoid(bias + sum(weights[k] * x[k] for k in feature_names)) >= 0.5) == y for x, y in val_data) / len(val_data)
            print(f"Epoch {epoch:>3} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.1%}")

    return weights, bias


if __name__ == '__main__':
    train_data, val_data = prepare_samples()
    weights, bias = train_logistic_regression(train_data, val_data)
    
    print("\n=== LEARNED FEATURE WEIGHTS ===")
    print(f"Bias: {bias:.4f}")
    for k, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {k:<25}: {w:>+.4f}")

    # Export learned model to json
    model_export = {
        'model_type': 'logistic_regression',
        'bias': bias,
        'weights': weights,
        'features': sorted(weights.keys())
    }
    with open('learned_model.json', 'w') as f:
        json.dump(model_export, f, indent=2)
    print("\nModel saved to learned_model.json!")
