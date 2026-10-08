#!/usr/bin/env python3
"""
Hugging Face Transformer Training Pipeline for Sequential Reciprocal Matching.
Trains a Transformer Cross-Encoder on profile text representations and interaction feedback.
Evaluates accuracy, AUC, and downstream simulator performance.
"""

import os
import json
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_member_profile_text(member):
    """Formats structured member attributes into descriptive profile text."""
    fields = member.get("fields", {})
    age = member.get("age", 30)
    gender = member.get("gender", "unknown")
    zone = member.get("zone", "zone_a")
    
    seeking = ", ".join(fields.get("who_to_meet", ["any"])) if isinstance(fields.get("who_to_meet"), list) else fields.get("who_to_meet", "any")
    goal = fields.get("relationship_goal", "unknown")
    pace = fields.get("relationship_pace", "unknown")
    lifestyle = fields.get("lifestyle", "unknown")
    conv = fields.get("conversations", "unknown")
    smoking = fields.get("smoking", "no")
    kids = fields.get("has_children", False)
    wants_kids = fields.get("wants_children", "unknown")
    schedule = ", ".join(fields.get("schedule", [])) if isinstance(fields.get("schedule"), list) else "flexible"
    
    text = (
        f"Member Profile: {gender}, age {age}, residing in {zone}. "
        f"Seeking: {seeking} between {fields.get('age_min', 20)} and {fields.get('age_max', 50)}. "
        f"Relationship Goal: {goal}, Pace: {pace}. "
        f"Lifestyle: {lifestyle}, Conversation style: {conv}. "
        f"Smoking: {smoking}, Has kids: {kids}, Wants kids: {wants_kids}. "
        f"Availability schedule: {schedule}."
    )
    return text

class MatchingPairDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_a, text_b, label, meta = self.samples[idx]
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_data_and_create_samples():
    print("=" * 70)
    print("1. LOADING ALL PUBLIC POOLS AND BUILDING HUGGING FACE TEXT DATASET")
    print("=" * 70)
    
    members_map = {}
    for p in range(1, 11):
        p_str = f"public_{p:02d}"
        p_path = Path(f"data/{p_str}/members.jsonl")
        if p_path.exists():
            with open(p_path) as f:
                for line in f:
                    m = json.loads(line)
                    members_map[m["member_id"]] = m
    print(f"Total Member Profiles Loaded: {len(members_map):,}")

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
                    
    print(f"Total Introductions: {len(intros_map):,}, Total Feedback Records: {len(all_feedbacks):,}")

    samples = []
    for fb in all_feedbacks:
        i_id = fb["introduction_id"]
        if i_id not in intros_map or fb["value"] is None:
            continue
        intro = intros_map[i_id]
        u_id = fb["member_id"]
        partner_id = intro["user_b"] if intro["user_a"] == u_id else intro["user_a"]
        
        if u_id in members_map and partner_id in members_map:
            text_a = format_member_profile_text(members_map[u_id])
            text_b = format_member_profile_text(members_map[partner_id])
            label = 1 if fb["value"] == "yes" else 0
            samples.append((text_a, text_b, label, {"user_a": u_id, "user_b": partner_id, "event": fb["event"]}))
            
    print(f"Constructed {len(samples)} reciprocal pair interaction samples.")
    pos = sum(1 for _, _, y, _ in samples if y == 1)
    neg = len(samples) - pos
    print(f"  Positive Class (Yes): {pos} ({pos/len(samples)*100:.1f}%)")
    print(f"  Negative Class (No):  {neg} ({neg/len(samples)*100:.1f}%)")
    return samples, members_map

def train_huggingface_model():
    set_seed(42)
    samples, members_map = load_data_and_create_samples()
    
    # Train / Validation / Test Split (80% Train, 20% Test)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    print(f"\nDataset Split: {len(train_samples)} Train, {len(test_samples)} Test")

    # Load Model & Tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nInitializing Hugging Face Transformer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_dataset = MatchingPairDataset(train_samples, tokenizer)
    test_dataset = MatchingPairDataset(test_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on Compute Device: {device}")
    model.to(device)

    # Optimizer & Scheduler
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print("2. FINE-TUNING HUGGING FACE TRANSFORMER (CROSS-ENCODER)")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training Cross-Entropy Loss: {avg_loss:.4f}")

    # Evaluation on Test Set
    print("\n" + "=" * 70)
    print("3. EVALUATION ON HELD-OUT TEST DATASET")
    print("=" * 70)
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    bce_loss = log_loss(all_targets, all_probs)

    print(f"Test Accuracy:  {acc*100:.2f}%")
    print(f"Test Precision: {p*100:.2f}%")
    print(f"Test Recall:    {r*100:.2f}%")
    print(f"Test F1-Score:  {f1*100:.2f}%")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Test Log-Loss:  {bce_loss:.4f}")

    # Save Model Weights & Artifacts
    output_dir = Path("hf_matching_model")
    output_dir.mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = {
        "model_architecture": "all-MiniLM-L6-v2 Cross-Encoder",
        "training_epochs": epochs,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "test_accuracy": acc,
        "test_precision": p,
        "test_recall": r,
        "test_f1": f1,
        "test_roc_auc": auc,
        "test_log_loss": bce_loss
    }
    with open("hf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel and Tokenizer successfully saved to: {output_dir}")
    print("Metrics saved to: hf_metrics.json")
    return metrics

if __name__ == "__main__":
    train_huggingface_model()
