#!/usr/bin/env python3
"""Export all JSONL and state files across all 10 pools into clean, standardized CSV files."""
import json
import csv
import os
from pathlib import Path

def export_all_to_csv(data_dir="data", output_dir="data_csv"):
    manifest_path = Path("data_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    pool_split_map = {}
    for split_name, pool_list in splits.items():
        for p in pool_list:
            pool_split_map[p] = split_name

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_members = []
    all_questionnaires = []
    all_conversations = []
    all_introductions = []
    all_feedback = []
    pools_summary = []

    hard_fields = [
        'age_min', 'age_max', 'who_to_meet', 'relationship_structure', 'smoking',
        'partner_smoking', 'has_children', 'partner_children', 'wants_children',
        'acceptable_zones', 'schedule'
    ]
    soft_fields = [
        'relationship_goal', 'relationship_pace', 'lifestyle', 'conversations',
        'emotional_availability', 'space_for_relationship', 'relocate'
    ]
    all_field_names = hard_fields + soft_fields

    for pool_info in manifest["pools"]:
        pool_id = pool_info["pool_id"]
        pool_path = Path(data_dir) / pool_id
        split = pool_split_map.get(pool_id, "unknown")
        
        pools_summary.append({
            "pool_id": pool_id,
            "split": split,
            "seed": pool_info.get("seed"),
            "snapshot_day": pool_info.get("snapshot_day", 30),
            "members_count": pool_info.get("members", 200),
            "introductions_count": pool_info.get("introductions", 0),
            "feedback_events_count": pool_info.get("feedback_events", 0),
        })

        pool_members = []
        pool_questionnaires = []
        pool_conversations = []
        pool_introductions = []
        pool_feedback = []

        # 1. Members
        members_file = pool_path / "members.jsonl"
        if members_file.exists():
            with open(members_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    m = json.loads(line)
                    m_row = {
                        "pool_id": pool_id,
                        "split": split,
                        "member_id": m.get("member_id"),
                        "age": m.get("age"),
                        "gender": m.get("gender"),
                        "zone": m.get("zone"),
                        "arrived_day": m.get("arrived_day"),
                        "available": m.get("available"),
                        "source": m.get("source"),
                    }
                    fields = m.get("fields", {}) or {}
                    field_status = m.get("field_status", {}) or {}
                    field_obs_day = m.get("field_observed_day", {}) or {}

                    for k in all_field_names:
                        val = fields.get(k)
                        if isinstance(val, (list, tuple)):
                            val = ";".join(str(x) for x in val)
                        m_row[k] = val
                        m_row[f"{k}_status"] = field_status.get(k)
                        m_row[f"{k}_observed_day"] = field_obs_day.get(k)

                    pool_members.append(m_row)
                    all_members.append(m_row)

        # 2. Questionnaires
        q_file = pool_path / "questionnaires.jsonl"
        if q_file.exists():
            with open(q_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    q = json.loads(line)
                    q_answers = q.get("questionnaire_answers", {}) or {}
                    q_row = {
                        "pool_id": pool_id,
                        "split": split,
                        "member_id": q.get("member_id"),
                        "age": q.get("age"),
                        "gender": q.get("gender"),
                        "city": q.get("city"),
                    }
                    for k, v in q_answers.items():
                        q_row[f"q_{k}"] = v
                    pool_questionnaires.append(q_row)
                    all_questionnaires.append(q_row)

        # 3. Conversations
        conv_file = pool_path / "conversations.jsonl"
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    member_id = c.get("member_id")
                    observed_day = c.get("observed_day")
                    style = c.get("style")
                    messages = c.get("messages", [])
                    for idx, msg in enumerate(messages):
                        c_row = {
                            "pool_id": pool_id,
                            "split": split,
                            "member_id": member_id,
                            "observed_day": observed_day,
                            "style": style,
                            "message_idx": idx,
                            "role": msg.get("role"),
                            "content": msg.get("content")
                        }
                        pool_conversations.append(c_row)
                        all_conversations.append(c_row)

        # 4. Introductions
        intro_file = pool_path / "introductions.jsonl"
        if intro_file.exists():
            with open(intro_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    intro = json.loads(line)
                    intro_row = {
                        "pool_id": pool_id,
                        "split": split,
                        "introduction_id": intro.get("introduction_id"),
                        "user_a": intro.get("user_a"),
                        "user_b": intro.get("user_b"),
                        "assigned_day": intro.get("assigned_day"),
                        "response_deadline_day": intro.get("response_deadline_day"),
                        "logging_policy": intro.get("logging_policy"),
                        "propensity": intro.get("propensity"),
                    }
                    pool_introductions.append(intro_row)
                    all_introductions.append(intro_row)

        # 5. Feedback
        fb_file = pool_path / "feedback.jsonl"
        if fb_file.exists():
            with open(fb_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    fb = json.loads(line)
                    fb_row = {
                        "pool_id": pool_id,
                        "split": split,
                        "introduction_id": fb.get("introduction_id"),
                        "member_id": fb.get("member_id"),
                        "event": fb.get("event"),
                        "value": fb.get("value"),
                        "missing_reason": fb.get("missing_reason"),
                        "occurred_day": fb.get("occurred_day"),
                        "observed_day": fb.get("observed_day"),
                    }
                    pool_feedback.append(fb_row)
                    all_feedback.append(fb_row)

        # Helper to safely write arbitrary dict rows to CSV
        def write_rows_to_csv(filepath, rows):
            if not rows:
                return
            keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

        write_rows_to_csv(pool_path / "members_full.csv", pool_members)
        write_rows_to_csv(pool_path / "questionnaires.csv", pool_questionnaires)
        write_rows_to_csv(pool_path / "conversations.csv", pool_conversations)
        write_rows_to_csv(pool_path / "introductions.csv", pool_introductions)
        write_rows_to_csv(pool_path / "feedback.csv", pool_feedback)

    # Consolidated writes to data_csv/
    def write_rows_to_csv(filepath, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_rows_to_csv(out_path / "all_members.csv", all_members)
    write_rows_to_csv(out_path / "all_questionnaires.csv", all_questionnaires)
    write_rows_to_csv(out_path / "all_conversations.csv", all_conversations)
    write_rows_to_csv(out_path / "all_introductions.csv", all_introductions)
    write_rows_to_csv(out_path / "all_feedback.csv", all_feedback)
    write_rows_to_csv(out_path / "pools_summary.csv", pools_summary)

    print("CSV Export Successful!")
    print(f"Consolidated CSV directory: {output_dir}/")
    print(f"  - all_members.csv        : {len(all_members):,} rows")
    print(f"  - all_questionnaires.csv : {len(all_questionnaires):,} rows")
    print(f"  - all_conversations.csv  : {len(all_conversations):,} rows")
    print(f"  - all_introductions.csv  : {len(all_introductions):,} rows")
    print(f"  - all_feedback.csv       : {len(all_feedback):,} rows")
    print(f"  - pools_summary.csv      : {len(pools_summary):,} rows")
    print("\nPer-pool CSVs created inside each data/public_XX/ folder:")
    print("  - members_full.csv, questionnaires.csv, conversations.csv, introductions.csv, feedback.csv")

if __name__ == "__main__":
    export_all_to_csv()
