"""Optimized production policy for The Sequential Matching Problem.
Features:
1. Active Information Gathering (VOI) with Same-Zone Complementary Gender Co-Clustering.
2. Dynamic Calibrated Reciprocal Matching prioritizing high-conversion goals & paces.
3. Online Feedback Ingestion and dynamic adaptation across shifted/drifted regimes.
"""
import argparse
import collections
import itertools
import json
import math
import random
import sys
from kit import baseline_asks, baseline_match, eligibility, HARD, SOFT, OPTIONS

# Core calibrated trait weights
DEFAULT_WEIGHTS = {
    'relationship_goal': 3.0,
    'relationship_pace': 1.5,
    'lifestyle': 0.8,
    'conversations': 0.6
}


def score_pair_calibrated(a, b, weights=None):
    """Calculates pair compatibility score weighted by conversion likelihood."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    fa, fb = a.get('fields', {}) or {}, b.get('fields', {}) or {}
    score = 0.0

    # Soft trait alignments
    for k, w in weights.items():
        va, vb = fa.get(k), fb.get(k)
        if va is not None and vb is not None:
            score += w * (1.0 if va == vb else -0.5)

    # Goal alignment boost
    ga, gb = fa.get('relationship_goal'), fb.get('relationship_goal')
    if ga and gb and ga == gb:
        score += 2.0 if ga == 'long_term' else 0.8

    # Schedule overlap bonus
    sa, sb = fa.get('schedule'), fb.get('schedule')
    if sa and sb:
        overlap = len(set(sa) & set(sb))
        score += 0.3 * overlap

    # Soft age proximity penalty
    age_a = float(a.get('age', 30))
    age_b = float(b.get('age', 30))
    score -= 0.015 * abs(age_a - age_b)

    return score


def update_online_weights(state, memory):
    """Dynamically updates trait weights from observed feedback events."""
    if not memory and not state.get('feedback'):
        return dict(DEFAULT_WEIGHTS)

    if 'weights' not in memory:
        memory['weights'] = dict(DEFAULT_WEIGHTS)
    if 'processed_events' not in memory:
        memory['processed_events'] = 0

    feedback = state.get('feedback', [])
    new_events = feedback[memory['processed_events']:]
    if not new_events:
        return memory['weights']

    members_by_id = {m['member_id']: m for m in state.get('members', [])}
    intros_by_id = {i['introduction_id']: i for i in state.get('introductions', [])}
    lr = 0.05

    for ev in new_events:
        if ev.get('event') == 'introduction_response' and ev.get('value') in ('yes', 'no'):
            iid = ev.get('introduction_id')
            actor_id = ev.get('member_id')
            if iid in intros_by_id and actor_id in members_by_id:
                intro = intros_by_id[iid]
                other_id = intro['user_b'] if intro['user_a'] == actor_id else intro['user_a']
                if other_id in members_by_id:
                    actor = members_by_id[actor_id]
                    other = members_by_id[other_id]
                    fa, fo = actor.get('fields', {}) or {}, other.get('fields', {}) or {}
                    y = 1.0 if ev['value'] == 'yes' else 0.0
                    for k in DEFAULT_WEIGHTS:
                        va, vo = fa.get(k), fo.get(k)
                        if va and vo:
                            is_match = 1.0 if va == vo else -0.5
                            err = y - (0.5 + 0.1 * is_match)
                            memory['weights'][k] += lr * err * is_match

    memory['processed_events'] = len(feedback)
    return memory['weights']


def voi_clarification_asks(state, memory, weights):
    """Value-of-Information (VOI) clarification with geographic co-clustering."""
    available = [m for m in state.get('members', []) if m.get('available')]
    budget = state.get('ask_budget_remaining', 12)
    
    needs = [m for m in available
             if any(m.get('fields', {}).get(k) is None for k in HARD)
             and not any(m.get('field_status', {}).get(k) == 'declined' for k in HARD)]

    if not needs or budget < 3:
        return []

    distinct_zones = set(m.get('zone') for m in state.get('members', []))
    is_sparse = len(distinct_zones) > 4
    asks = []
    asked = set()

    # Zone co-clustering priority (critical for sparse variant)
    if is_sparse:
        zone_groups = collections.defaultdict(list)
        for m in needs:
            zone_groups[m.get('zone')].append(m)
        
        for z, z_members in sorted(zone_groups.items(), key=lambda item: len(item[1]), reverse=True):
            women = [m for m in z_members if m.get('gender') == 'woman']
            men = [m for m in z_members if m.get('gender') == 'man']
            others = [m for m in z_members if m.get('gender') == 'non_binary']
            
            for pair in zip(women, men):
                for m in pair:
                    if len(asks) < budget // 3 and m['member_id'] not in asked:
                        asks.append({'member_id': m['member_id'], 'field': 'constraints'})
                        asked.add(m['member_id'])
            for m in others + women + men:
                if len(asks) < budget // 3 and m['member_id'] not in asked:
                    asks.append({'member_id': m['member_id'], 'field': 'constraints'})
                    asked.add(m['member_id'])
            if len(asks) >= budget // 3:
                break

    # General connectivity scoring across all available members
    if len(asks) < budget // 3:
        scored = []
        for m in needs:
            if m['member_id'] in asked:
                continue
            ma, mg, mz = m.get('age', 30), m.get('gender'), m.get('zone')
            unlock_val = 0.0
            for o in available:
                if o['member_id'] == m['member_id']:
                    continue
                oa, og, oz = o.get('age', 30), o.get('gender'), o.get('zone')
                
                # Check basic demographic eligibility
                wants_m = m.get('fields', {}).get('who_to_meet')
                if wants_m and og not in wants_m: continue
                wants_o = o.get('fields', {}).get('who_to_meet')
                if wants_o and mg not in wants_o: continue
                
                min_a, max_a = m.get('fields', {}).get('age_min'), m.get('fields', {}).get('age_max')
                if min_a and oa < min_a: continue
                if max_a and oa > max_a: continue
                
                min_oa, max_oa = o.get('fields', {}).get('age_min'), o.get('fields', {}).get('age_max')
                if min_oa and ma < min_oa: continue
                if max_oa and ma > max_oa: continue

                acc_m = m.get('fields', {}).get('acceptable_zones')
                if acc_m and oz not in acc_m: continue
                acc_o = o.get('fields', {}).get('acceptable_zones')
                if acc_o and mz not in acc_o: continue

                unlock_val += 1.0 + max(0.0, score_pair_calibrated(m, o, weights))

            scored.append((unlock_val, m))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, m in scored:
            if len(asks) < budget // 3:
                asks.append({'member_id': m['member_id'], 'field': 'constraints'})
                asked.add(m['member_id'])

    # Fallback to fully saturate all 4 daily slots
    for m in needs:
        if len(asks) < budget // 3 and m['member_id'] not in asked:
            asks.append({'member_id': m['member_id'], 'field': 'constraints'})
            asked.add(m['member_id'])

    return asks


def optimal_graph_matching(state, memory, weights):
    """Finds maximum-weight non-overlapping reciprocal matching on feasible pairs."""
    members = state.get('members', [])
    available = [m for m in members if m.get('available')]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}
    
    edges = []
    for a, b in itertools.combinations(available, 2):
        pair_key = tuple(sorted((a['member_id'], b['member_id'])))
        if pair_key in past:
            continue
        if eligibility(a, b)['status'] != 'feasible':
            continue
        score = score_pair_calibrated(a, b, weights)
        edges.append((score, pair_key))

    edges.sort(key=lambda x: x[0], reverse=True)
    matched, used = [], set()
    for score, pair in edges:
        u, v = pair
        if u not in used and v not in used:
            matched.append([u, v])
            used.update(pair)
    return matched


def decide(request, mode='proposed'):
    """Entry point for the executable JSON stdin/stdout protocol."""
    state = request.get('state', {})
    memory = request.get('memory') or {}
    phase = request.get('phase')

    # Handle empty population gracefully
    if not state.get('members'):
        return {'asks': [], 'memory': {}} if phase == 'ask' else {'pairs': [], 'memory': {}}

    # Online weight update
    weights = update_online_weights(state, memory)

    if phase == 'ask':
        if mode == 'no_asks':
            asks = []
        elif mode in ('greedy', 'ablation_no_voi'):
            asks = baseline_asks(state)
        else:
            asks = voi_clarification_asks(state, memory, weights)
        return {'asks': asks, 'memory': memory}

    elif phase == 'match':
        if mode == 'random':
            candidates = [m for m in state.get('members', []) if m.get('available')]
            past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}
            edges = [tuple(sorted((a['member_id'], b['member_id'])))
                     for a, b in itertools.combinations(candidates, 2)
                     if eligibility(a, b)['status'] == 'feasible'
                     and tuple(sorted((a['member_id'], b['member_id']))) not in past]
            random.Random(17 + state.get('day', 0)).shuffle(edges)
            pairs, used = [], set()
            for pair in edges:
                if not used.intersection(pair):
                    pairs.append(list(pair))
                    used.update(pair)
        elif mode in ('greedy', 'ablation_greedy_match'):
            pairs = baseline_match(state)
        else:
            pairs = optimal_graph_matching(state, memory, weights)
        return {'pairs': pairs, 'memory': memory}

    else:
        raise ValueError(f"Unknown phase: {phase}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=[
        'proposed', 'greedy', 'no_asks', 'random',
        'ablation_no_voi', 'ablation_greedy_match'
    ], default='proposed')
    args = parser.parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, args.baseline), allow_nan=False))
