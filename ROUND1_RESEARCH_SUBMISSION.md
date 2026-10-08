# ROUND 1 - RESEARCH & IDEATION
## The One Introduction Problem: Sequential Reciprocal Matching Under Uncertainty

**Team:** Team Sequential Optimizer  
**Members:** Hemant Dhayal  
**Repository:** [https://github.com/hemantdhayal7/vouchsafe-sequential-matching](https://github.com/hemantdhayal7/vouchsafe-sequential-matching)  
**Spec:** Final participant specification v1.0.0 (5 Oct 2026)  
**Submission Deadline:** 9 October 2026, 23:59 IST (via Official Google Form)

> *Prepared from public repository specification and submission guidance. This document describes a research plan and decision-theoretic methodology for Round 1.*

---

### 1. Problem Understanding

We treat this as a **sequential decision problem under partial observability**, not a static ranking or unilateral recommendation problem.

At each decision step $t$, the policy observes only:
- Currently available members $\mathcal{M}_t$,
- Known preferences with missingness/declined states $\mathbf{x}_u^{\text{obs}}$,
- Past introduction records $\mathcal{H}_t$, and
- Feedback that has already become observable (acceptances, declines, dates, and post-date survey debriefs).

The system must decide:
- **Clarify:** What to ask given a strict daily budget $B_t = 12$ units.
- **Validate:** Which pairs are bidirectionally, reciprocally feasible.
- **Allocate / Wait:** Whether to introduce now or preserve member availability for higher future value.

**Core Research Question:** How to maximize high-quality mutual outcomes (Mutual Second-Meeting Intention, MSMI) while respecting hard reciprocal constraints, limited clarification budget, changing availability, and multi-week feedback delays?

---

### 2. Research Hypotheses

- **H1 (Global Allocation Beats Greedy):** Independent top-$k$ selection causes negative externalities by blocking multiple other viable pairs. Maximum-weight non-overlapping matching over the feasible graph increases total expected value per batch.
- **H2 (Selective Clarification Beats Indiscriminate):** Value of information (VOI) is uneven across the graph. Asking when the answer can change feasibility or resolve graph bottlenecks dominates uniform or fixed questionnaires.
- **H3 (Constraint-First Improves Validity):** Hard constraints are strict binary gates. A high soft affinity score must never override a known reciprocal violation.
- **H4 (Waiting Improves Long-Run Value):** Introducing a marginal pair today incurs an opportunity cost of locking both individuals for up to 20 days. An explicit wait action preserves option value for higher-affinity arrivals.

---

### 3. Proposed Decision Pipeline

```
                     ┌────────────────────────────────────────┐
                     │          OBSERVE STATE s_t             │
                     │  Available Members, History, Budget 12 │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │     IDENTIFY UNKNOWN / DECLINED        │
                     │  - Estimate Value of Information (VOI) │
                     │  - Query high-value bundles (<=12 pts) │
                     │  - Spatial co-clustering (sparse zones)│
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │          REFRESH STATE s_t'            │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │        GENERATE CANDIDATE PAIRS        │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │    RECIPROCAL HARD-CONSTRAINT FILTER   │
                     │  - Infeasible: remove with reason code │
                     │  - Unresolved: hold for queue          │
                     │  - Feasible: continue                  │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │       ESTIMATE PAIR VALUE V(A, B)      │
                     │   Calibrated feature attribution score │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │     GLOBAL NON-OVERLAPPING ALLOCATION  │
                     │  - Solve Max-Weight Matching           │
                     │  - Introduce selected disjoint pairs   │
                     │  - Leave weak pairs unmatched = WAIT   │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   OBSERVE DELAYED FEEDBACK -> UPDATE   │
                     └────────────────────────────────────────┘
```

---

### 4. Reciprocal Feasibility & Auditability

For pair $(A, B)$ to be eligible, both directions must pass:

$$\text{Eligible}(A, B) \iff \text{Feasible}_{A \to B} \land \text{Feasible}_{B \to A}$$

- **Known Hard Conflict (`blocked`):** Infeasible; must not be introduced regardless of soft score (e.g., age boundaries, gender/orientation mismatch, relationship structure conflict, smoking/children mismatch, zone incompatibility, duplicate pair, or outstanding active introduction).
- **Missing Decisive Info (`unknown`):** Not feasible yet, not infeasible. Routed to the clarification queue.
- **All Hard Checks Satisfied (`feasible`):** Candidate for valuation and matching.

**Implementation Standard:** The system maintains tri-state statuses `feasible | blocked | unknown` with structured reason codes for auditability and verification.

---

### 5. Pair Valuation

Transparent base valuation model:

$$V(A, B) = w_1 \cdot \text{goal\_match} + w_2 \cdot \text{pace\_match} + w_3 \cdot \text{lifestyle\_match} + w_4 \cdot \text{conversation\_match} + w_5 \cdot \text{space\_match} + w_6 \cdot \text{schedule\_fit} - \gamma \cdot \text{age\_diff} + \text{feedback\_terms}$$

- Weights are empirically tuned on historical interaction data with regularized estimation.
- In probabilistic modeling, directional and joint targets are strictly distinguished:
  - $P(A \text{ accepts } B)$ vs. $P(B \text{ accepts } A)$ vs. $P(\text{Mutual Acceptance})$ vs. $P(\text{MSMI} \mid \text{Date})$.

---

### 6. Global Allocation Strategy

We model available individuals as vertices $V_t$ and feasible pairs as edges $E_t$ weighted by compatibility score $V(A, B)$. Because each person can participate in at most one active introduction at any time:

$$\max_{\mathcal{P}_t \subseteq E_t} \sum_{(u, v) \in \mathcal{P}_t} V(u, v) \quad \text{subject to} \quad \text{deg}(u) \le 1 \quad \forall u \in V_t$$

This formulation directly resolves the global opportunity cost dilemma: proposing a slightly lower-ranked edge can unlock multiple mutually exclusive high-value edges across the pool, outperforming greedy FIFO assignment.

---

### 7. Clarification Strategy (Value of Information)

The 12-unit daily budget is treated as a scarce informational resource:

$$\text{Priority}(q) = \text{Expected Decision Improvement}(q) - \text{cost}(q)$$

where expected improvement measures:
1. Flipping the feasibility of high-potential candidate edges.
2. Resolving ranking ambiguity among top candidate pairs.
3. Unblocking bottleneck nodes with high demographic degree.

**Policy:** Hard-constraint bundles (cost 3) take precedence when unobserved hard traits stall candidate edges. Soft traits (cost 1) are queried when hard feasibility is established to differentiate competitive pairings. Declined/withheld responses are respected as informative signals without aggressive re-querying.

---

### 8. Waiting Policy

Waiting is modeled as an **explicit, value-preserving action**. A candidate pair is left unmatched if:

$$\mathbb{E}[V_{\text{now}}] < \tau \quad \lor \quad \mathbb{E}[V_{\text{future\_preserved}}] - \text{RiskOfLoss} > \mathbb{E}[V_{\text{now}}]$$

where $\tau$ is a calibrated feasibility and quality threshold. Waiting preserves participant availability for higher-affinity arrivals on subsequent days.

---

### 9. Planned Reference Baselines

1. **Greedy Baseline:** Feasible pairs sorted by $V(A, B)$ and introduced sequentially in FIFO order (tests **H1**).
2. **No-Clarification Baseline:** Never requests additional information ($B_t = 0$) and operates solely on initial profile observations (tests **H2**).
3. **Random-Feasible Baseline:** Random pairing among valid feasible edges, establishing the non-informative feasibility lower bound.

---

### 10. Planned Experiments & Evaluation Suite

1. **Greedy vs. Global Allocation:** Identical feasibility filter and scoring function with greedy allocator vs. maximum-weight bipartite matching.
2. **Clarification Ablation:** Selective VOI clarification vs. uniform random querying vs. zero clarification.
3. **Valuation Weight Ablation:** Uniform trait weights vs. regularized empirical feature weights.
4. **Waiting Threshold Ablation:** Always-match eager policy vs. calibrated wait thresholds.
5. **Scenario Robustness Grid:** Evaluated across all six official benchmark variants (`development`, `sparse`, `cold_start`, `delayed`, `shift`, `drift`) over independent pseudo-random seeds ($101, 102, 103$).

**Metrics:**
- **Primary:** Mutual Second-Meeting Intentions (MSMI) per 100 arrived members.
- **Secondary:** Distinct member coverage, mutual acceptance rate, mean first-introduction wait time, unserved member rate, and clarification cost per MSMI.

---

### 11. Failure Modes & Mitigation Regimes

- **Sparse Geographic Pools (12 Zones):** Low density causes edge fragmentation; mitigated via same-zone spatial co-clustering.
- **Withheld / Declined Attributes:** Handled gracefully by tri-state logic without negative imputation.
- **Delayed & Right-Censored Feedback:** Feedback separation between observed and pending states; horizon discounting for late-stage introductions.
- **Distribution Shift & Non-Stationarity:** Transparent constraint gating and robust agreement-based symmetric scoring.

---

### 12. Expected Contribution

A modular, constraint-first sequential matching framework integrating:
1. Strict, auditable reciprocal feasibility gating.
2. Value-of-Information (VOI) active clarification under strict daily budgets.
3. Calibrated pairwise compatibility valuation.
4. Global maximum-weight graph allocation.
5. Explicit opportunity-cost waiting.

---

### 13. Limitations & Ethical Boundary

- All personas, conversations, and interaction events in this evaluation are **synthetic**. Simulator outcomes test mathematical and operational decision dynamics, not genuine human emotional compatibility.
- Real-world deployment requires informed user consent, strict privacy safeguards, transparent auditability, and fairness guarantees.

---

### 14. Checklist Before Submission

- [x] **Team Name:** Team Sequential Optimizer
- [x] **Team Member:** Hemant Dhayal
- [x] **Repository URL:** `https://github.com/hemantdhayal7/vouchsafe-sequential-matching`
- [x] **Hypotheses Formulated:** H1 (Global vs Greedy), H2 (VOI Clarification), H3 (Constraint-First), H4 (Option Value Waiting)
- [x] **Feasibility Logic & Auditability:** Tri-state `feasible | blocked | unknown` with reason codes
- [x] **Allocation, Clarification & Waiting Formulations:** Fully specified
- [x] **Baselines & Experiments Defined:** Greedy, No-Asks, Random-Feasible across 6 scenario families
- [x] **Risks, Failure Cases & Limitations Documented**
- [x] **Format Compliant:** Ready for Markdown upload and PDF export
