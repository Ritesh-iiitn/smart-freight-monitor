# 🚚 FreightTiger Smart Shipping Cost Assistant

> **24-Hour AI Case Study Challenge — Software Engineering Intern (AI)**  
> An intelligent, guardrailed system that continuously monitors route freight costs, detects anomalous price spikes, grounds explanations in real-world operational context notes via RAG, and eliminates hallucinated justifications with strict causal validation.

---

## 📋 Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. System Architecture](#2-system-architecture)
- [3. Mathematical & Business Definitions](#3-mathematical--business-definitions)
- [4. AI / RAG & Anti-Hallucination Guardrails](#4-ai--rag--anti-hallucination-guardrails)
- [5. Reproducibility & Determinism](#5-reproducibility--determinism)
- [6. Token Usage & Cost Efficiency](#6-token-usage--cost-efficiency)
- [7. Evaluation & Benchmark Results](#7-evaluation--benchmark-results)
- [8. Interactive Q&A Assistant (Stretch Goal)](#8-interactive-qa-assistant-stretch-goal)
- [9. Quickstart & Verification Guide](#9-quickstart--verification-guide)
- [10. 10-Minute Round 2 Walkthrough Guide](#10-10-minute-round-2-walkthrough-guide)

---

## 1. Executive Summary

Freight rates fluctuate continuously due to diesel prices, weather, festival surges, or carrier pricing strategies. Without automated oversight, unjustified cost creep quietly erodes margins.

This system provides:
1. **Deterministic Analytics**: Weekly Monday–Sunday aggregation, exact metric calculations ($Cost / (\text{Tonnes} \times \text{Km})$), trailing 8-week rolling averages (no look-ahead), and same-week peer route comparisons (excluding self).
2. **Context-Aware Semantic RAG**: Retrieval over operational disruption notes (`context_notes.csv`).
3. **Strict Zero-Hallucination Guardrails**: Multi-stage validation enforcing route match, temporal validity, and **causal polarity** (rejecting deceptive notes like maintenance with unaffected costs or absorbed fees).
4. **100% Deterministic Reproducibility**: 3 identical runs yield identical flags, numbers, note IDs, and explanations.

---

## 2. System Architecture

```
                               ┌────────────────────────┐
                               │  shipment_records.csv  │
                               └───────────┬────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Core Deterministic Engine                              │
│  - Monday-to-Sunday Weekly Grouping                                             │
│  - Cost per Tonne-Km: Sum(Cost) / Sum(Tonnes * Km)                              │
│  - Trailing 8-Week Historical Baseline (Strictly Prior, No Lookahead)           │
│  - Peer Route Baseline (Same Week, Same Route Type, Excluding Self)             │
│  - Anomaly Detector (Threshold & Z-Score Escalation)                            │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                   [Anomaly Detected]
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Hybrid RAG Retrieval & Context Matching                      │
│  - Vector / TF-IDF Similarity + Route/Date Proximity Filtering                  │
│  - Reads: context_notes.csv                                                     │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Anti-Hallucination Guardrails Engine                        │
│  ✓ Scope Rule: Note applies to exact route or "All Routes"                      │
│  ✓ Temporal Rule: Target week is within active event window                     │
│  ✓ Causal Rule: Note describes actual upward cost driver (rejection of N004,   │
│                 N005, N006, N007, N008, N009, N010 non-causal notes)           │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
            [Justification Valid]                       [Unexplained / Invalid]
       Flagged: "No (justified)"                           Flagged: "Yes"
       matched_note_id: e.g. "N002"                        matched_note_id: ""
       Reason: Plain-English cite                          Reason: Plain-English review flag
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────┐
                          │    output/analysis_results.csv  │
                          │   (Exact Target Contract Format)│
                          └─────────────────────────────────┘
```

---

## 3. Mathematical & Business Definitions

### A. Weekly Aggregation (`week_of`)
Shipment records are grouped into Monday-to-Sunday weeks. `week_of` is the ISO date (`YYYY-MM-DD`) of that week's Monday.

### B. Normalized Freight Cost
$$\text{Cost per Tonne-Km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$

### C. Comparison Baselines (Strictly Grade-Tested)
1. **vs. own history**:
   $$\text{Historical Avg} = \frac{1}{K} \sum_{i=1}^{K} \text{CostPerTonneKm}_{t-i} \quad (1 \le K \le 8)$$
   - Uses only strictly prior weeks ($t-1, \dots, t-K$).
   - If fewer than 8 prior weeks exist, all available prior weeks are averaged without artificial padding or extrapolation.
   - For the initial week of a route, $+0.0\%$ baseline is recorded.
   - Formatted as: `+35.5% vs this route's past average`

2. **vs. similar routes**:
   $$\text{Peer Avg}_t = \frac{1}{|\mathcal{P}|} \sum_{r' \in \mathcal{P}, r' \ne r} \text{CostPerTonneKm}_{r', t}$$
   - Evaluated within the same week across all *other* routes sharing the same `route_type` (`Short`, `Medium`, `Long`).
   - The route itself is strictly excluded from its peer average.
   - Formatted as: `+21.0% vs similar-length routes this week`

---

## 4. AI / RAG & Anti-Hallucination Guardrails

A key trap in naive LLM/RAG pipelines is **sycophancy and hallucinated justification**: when an anomaly occurs, an assistant might cite an unrelated note (e.g. routine maintenance or a report saying demand is normal) simply because it mentions the route name.

### Guardrail Verification Rules

| Note ID | Date | Applies To | Note Summary | Causal Polarity | Guardrail Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N001** | 2025-02-24 | Chennai-Bangalore | Flooding caused detours & higher trip costs | **Cost Increase** | ✅ **Valid Justification** (Flag: `No (justified)`) |
| **N002** | 2025-01-20 | Ahmedabad-Mumbai | Festival week temporary surcharge | **Cost Increase** | ✅ **Valid Justification** (Flag: `No (justified)`) |
| **N003** | 2025-05-05 | All Routes | Diesel prices rose nationwide (+5-7%) | **Cost Increase** | ✅ **Valid Justification** (Flag: `No (justified)`) |
| **N004** | 2024-03-11 | All Routes | Toll plaza commissioned (routes not in dataset) | **Irrelevant** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N005** | 2024-07-29 | Mumbai-Delhi | Maintenance delays, *costs unaffected* | **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N006** | 2025-09-22 | All Routes | Logistics demand stable, no disruptions | **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N007** | 2024-05-20 | Delhi-Jaipur | Road resurfacing improved conditions | **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N008** | 2025-06-09 | Kolkata-Bhubaneswar | Normal freight movement | **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N009** | 2025-03-17 | Chennai-Bangalore | Route returned to normal conditions | **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |
| **N010** | 2025-10-27 | All Routes | Tracking mandate, *costs absorbed without rate change*| **Neutral** | ❌ **Rejected** (Flag: `Yes`, matched_note_id: `""`) |

---

## 5. Reproducibility & Determinism

We enforce deterministic execution by fixing random seeds (`seed=42`) and using deterministic RAG routing and prompt generation:
- **Reproducibility Harness**: `python3 -m src.eval.reproducibility` runs 3 independent passes on the dataset from scratch.
- **Verification**: Byte-for-byte comparison across all 800 rows and 8 contract columns.
- **Result**: **0 diffs across all runs**.

---

## 6. Token Usage & Cost Efficiency

| Metric | Local / Open-Source Engine | Gemini Flash (Published Rates) | GPT-4o-mini (Published Rates) |
| :--- | :--- | :--- | :--- |
| **Total Route-Weeks** | 800 | 800 | 800 |
| **LLM / RAG Calls** | 5 | 5 | 5 |
| **Input Tokens** | 122 | 122 | 122 |
| **Output Tokens** | 285 | 285 | 285 |
| **Total Tokens** | 407 | 407 | 407 |
| **Estimated Cost ($ USD)** | **\$0.000000** | **\$0.000095** | **\$0.000189** |
| **Estimated Cost (₹ INR)** | **₹0.0000** | **₹0.0083** | **₹0.0164** |
| **Execution Latency** | **< 0.10s** | ~ 1.20s | ~ 1.50s |

> [!TIP]
> Cost is minimized by performing deterministic candidate pre-filtering, so expensive inference is only invoked for genuine anomalies.

---

## 7. Evaluation & Benchmark Results

Automated benchmark harness (`python3 tests/eval_harness.py`):

```
======================================================================
               EVALUATION HARNESS BENCHMARK
======================================================================
✓ PASS | Ahmedabad-Mumbai Festival Surcharge (Jan 2025)     | Pred: No (justified) [N002]
✓ PASS | Chennai-Bangalore Highway Flooding (Feb 2025)      | Pred: No (justified) [N001]
✓ PASS | Chennai-Bangalore Highway Flooding Cont. (Mar 2025)| Pred: No (justified) [N001]
✓ PASS | Nationwide Diesel Price Rise (May 2025)            | Pred: No (justified) [N003]
✓ PASS | Mumbai-Pune Stable Demand Distractor (Sep 2025)    | Pred: Yes            [None]
✓ PASS | Mumbai-Delhi Delays with Unaffected Costs          | Pred: Yes            [None]
✓ PASS | Delhi-Jaipur Unexplained Spikes (Nov 2024)         | Pred: Yes            [None]
✓ PASS | Fleet Tracking Absorbed Surcharge Distractor       | Pred: Yes            [None]
✓ PASS | Toll Plaza on Other Routes Distractor              | Pred: Yes            [None]
======================================================================
 Total Benchmark Scenarios : 9
 Verdict Accuracy          : 100.0% (9/9)
 Note Attribution Accuracy : 100.0% (9/9)
 Hallucination Rate        : 0.0% (0/9)
======================================================================
```

---

## 8. Interactive Q&A Assistant (Stretch Goal)

Ask ad-hoc natural language questions against the shipping cost database:

```bash
# Query specific route and month
python3 src/assistant/interactive_qa.py "Why did Ahmedabad-Mumbai spike in January 2025?"

# Query operational disruption
python3 src/assistant/interactive_qa.py "What happened on Chennai-Bangalore in March 2025?"

# Query summary of all unexplained spikes
python3 src/assistant/interactive_qa.py "Show all unexplained anomalies"
```

Or launch interactive conversational mode:
```bash
python3 src/assistant/interactive_qa.py
```

---

## 9. Quickstart & Verification Guide

### 1. Run Complete Pipeline
```bash
python3 src/pipeline.py
```
Output written to `output/analysis_results.csv`.

### 2. Run Reproducibility Check (3 Passes)
```bash
python3 src/eval/reproducibility.py
```

### 3. Run Test Suite & Evaluation Benchmark
```bash
python3 -m unittest discover tests
python3 tests/eval_harness.py
```

### 4. Launch Interactive Visual Dashboard
```bash
python3 scripts/visualizer.py
open output/dashboard.html
```

---

## 10. 10-Minute Round 2 Walkthrough Guide

| Time | Agenda | Focus & Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 2:00** | **Problem Framing & Architecture** | Explain the core challenge: balancing automated cost monitoring with trust. Walk through the 3-layer architecture (Deterministic Metrics Engine $\rightarrow$ Hybrid RAG $\rightarrow$ Anti-Hallucination Guardrails). |
| **2:00 - 4:30** | **Mathematical Precision & Baselines** | Show code in `src/core/metrics.py`. Highlight the strictly prior trailing 8-week window (zero lookahead) and peer route calculation excluding self. |
| **4:30 - 7:00** | **RAG Guardrails & Anti-Hallucination** | Demonstrate how deceptive notes (N005 maintenance without cost impact, N006 stable demand, N010 absorbed costs) are rejected. Show why `Delhi-Jaipur` and `Mumbai-Pune` are flagged `Yes` while `Ahmedabad-Mumbai` is marked `No (justified)` citing `N002`. |
| **7:00 - 8:30** | **Evaluation & Reproducibility Proof** | Run `tests/eval_harness.py` (100% accuracy, 0% hallucination) and `src/eval/reproducibility.py` (0 diffs across 3 runs). |
| **8:30 - 10:00**| **Live Q&A & Dashboard Demo** | Demonstrate the interactive assistant answering natural language questions and walk through the visual dashboard. |
