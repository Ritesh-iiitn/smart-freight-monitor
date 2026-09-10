# 🚚 FreightTiger Smart Shipping Cost Assistant
### *24-Hour AI Challenge — Software Engineering Intern (AI) Case Study*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-ff4b4b?style=for-the-badge&logo=streamlit)](https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.9+-38bdf8.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-100%25%20(0%20diffs)-10b981.svg)]()
[![Hallucination Rate](https://img.shields.io/badge/hallucination%20rate-0.0%25-10b981.svg)]()

> 🔗 **Live Cloud Deployment**: [https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/](https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/)  
> An enterprise-grade, guardrailed AI assistant designed to monitor freight shipping costs, compute rolling historical and peer baselines with mathematical rigor, detect anomalous price creep, and ground explanations in operational context notes using **Retrieval-Augmented Generation (RAG)** with strict **zero-hallucination causal validation**.

---

## 📑 Table of Contents
1. [Problem Statement & Objectives](#1-problem-statement--objectives)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Mathematical Formulations & Baselines](#3-mathematical-formulations--baselines)
4. [AI / RAG & Anti-Hallucination Guardrails](#4-ai--rag--anti-hallucination-guardrails)
5. [Evaluation Benchmark & Trust Verification](#5-evaluation-benchmark--trust-verification)
6. [Reproducibility Audit Across Runs](#6-reproducibility-audit-across-runs)
7. [Token Consumption & Pricing Ledger](#7-token-consumption--pricing-ledger)
8. [Live Streamlit Web Application](#8-live-streamlit-web-application)
9. [Project Layout & Code Structure](#9-project-layout--code-structure)
10. [Quickstart & Execution Commands](#10-quickstart--execution-commands)
11. [10-Minute Round 2 Walkthrough Guide](#11-10-minute-round-2-walkthrough-guide)

---

## 1. Problem Statement & Objectives

In freight logistics, companies pay carriers based on weight carried and distance traversed ($\text{INR} / (\text{Tonne} \times \text{Km})$). Over time, shipping rates quietly creep up on specific corridors:
* **Legitimate Drivers**: Fuel price hikes, monsoon detours, regional festival surcharges, toll plaza changes.
* **Unjustified Creep**: Carrier rate hikes, artificial capacity constraints, or unverified surcharges.

### Core Objectives & Deliverables
* **Deterministic Weekly Aggregation**: Normalize all shipments by corridor (`origin-destination`) and provided length bucket (`Short`, `Medium`, `Long`) into Monday–Sunday weekly windows (`week_of`).
* **Dual Baseline Tracking**:
  - **vs. Own History**: Trailing 8-week rolling average strictly prior to current week (zero lookahead).
  - **vs. Similar Routes**: Same-week peer route average across identical length buckets (excluding self).
* **Grounded RAG Verification**: Query operational disruption notes (`context_notes.csv`) to corroborate price jumps.
* **Strict Anti-Hallucination Guardrails**: Eliminate fabricated justifications; reject non-causal notes (e.g. routine maintenance with unaffected costs or absorbed compliance fees).
* **Deterministic Output & Reproducibility**: 3 untouched runs on the same input produce identical numbers, flags, and cited notes.

---

## 2. End-to-End System Architecture

```
                                  ┌───────────────────────────┐
                                  │   shipment_records.csv    │
                                  │  (11,227 Shipment Events) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             Deterministic Analytics Layer                                │
│                                                                                          │
│   1. Group into Monday-to-Sunday Weeks: week_of = Monday (YYYY-MM-DD)                   │
│   2. Weighted Cost Metric: CPTK = Sum(freight_cost_inr) / Sum(quantity * distance)       │
│   3. Trailing 8-Week Baseline (Strictly Prior Weeks, No Look-Ahead)                      │
│   4. Peer Group Baseline (Same Week, Same Route Type, Excluding Self)                    │
│   5. Anomaly Escalation Filter: vs_own_history >= +8.0% OR vs_peers >= +15.0%            │
└───────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                │
                                    [Cost Spike Detected]
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Semantic & Lexical RAG Retriever                          │
│                                                                                          │
│   • Source Document: context_notes.csv (Indexed In-Memory)                               │
│   • Vector / TF-IDF Dense Cosine Scoring + Temporal & Corridor Proximity Filtering       │
│   • Retrieves Top-K Candidate Context Notes for the Target Route-Week                    │
└───────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        Strict Anti-Hallucination Guardrail Engine                        │
│                                                                                          │
│   ✓ Scope Rule: Candidate note matches route corridor OR "All Routes"                    │
│   ✓ Temporal Rule: Target week falls strictly within active event start-to-end window    │
│   ✓ Causal Rule: Note describes an actual cost-increasing driver                         │
│                  (Filters out N004, N005, N006, N007, N008, N009, N010 non-causal notes)  │
└───────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
            [Justification Valid]                                [Invalid / Unexplained]
       • flagged: "No (justified)"                          • flagged: "Yes"
       • matched_note_id: "N002"                            • matched_note_id: ""
       • reason: Cites note & exact causal dates            • reason: Grounded explanation for review
                      │                                                   │
                      └─────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Output & Presentation Hub                                │
│                                                                                          │
│   • output/analysis_results.csv (Exact Grade-Tested Contract Format)                     │
│   • Live Streamlit App (https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/) │
│   • Live Natural Language Q&A Assistant (CLI + Streamlit Chat)                           │
│   • Automated Verification & Reproducibility Suite (3 Independent Passes)                │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations & Baselines

To monitor freight shipping rate creep accurately and fairly across diverse routes and haul lengths, the system implements a **deterministic four-step computation pipeline**:

1. **Time-Series Alignment (`week_of`)**: Standardizes daily shipment timestamps into uniform Monday-to-Sunday weekly windows.
2. **Normalized Work Metric (`cost_per_tonne_km`)**: Evaluates freight cost against total payload work (weight × distance).
3. **Internal Trend Baseline (`vs_own_history`)**: Measures rate changes against a trailing 8-week historical window (strictly zero lookahead).
4. **Market Cohort Baseline (`vs_similar_routes`)**: Benchmarks against peer routes of identical distance classification in the same week.

---

### A. Weekly Time-Series Grouping (`week_of`)

* **Objective**: Partition continuous daily shipments into non-overlapping Monday–Sunday weekly cycles.
* **Definition**: `week_of` represents the ISO date (`YYYY-MM-DD`) corresponding to that week's Monday.

$$
\text{week\_of} = \text{Shipment Date} - (\text{Weekday Index} \times 1\text{ Day}) \quad \text{where } \text{Monday} = 0, \, \dots, \, \text{Sunday} = 6
$$

> **💡 Intuitive Example:**
> - A shipment on **Wednesday, Jan 15, 2025** (`weekday = 2`) rolls back 2 days $\rightarrow$ `2025-01-13` (Monday).
> - A shipment on **Sunday, Jan 19, 2025** (`weekday = 6`) rolls back 6 days $\rightarrow$ `2025-01-13` (Monday).

---

### B. Normalized Cost per Tonne-Km ($\text{CPTK}$)

* **Objective**: Eliminate distortions caused by differences in truckload sizes or journey lengths. Simply averaging invoice totals is misleading because transporting 50 tonnes over 1,000 km represents substantially more logistics work than transporting 2 tonnes over 50 km.
* **Definition**: Total freight spend divided by total tonne-kilometers across all shipments on corridor $r$ in week $w$.

$$
\text{CPTK}_{r, w} = \frac{\text{Total Freight Cost (INR)}}{\text{Total Transportation Work (Tonnes} \times \text{Km})} = \frac{\sum_{i \in \mathcal{S}_{r, w}} \text{FreightCost}_i}{\sum_{i \in \mathcal{S}_{r, w}} (\text{Quantity}_i \times \text{Distance}_i)}
$$

* **Rounding Rule**: Rounded to **2 decimal places** (e.g. `3.45`).

> **💡 Step-by-Step Worked Example:**
> Suppose Corridor **Mumbai–Pune** has 2 shipments in a given week:
> - **Shipment 1**: 20 Tonnes over 150 km @ ₹9,000 $\rightarrow \text{Work} = 20 \times 150 = 3,000 \text{ Tonne-Km}$
> - **Shipment 2**: 10 Tonnes over 150 km @ ₹6,000 $\rightarrow \text{Work} = 10 \times 150 = 1,500 \text{ Tonne-Km}$
>
> $$
> \text{Total Spend} = 9,000 + 6,000 = ₹15,000 \quad \Big| \quad \text{Total Work} = 3,000 + 1,500 = 4,500 \text{ Tonne-Km}
> $$
> $$
> \text{CPTK} = \frac{15,000}{4,500} = \mathbf{₹3.33} \text{ per Tonne-Km}
> $$

---

### C. Baseline 1: vs. Own History (`vs_own_history`)

* **Objective**: Detect whether a route's shipping rate has spiked relative to its own recent historical baseline.
* **Rolling Window**: Trailing arithmetic mean over the prior $K$ weeks ($1 \le K \le 8$), **strictly excluding the current week** to prevent look-ahead bias and data leakage.

$$
\text{HistAvg}_{r, w} = \frac{1}{K} \sum_{k=1}^{K} \text{CPTK}_{r, w-k} \quad \text{where } K = \min(8, \text{number of prior available weeks})
$$

$$
\Delta \text{Hist\%} = \left( \frac{\text{CPTK}_{r, w} - \text{HistAvg}_{r, w}}{\text{HistAvg}_{r, w}} \right) \times 100
$$

* **Strict Contract Output Format**: Formatted with explicit sign and 1 decimal place:
  - `+35.5% vs this route's past average`
  - `-12.3% vs this route's past average`
* **Edge Cases & Zero Padding**:
  - **Initial Week** ($0$ prior weeks available): Defaults to `+0.0% vs this route's past average`.
  - **$< 8$ Prior Weeks**: Averages all available prior weeks ($1 \le K < 8$).

> **💡 Step-by-Step Worked Example:**
> - Current Week CPTK: **₹4.50**
> - Trailing 8-Week Historical Average ($\text{HistAvg}$): **₹3.00**
> - Percentage Shift: $\frac{4.50 - 3.00}{3.00} \times 100 = \mathbf{+50.0\%}$
> - Contract Result: `+50.0% vs this route's past average`

---

### D. Baseline 2: vs. Similar Routes (`vs_similar_routes`)

* **Objective**: Benchmark a corridor against market peers operating over comparable haul distances in the **exact same week**, isolating local corridor inflation from broader macroeconomic shifts (e.g., nationwide fuel hikes).
* **Peer Cohort Grouping**: All *other* corridors sharing the same `route_type` length category:
  - **Short**: $< 500\text{ km}$
  - **Medium**: $500\text{--}1,200\text{ km}$
  - **Long**: $> 1,200\text{ km}$
* **Peer Group Average**: Arithmetic mean of CPTK across all peer routes in that week, **strictly excluding the target route itself**:

$$
\text{PeerAvg}_{r, w} = \frac{1}{|\mathcal{P}_{w}| - 1} \sum_{p \in \mathcal{P}_{w}, \, p \ne r} \text{CPTK}_{p, w} \quad (\text{where } \mathcal{P}_w \text{ is the set of all routes in that cohort})
$$

$$
\Delta \text{Peer\%} = \left( \frac{\text{CPTK}_{r, w} - \text{PeerAvg}_{r, w}}{\text{PeerAvg}_{r, w}} \right) \times 100
$$

* **Strict Contract Output Format**: Formatted with explicit sign and 1 decimal place:
  - `+21.0% vs similar-length routes this week`
  - `-5.4% vs similar-length routes this week`
* **Edge Case**: If a corridor is the only route operating in its distance category that week ($|\mathcal{P}_w| = 1$), it defaults to `+0.0% vs similar-length routes this week`.

> **💡 Step-by-Step Worked Example:**
> In week `2025-02-03`, there are 3 routes in the **Short** cohort:
> - **Target Route (Delhi–Jaipur)**: $\text{CPTK} = \mathbf{₹4.00}$
> - **Peer Route 1 (Mumbai–Pune)**: $\text{CPTK} = ₹3.00$
> - **Peer Route 2 (Chennai–Bangalore)**: $\text{CPTK} = ₹2.00$
>
> $$
> \text{Peer Average (excluding Delhi-Jaipur)} = \frac{3.00 + 2.00}{2} = \mathbf{₹2.50}
> $$
> $$
> \Delta \text{Peer\%} = \left(\frac{4.00 - 2.50}{2.50}\right) \times 100 = \mathbf{+60.0\%}
> $$
> - Contract Result: `+60.0% vs similar-length routes this week`

---

### E. Anomaly Escalation & Trigger Rules

A route-week is flagged as a candidate cost spike and routed to the AI/RAG verification pipeline when it meets either of the following conditions:

| Trigger Rule | Threshold Condition | Operational Rationale |
| :--- | :--- | :--- |
| **Primary Trigger (Historical Spike)** | $\Delta \text{ vs. Own History} \ge \mathbf{+8.0\%}$ | Corridor cost is noticeably higher than its own recent 8-week norm. |
| **Secondary Trigger (Peer Divergence)** | $\Delta \text{ vs. Similar Routes} \ge \mathbf{+15.0\%}$ <br> *AND* $\Delta \text{ vs. Own History} \ge \mathbf{+5.0\%}$ | Corridor cost significantly exceeds peer market rates while simultaneously trending upward. |

---

### F. Output Schema & Specification Summary

| CSV Column Name | Mathematical Source / Definition | Data Type & Precision | Concrete Example Output |
| :--- | :--- | :--- | :--- |
| `route` | `origin-destination` corridor identifier | String (`Orig-Dest`) | `Chennai-Bangalore` |
| `week_of` | ISO date of Monday in the shipment week | String (`YYYY-MM-DD`) | `2025-02-24` |
| `cost_per_tonne_km` | Total Spend $\div$ Total Tonne-Km | Float (2 decimal places) | `3.45` |
| `vs_own_history` | Percentage change vs trailing 8-week mean | Signed String (1 decimal place) | `+35.5% vs this route's past average` |
| `vs_similar_routes` | Percentage change vs same-week cohort mean | Signed String (1 decimal place) | `+21.0% vs similar-length routes this week` |
| `flagged` | Anomaly status after causal verification | String (`Yes`, `No (justified)`, `No`) | `No (justified)` |
| `matched_note_id` | Validated causal context note ID (if any) | String (`N001` - `N010`, or empty) | `N001` |
| `reason` | Factual explanation of cost trend or disruption | String (Plain English explanation) | `Highway flooding caused detour...` |

---

## 4. AI / RAG & Anti-Hallucination Guardrails

A major risk in automated AI assistants is **sycophancy and false justification**: when a cost spike occurs, an unconstrained LLM might cite an unrelated note (e.g. routine maintenance or a report of stable demand) simply because it contains matching city names.

### Context Note Ground-Truth & Guardrail Matrix

| Note ID | Date | Corridor | Note Content & Intent | Impact Polarity | Guardrail Action | Output Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N001** | 2025-02-24 | Chennai-Bangalore | Flooding caused longer detours & higher trip costs (Feb 24 - Mar 8) | **Cost Increase** | ✅ **Valid Causal Factor** | `No (justified)` |
| **N002** | 2025-01-20 | Ahmedabad-Mumbai | Festival week temporary surcharge due to capacity shortage | **Cost Increase** | ✅ **Valid Causal Factor** | `No (justified)` |
| **N003** | 2025-05-05 | All Routes | Diesel prices rose nationwide (+5-7%) | **Cost Increase** | ✅ **Valid Causal Factor** | `No (justified)` |
| **N004** | 2024-03-11 | All Routes | Toll plaza commissioned, but *routes not in dataset* | **Irrelevant** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N005** | 2024-07-29 | Mumbai-Delhi | Maintenance delays, but *costs were not affected* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N006** | 2025-09-22 | All Routes | Logistics report: demand stable, *no major disruptions* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N007** | 2024-05-20 | Delhi-Jaipur | Road resurfacing completed, *improved conditions* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N008** | 2025-06-09 | Kolkata-Bhubaneswar | No disruptions, *freight movement normal* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N009** | 2025-03-17 | Chennai-Bangalore | Repairs completed, *returned to normal* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |
| **N010** | 2025-10-27 | All Routes | Fleet tracking mandate: *costs absorbed without rate change* | **Neutral** | ❌ **Rejected Non-Causal** | `Yes` (Review) |

---

## 5. Evaluation Benchmark & Trust Verification

We constructed a rigorous evaluation benchmark harness (`tests/eval_harness.py`) testing edge cases, non-causal distractors, and valid disruptions:

```
======================================================================
               EVALUATION HARNESS BENCHMARK SCORECARD
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

## 6. Reproducibility Audit Across Runs

To satisfy the grading requirement (*"Three untouched runs on the same input: identical flags and numbers"*), `src/eval/reproducibility.py` executes 3 complete independent passes:

```bash
python3 src/eval/reproducibility.py
```

```
============================================================
      STARTING REPRODUCIBILITY AUDIT (3 INDEPENDENT RUNS)
============================================================
Executing Run #1 -> output/reproducibility_test/run_1.csv
Executing Run #2 -> output/reproducibility_test/run_2.csv
Executing Run #3 -> output/reproducibility_test/run_3.csv

Comparing Run 1, Run 2, and Run 3 results...
 [PASS] Exact Byte-for-Byte Match across all 3 runs.
============================================================
 [SUCCESS] 100% REPRODUCIBILITY VERIFIED across 800 rows x 8 fields.
 - Numbers, verdicts, cited notes, and explanations are 100% identical.
 - Total Diff Count: 0
============================================================
```

---

## 7. Token Consumption & Pricing Ledger

To satisfy responsible AI grading (*"report total input tokens, total output tokens, number of LLM calls, and rough cost"*), our pipeline incorporates a transparent cost tracker:

| Metric | Open-Source / Local Engine (Default) | Google Gemini Flash | OpenAI GPT-4o-mini |
| :--- | :--- | :--- | :--- |
| **Total Route-Weeks** | 800 | 800 | 800 |
| **LLM / RAG Invocations** | 5 (Anomalies Only) | 5 | 5 |
| **Prompt Input Tokens** | 122 | 122 | 122 |
| **Completion Output Tokens** | 285 | 285 | 285 |
| **Total Tokens** | 407 | 407 | 407 |
| **Total Cost ($ USD)** | **\$0.000000** | **\$0.000095** | **\$0.000189** |
| **Total Cost (₹ INR)** | **₹0.0000** | **₹0.0083** | **₹0.0164** |
| **Execution Latency** | **< 0.09s** | ~ 1.20s | ~ 1.50s |

> [!TIP]
> Cost is kept ultra-low by running deterministic anomaly pre-filtering, so model inference is invoked only when a genuine price surge is flagged.

---

## 8. Live Streamlit Web Application

🌐 **Live Deployed App**: [https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/](https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/)

To run the application locally:
```bash
streamlit run streamlit_app.py
```

### Key UI Features:
1. **Executive KPI Cards**: Real-time counts of monitored route-weeks, unexplained spikes, justified events, and 0% hallucination rate.
2. **Interactive Chart View**: Multi-corridor line charts with dropdown route filters.
3. **Conversational AI Assistant**: Chat panel with quick-action prompt buttons for live Q&A.
4. **Searchable Anomaly Grid**: Filter records by status (`All`, `Unexplained Spikes`, `Justified`), with real-time column searching.
5. **Causal Guardrails Matrix**: Complete transparency into all 10 context notes and their validation verdicts.
6. **Live Verification Hub**: 1-click execution of the 3-pass reproducibility check and evaluation harness benchmark.

---

## 9. Project Layout & Code Structure

```
freighttiger/
├── data/
│   ├── context_notes.csv             # Operational disruption notes (RAG knowledge base)
│   ├── sample_output_format.csv      # Target output schema contract
│   └── shipment_records.csv          # 11,227 synthetic shipment records (2024-2025)
├── src/
│   ├── core/
│   │   ├── metrics.py                # Weekly aggregation & trailing 8-wk / peer baselines
│   │   └── anomaly_detector.py       # Threshold & statistical anomaly detection
│   ├── rag/
│   │   ├── context_store.py          # Context note parser & temporal window extractor
│   │   ├── retriever.py              # Semantic vector + lexical hybrid retriever
│   │   ├── guardrails.py             # Anti-hallucination causal verification engine
│   │   ├── explainer.py              # Grounded plain-English explanation generator
│   │   └── llm_client.py             # Pluggable adapter (Local, Gemini, OpenAI)
│   ├── assistant/
│   │   ├── interactive_qa.py         # Natural language Q&A assistant (CLI & Web)
│   │   └── terminal_dashboard.py     # Interactive Terminal TUI Dashboard
│   ├── eval/
│   │   ├── cost_tracker.py           # Token usage and pricing auditor
│   │   └── reproducibility.py        # 3-pass untouched run verification tool
│   └── pipeline.py                   # Full end-to-end pipeline orchestrator
├── tests/
│   ├── test_metrics.py               # Unit tests for baseline math & peer exclusion
│   ├── test_guardrails.py            # Unit tests for RAG guardrails & note filtering
│   └── eval_harness.py               # Automated benchmark evaluation scorecard
├── scripts/
│   └── generate_sample_shipments.py  # Deterministic shipment dataset generator
├── output/
│   └── analysis_results.csv          # Final grade-tested output CSV
├── streamlit_app.py                  # Primary Streamlit Interactive Dashboard
├── requirements.txt                  # Python dependencies
└── README.md                         # Comprehensive documentation & architecture guide
```

---

## 10. Quickstart & Execution Commands

```bash
# 1. Run Complete Analysis Pipeline (Generates output/analysis_results.csv)
python3 src/pipeline.py

# 2. Run Reproducibility Verification (3 Independent Passes)
python3 src/eval/reproducibility.py

# 3. Run Unit Tests & Benchmark Evaluation Harness
python3 -m unittest discover tests
python3 tests/eval_harness.py

# 4. Run Interactive Q&A Assistant via CLI
python3 src/assistant/interactive_qa.py "Why did Ahmedabad-Mumbai spike in January 2025?"
python3 src/assistant/interactive_qa.py "Show all unexplained anomalies"

# 5. Launch Primary Streamlit Interactive UI Locally
streamlit run streamlit_app.py
```

---

## 11. 10-Minute Round 2 Walkthrough Guide

| Timestamp | Presentation Phase | Key Talking Points & Live Actions |
| :--- | :--- | :--- |
| **0:00 - 2:00** | **Problem Framing & Architecture** | Explain the trade-off between AI capability and trust. Show the 3-tier architecture: Deterministic Metrics $\rightarrow$ Hybrid RAG $\rightarrow$ Causal Guardrails. |
| **2:00 - 4:00** | **Mathematical Rigor & Baselines** | Open `src/core/metrics.py`. Highlight the volume-weighted metric ($\frac{\sum \text{Cost}}{\sum \text{Tonnes} \times \text{Km}}$), strictly prior trailing 8-week baseline (zero lookahead), and peer route average excluding self. |
| **4:00 - 6:30** | **RAG Guardrails & Anti-Hallucination** | Explain how deceptive notes (N004, N005, N006, N010) are handled. Show why `Delhi-Jaipur` and `Mumbai-Pune` are flagged `Yes` (unexplained) while `Ahmedabad-Mumbai` is marked `No (justified)` citing `N002`. |
| **6:30 - 8:00** | **Evaluation & Reproducibility Proof** | Run `python3 tests/eval_harness.py` (100% accuracy, 0% hallucination) and `python3 src/eval/reproducibility.py` (0 diffs across 3 runs). |
| **8:00 - 10:00**| **Live Streamlit UI Demo & Q&A** | Open the [Live Streamlit App](https://ritesh-iiitn-smart-freight-monitor-streamlit-app-hrxkhq.streamlit.app/), demo the interactive charts, ask live questions in the chat panel, and take interviewer questions. |
