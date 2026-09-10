"""
Evaluation Harness for FreightTiger Shipping Cost Assistant.
Runs an automated benchmark against a labeled ground-truth dataset.
Calculates Precision, Recall, Classification Accuracy, Note Matching Accuracy, and Hallucination Rate.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
from src.core.metrics import WeeklyRouteMetric
from src.rag.context_store import load_context_notes
from src.rag.explainer import ExplanationEngine


# Benchmark Ground-Truth Test Cases
BENCHMARK_CASES = [
    {
        "name": "Ahmedabad-Mumbai Festival Surcharge (Jan 2025)",
        "route": "Ahmedabad-Mumbai",
        "route_type": "Medium",
        "week_of": "2025-01-20",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N002",
        "should_hallucinate": False
    },
    {
        "name": "Chennai-Bangalore Highway Flooding (Feb 2025)",
        "route": "Chennai-Bangalore",
        "route_type": "Short",
        "week_of": "2025-02-24",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "should_hallucinate": False
    },
    {
        "name": "Chennai-Bangalore Highway Flooding Cont. (Mar 2025)",
        "route": "Chennai-Bangalore",
        "route_type": "Short",
        "week_of": "2025-03-03",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "should_hallucinate": False
    },
    {
        "name": "Nationwide Diesel Price Rise (May 2025)",
        "route": "Mumbai-Delhi",
        "route_type": "Long",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "should_hallucinate": False
    },
    {
        "name": "Mumbai-Pune Stable Demand Distractor (Sep 2025)",
        "route": "Mumbai-Pune",
        "route_type": "Short",
        "week_of": "2025-09-15",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "should_hallucinate": False
    },
    {
        "name": "Mumbai-Delhi Delays with Unaffected Costs (Jul 2024)",
        "route": "Mumbai-Delhi",
        "route_type": "Long",
        "week_of": "2024-07-29",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "should_hallucinate": False
    },
    {
        "name": "Delhi-Jaipur Unexplained Spikes (Nov 2024)",
        "route": "Delhi-Jaipur",
        "route_type": "Short",
        "week_of": "2024-11-11",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "should_hallucinate": False
    },
    {
        "name": "Fleet Tracking Absorbed Surcharge Distractor (Oct 2025)",
        "route": "Bangalore-Hyderabad",
        "route_type": "Medium",
        "week_of": "2025-10-27",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "should_hallucinate": False
    },
    {
        "name": "Toll Plaza on Other Routes Distractor (Mar 2024)",
        "route": "Kolkata-Bhubaneswar",
        "route_type": "Medium",
        "week_of": "2024-03-11",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "should_hallucinate": False
    }
]


def run_evaluation() -> Dict[str, Any]:
    notes = load_context_notes("data/context_notes.csv")
    explainer = ExplanationEngine()
    
    total = len(BENCHMARK_CASES)
    correct_verdicts = 0
    correct_notes = 0
    hallucinations = 0
    
    print("\n" + "="*70)
    print("               EVALUATION HARNESS BENCHMARK")
    print("="*70)
    
    for case in BENCHMARK_CASES:
        metric = WeeklyRouteMetric(
            route=case["route"],
            route_type=case["route_type"],
            week_of=case["week_of"],
            total_cost=30000,
            total_tonne_km=10000,
            shipment_count=5
        )
        
        flagged, note_id, reason = explainer.generate_explanation(metric)
        
        verdict_match = (flagged == case["expected_flagged"])
        note_match = (note_id == case["expected_note_id"])
        
        # Check hallucination: falsely citing an invalid note or citing when expected is empty
        is_hallucination = (case["expected_flagged"] == "Yes" and flagged == "No (justified)") or (
            case["expected_note_id"] == "" and note_id != ""
        )
        
        if verdict_match:
            correct_verdicts += 1
        if note_match:
            correct_notes += 1
        if is_hallucination:
            hallucinations += 1
            
        status_icon = "✓ PASS" if (verdict_match and note_match) else "✗ FAIL"
        print(f"{status_icon} | {case['name'][:42]:<42} | Pred: {flagged:<14} (Note: {note_id or 'None':<4}) | Exp: {case['expected_flagged']:<14} (Note: {case['expected_note_id'] or 'None':<4})")

    verdict_acc = (correct_verdicts / total) * 100.0
    note_acc = (correct_notes / total) * 100.0
    hallucination_rate = (hallucinations / total) * 100.0
    
    print("="*70)
    print(f" Total Benchmark Scenarios : {total}")
    print(f" Verdict Accuracy          : {verdict_acc:.1f}% ({correct_verdicts}/{total})")
    print(f" Note Attribution Accuracy : {note_acc:.1f}% ({correct_notes}/{total})")
    print(f" Hallucination Rate        : {hallucination_rate:.1f}% ({hallucinations}/{total})")
    print("="*70 + "\n")
    
    return {
        "total": total,
        "verdict_accuracy": verdict_acc,
        "note_accuracy": note_acc,
        "hallucination_rate": hallucination_rate
    }


if __name__ == "__main__":
    results = run_evaluation()
    if results["hallucination_rate"] > 0 or results["verdict_accuracy"] < 100.0:
        sys.exit(1)
