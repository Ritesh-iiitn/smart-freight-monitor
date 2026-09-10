"""
End-to-End Shipping Cost Assistant Pipeline.
Orchestrates ingestion, aggregation, baseline calculations, anomaly detection,
RAG context matching with anti-hallucination guardrails, and output generation.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import argparse
from typing import List, Optional
from src.core.metrics import (
    load_shipment_records, aggregate_weekly_metrics,
    compute_baselines, WeeklyRouteMetric
)
from src.core.anomaly_detector import AnomalyDetector
from src.rag.context_store import load_context_notes
from src.rag.retriever import VectorRetriever
from src.rag.explainer import ExplanationEngine
from src.eval.cost_tracker import CostTracker


OUTPUT_FIELDNAMES = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason"
]


def run_pipeline(
    shipments_path: str = "data/shipment_records.csv",
    notes_path: str = "data/context_notes.csv",
    output_path: str = "output/analysis_results.csv",
    only_flagged: bool = False,
    cost_tracker: Optional[CostTracker] = None
) -> List[WeeklyRouteMetric]:
    """Execute complete analysis pipeline."""
    tracker = cost_tracker or CostTracker(model_name="local_opensource")
    
    # 1. Load data
    records = load_shipment_records(shipments_path)
    notes = load_context_notes(notes_path)
    
    # 2. Aggregate weekly metrics
    metrics_dict = aggregate_weekly_metrics(records)
    
    # 3. Compute trailing 8-week and peer baselines
    metrics_list = compute_baselines(metrics_dict)
    
    # 4. Initialize detector, retriever, and guardrailed explainer
    detector = AnomalyDetector(own_history_threshold_pct=8.0, similar_routes_threshold_pct=15.0)
    retriever = VectorRetriever(notes)
    explainer = ExplanationEngine(retriever)
    
    # 5. Process each weekly route entry
    for metric in metrics_list:
        is_spike = detector.is_cost_spike(metric)
        if is_spike:
            # Query RAG + Guardrails
            flagged, note_id, reason = explainer.generate_explanation(metric)
            metric.flagged = flagged
            metric.matched_note_id = note_id
            metric.reason = reason
            
            # Record prompt & completion for token audit
            prompt_audit = f"Route: {metric.route}, Week: {metric.week_of}, CPTK: {metric.cost_per_tonne_km}, Baseline: {metric.vs_own_history_str}"
            tracker.record_call(prompt_audit, reason)
        else:
            metric.flagged = "No"
            metric.matched_note_id = ""
            metric.reason = "Cost is within normal historical baseline and peer group range."

    # Filter if requested
    export_metrics = [m for m in metrics_list if m.flagged != "No"] if only_flagged else metrics_list
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 6. Write output CSV matching strict specification
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        for m in export_metrics:
            writer.writerow(m.to_dict())

    print(f"Pipeline executed successfully. Processed {len(metrics_list)} route-weeks.")
    print(f"Exported {len(export_metrics)} rows to: {output_path}")
    
    return export_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreightTiger Shipping Cost Assistant")
    parser.add_argument("--shipments", default="data/shipment_records.csv", help="Path to shipment records CSV")
    parser.add_argument("--notes", default="data/context_notes.csv", help="Path to context notes CSV")
    parser.add_argument("--output", default="output/analysis_results.csv", help="Path to output CSV")
    parser.add_argument("--only-flagged", action="store_true", help="Output only flagged or justified rows")
    parser.add_argument("--model", default="local_opensource", choices=["local_opensource", "gemini_flash", "gpt_4o_mini"])
    
    args = parser.parse_args()
    
    tracker = CostTracker(model_name=args.model)
    run_pipeline(
        shipments_path=args.shipments,
        notes_path=args.notes,
        output_path=args.output,
        only_flagged=args.only_flagged,
        cost_tracker=tracker
    )
    tracker.print_report()
