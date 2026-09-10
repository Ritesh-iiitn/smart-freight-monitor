"""
Explanation Generator for Shipping Cost Assistant.
Produces deterministic, concise, plain-English explanations adhering to strict guardrails.
"""

from typing import Tuple, Optional
from src.core.metrics import WeeklyRouteMetric
from src.rag.context_store import ContextNote
from src.rag.retriever import VectorRetriever
from src.rag.guardrails import GuardrailEngine


class ExplanationEngine:
    def __init__(self, retriever: Optional[VectorRetriever] = None):
        self.retriever = retriever or VectorRetriever()
        self.guardrails = GuardrailEngine()

    def generate_explanation(self, metric: WeeklyRouteMetric) -> Tuple[str, str, str]:
        """
        Evaluate an anomalous metric and return:
        (flagged_status, matched_note_id, plain_english_reason)
        """
        # Retrieve candidate notes
        candidates = self.retriever.retrieve(
            route=metric.route,
            week_of=metric.week_of,
            query="freight cost increase surcharge disruption price hike"
        )
        
        # Check top candidates through strict guardrails
        justifying_note: Optional[ContextNote] = None
        closest_invalid_note: Optional[ContextNote] = None
        
        for note, score in candidates:
            # First check if note is temporally applicable or proximate
            if not note.is_route_applicable(metric.route):
                continue
                
            if note.is_date_applicable(metric.week_of):
                is_valid, val_reason = self.guardrails.validate_candidate(
                    candidate_note=note,
                    route=metric.route,
                    week_of=metric.week_of
                )
                if is_valid:
                    justifying_note = note
                    break
                else:
                    if closest_invalid_note is None:
                        closest_invalid_note = note
            elif note.is_temporally_proximate(metric.week_of, max_days=21):
                if closest_invalid_note is None:
                    closest_invalid_note = note

        if justifying_note:
            flagged = "No (justified)"
            matched_note_id = justifying_note.note_id
            
            # Format reason from note content
            note_summary = justifying_note.text.strip().rstrip(".")
            if len(note_summary) > 0 and note_summary[0].isupper():
                note_summary = note_summary[0].lower() + note_summary[1:]
                
            reason = (
                f"Matches note {justifying_note.note_id} dated {justifying_note.date_str}: "
                f"{note_summary}. The cost rise has a clear explanation."
            )
            return flagged, matched_note_id, reason

        elif closest_invalid_note:
            flagged = "Yes"
            matched_note_id = ""
            note_summary = closest_invalid_note.text.strip().rstrip(".")
            if len(note_summary) > 0 and note_summary[0].isupper():
                note_summary = note_summary[0].lower() + note_summary[1:]
                
            reason = (
                f"The closest note ({closest_invalid_note.note_id}, {closest_invalid_note.date_str}) "
                f"mentions {note_summary} -- it does not describe a reason for a cost rise on this route. "
                f"No genuine justification found; flagged for review."
            )
            return flagged, matched_note_id, reason

        else:
            flagged = "Yes"
            matched_note_id = ""
            reason = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
            return flagged, matched_note_id, reason
