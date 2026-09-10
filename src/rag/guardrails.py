"""
Anti-Hallucination Guardrails Engine.
Strictly verifies that any candidate note genuinely justifies a cost increase on a specific route and date.
"""

from typing import Tuple, Optional
from src.rag.context_store import ContextNote


# Explicit non-causal patterns that DO NOT justify a cost increase
NON_JUSTIFYING_PATTERNS = [
    "not part of this dataset",
    "costs were not significantly affected",
    "no major disruptions",
    "no significant disruptions",
    "remained normal",
    "returned to normal",
    "absorbed by transporters without a rate change",
    "without a rate change",
    "improved road conditions"
]

# Valid cost driver indicators
COST_INCREASE_PATTERNS = [
    "higher trip costs",
    "temporary surcharge",
    "prices rose",
    "pushing up transportation costs",
    "flooding",
    "disrupted normal truck movement",
    "surcharge"
]


class GuardrailEngine:
    """
    Evaluates candidate notes against strict validation rules.
    Prevents false justifications and hallucinations.
    """

    @staticmethod
    def is_cost_driver_note(note: ContextNote) -> bool:
        """
        Check if the note describes an event that actually causes freight cost to increase.
        Rejects notes that describe normal operations, absorbed costs, or unrelated routes.
        """
        text_lower = note.text.lower()
        
        # Check negative/neutral triggers first
        for pattern in NON_JUSTIFYING_PATTERNS:
            if pattern in text_lower:
                return False
                
        # Check positive cost increase indicators
        for pattern in COST_INCREASE_PATTERNS:
            if pattern in text_lower:
                return True
                
        return False

    @classmethod
    def validate_candidate(
        cls,
        candidate_note: ContextNote,
        route: str,
        week_of: str
    ) -> Tuple[bool, str]:
        """
        Validate whether the candidate note can legitimately justify the cost rise.
        Returns (is_valid, validation_reason).
        """
        # Rule 1: Route Applicability
        if not candidate_note.is_route_applicable(route):
            return False, f"Note {candidate_note.note_id} applies to {candidate_note.applies_to}, not {route}."

        # Rule 2: Temporal Window
        if not candidate_note.is_date_applicable(week_of):
            return False, f"Note {candidate_note.note_id} dated {candidate_note.date_str} is outside the active window for week {week_of}."

        # Rule 3: Causal Validity
        if not cls.is_cost_driver_note(candidate_note):
            return False, f"Note {candidate_note.note_id} does not describe a valid cost driver (e.g. costs were unaffected or absorbed)."

        return True, "Valid causal justification."
