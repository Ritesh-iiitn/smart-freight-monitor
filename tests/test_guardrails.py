"""
Unit tests for Anti-Hallucination Guardrails and RAG Explainer.
"""

import unittest
from src.core.metrics import WeeklyRouteMetric
from src.rag.context_store import ContextNote, load_context_notes
from src.rag.guardrails import GuardrailEngine
from src.rag.explainer import ExplanationEngine


class TestGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = load_context_notes("data/context_notes.csv")
        cls.note_map = {n.note_id: n for n in cls.notes}
        cls.explainer = ExplanationEngine()

    def test_rejection_of_non_causal_notes(self):
        # N004: toll plaza not part of this dataset
        self.assertFalse(GuardrailEngine.is_cost_driver_note(self.note_map["N004"]))
        
        # N005: maintenance but costs unaffected
        self.assertFalse(GuardrailEngine.is_cost_driver_note(self.note_map["N005"]))
        
        # N006: stable demand
        self.assertFalse(GuardrailEngine.is_cost_driver_note(self.note_map["N006"]))
        
        # N008: normal movement
        self.assertFalse(GuardrailEngine.is_cost_driver_note(self.note_map["N008"]))
        
        # N010: costs absorbed without rate change
        self.assertFalse(GuardrailEngine.is_cost_driver_note(self.note_map["N010"]))

    def test_acceptance_of_valid_cost_drivers(self):
        # N001: flooding & detours
        self.assertTrue(GuardrailEngine.is_cost_driver_note(self.note_map["N001"]))
        
        # N002: festival surcharge
        self.assertTrue(GuardrailEngine.is_cost_driver_note(self.note_map["N002"]))
        
        # N003: nationwide diesel price rise
        self.assertTrue(GuardrailEngine.is_cost_driver_note(self.note_map["N003"]))

    def test_explanation_generation_justified_ahmedabad_mumbai(self):
        metric = WeeklyRouteMetric(
            route="Ahmedabad-Mumbai",
            route_type="Medium",
            week_of="2025-01-20",
            total_cost=32900,
            total_tonne_km=10000,
            shipment_count=5
        )
        flagged, note_id, reason = self.explainer.generate_explanation(metric)
        self.assertEqual(flagged, "No (justified)")
        self.assertEqual(note_id, "N002")
        self.assertIn("N002", reason)
        self.assertIn("surcharge", reason.lower())

    def test_explanation_generation_unexplained_delhi_jaipur(self):
        metric = WeeklyRouteMetric(
            route="Delhi-Jaipur",
            route_type="Short",
            week_of="2024-11-11",
            total_cost=41700,
            total_tonne_km=10000,
            shipment_count=5
        )
        flagged, note_id, reason = self.explainer.generate_explanation(metric)
        self.assertEqual(flagged, "Yes")
        self.assertEqual(note_id, "")
        self.assertIn("No matching note found", reason)

    def test_explanation_generation_closest_non_causal_mumbai_pune(self):
        metric = WeeklyRouteMetric(
            route="Mumbai-Pune",
            route_type="Short",
            week_of="2025-09-15",
            total_cost=39800,
            total_tonne_km=10000,
            shipment_count=5
        )
        flagged, note_id, reason = self.explainer.generate_explanation(metric)
        self.assertEqual(flagged, "Yes")
        self.assertEqual(note_id, "")
        self.assertIn("N006", reason)
        self.assertIn("does not describe a reason for a cost rise", reason)


if __name__ == "__main__":
    unittest.main()
