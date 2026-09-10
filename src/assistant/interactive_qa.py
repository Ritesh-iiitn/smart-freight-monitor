"""
Interactive Natural Language Q&A Assistant for Shipping Costs.
Allows users to ask questions in plain English about route cost anomalies, baselines,
historical trends, and context notes.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import csv
import re
from typing import List, Dict, Any, Optional
from src.rag.context_store import load_context_notes, ContextNote


class ShippingAssistantQA:
    def __init__(
        self,
        analysis_csv_path: str = "output/analysis_results.csv",
        notes_csv_path: str = "data/context_notes.csv"
    ):
        self.records: List[Dict[str, str]] = []
        if os.path.exists(analysis_csv_path):
            with open(analysis_csv_path, "r", encoding="utf-8") as f:
                self.records = list(csv.DictReader(f))
        self.notes: List[ContextNote] = load_context_notes(notes_csv_path)

    def answer_question(self, query: str) -> str:
        """Process a natural language query and return an analytical answer."""
        query_lower = query.lower()
        
        # 1. Check for summary / list of anomalies
        if any(w in query_lower for w in ["unexplained", "flagged", "all anomalies", "all flags", "summary", "overview"]):
            return self._summarize_anomalies(query_lower)

        # 2. Extract Route if mentioned
        target_route = self._extract_route(query_lower)
        
        # 3. Extract Year/Month if mentioned
        target_year, target_month = self._extract_date_hints(query_lower)
        
        # Filter records
        matching_rows = self.records
        if target_route:
            matching_rows = [r for r in matching_rows if r["route"].lower() == target_route.lower()]
        if target_year:
            matching_rows = [r for r in matching_rows if r["week_of"].startswith(target_year)]
        if target_month:
            matching_rows = [r for r in matching_rows if f"-{target_month:02d}-" in r["week_of"]]

        if not matching_rows:
            return f"I could not find shipment records matching route '{target_route or 'Any'}' for the specified timeframe."

        # Find flagged or significant events in the matching window
        spikes = [r for r in matching_rows if r["flagged"] != "No"]
        
        if spikes:
            responses = []
            for s in spikes:
                verdict_str = "JUSTIFIED" if "justified" in s["flagged"].lower() else "UNJUSTIFIED SPIKE (FLAGGED)"
                responses.append(
                    f"• Route: {s['route']} (Week of {s['week_of']})\n"
                    f"  Cost: {s['cost_per_tonne_km']} INR/t-km ({s['vs_own_history']}, {s['vs_similar_routes']})\n"
                    f"  Verdict: {verdict_str} [Note: {s['matched_note_id'] or 'None'}]\n"
                    f"  Explanation: {s['reason']}"
                )
            return "\n\n".join(responses)
        else:
            # Normal route behavior
            avg_cptk = sum(float(r["cost_per_tonne_km"]) for r in matching_rows) / len(matching_rows)
            return (
                f"For {target_route or 'the selected routes'} during this period, freight costs were stable.\n"
                f"Average cost: {avg_cptk:.2f} INR per tonne-km with no unjustified spikes or operational disruptions detected."
            )

    def _extract_route(self, text: str) -> Optional[str]:
        known_routes = [
            "Mumbai-Pune", "Delhi-Jaipur", "Chennai-Bangalore",
            "Ahmedabad-Mumbai", "Kolkata-Bhubaneswar", "Bangalore-Hyderabad",
            "Mumbai-Delhi", "Delhi-Kolkata"
        ]
        for r in known_routes:
            # Direct match or cities mentioned
            cities = r.lower().split("-")
            if r.lower() in text or (cities[0] in text and cities[1] in text):
                return r
        return None

    def _extract_date_hints(self, text: str) -> (Optional[str], Optional[int]):
        year = "2024" if "2024" in text else ("2025" if "2025" in text else None)
        
        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
            "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        found_month = None
        for m_name, m_num in months.items():
            if re.search(rf"\b{m_name}\b", text):
                found_month = m_num
                break
                
        return year, found_month

    def _summarize_anomalies(self, text: str) -> str:
        flagged_rows = [r for r in self.records if r["flagged"] != "No"]
        if not flagged_rows:
            return "No cost anomalies have been detected in the dataset."
            
        unexplained = [r for r in flagged_rows if r["flagged"] == "Yes"]
        justified = [r for r in flagged_rows if "justified" in r["flagged"].lower()]
        
        lines = [
            "==================================================",
            f"          SHIPPING COST ANOMALIES AUDIT ({len(flagged_rows)} Events)",
            "==================================================",
            f"\n🚨 UNEXPLAINED / ACTION REQUIRED ({len(unexplained)} items):"
        ]
        for u in unexplained:
            lines.append(
                f" • {u['route']} (Week {u['week_of']}): {u['cost_per_tonne_km']} INR/t-km ({u['vs_own_history']})\n"
                f"   Reason: {u['reason']}"
            )
            
        lines.append(f"\n✅ JUSTIFIED COST RISES ({len(justified)} items):")
        for j in justified:
            lines.append(
                f" • {j['route']} (Week {j['week_of']}): {j['cost_per_tonne_km']} INR/t-km ({j['vs_own_history']})\n"
                f"   Note: [{j['matched_note_id']}] {j['reason']}"
            )
        lines.append("==================================================")
        return "\n".join(lines)

    def start_cli_session(self):
        """Run interactive conversational loop."""
        print("\n" + "="*60)
        print("  🤖 FreightTiger Smart Shipping Cost Assistant (Q&A Mode)")
        print("  Type your question or 'exit' / 'quit' to finish.")
        print("  Examples:")
        print("   - 'Why did Ahmedabad-Mumbai spike in Jan 2025?'")
        print("   - 'What caused delays or higher costs on Chennai-Bangalore in March 2025?'")
        print("   - 'List all unexplained anomalies'")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("User Question ❯ ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                    
                answer = self.answer_question(user_input)
                print(f"\nAssistant:\n{answer}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    assistant = ShippingAssistantQA()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(assistant.answer_question(query))
    else:
        assistant.start_cli_session()
