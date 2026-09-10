"""
Interactive Natural Language Q&A Assistant for Shipping Costs.
Allows users to ask questions in plain English about route cost anomalies, baselines,
historical trends, and context notes.
Formats output with executive clarity and clean, readable structure.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import csv
import re
from typing import List, Dict, Any, Optional, Tuple
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
        """Process a natural language query and return a well-formatted, executive response."""
        query_lower = query.lower().strip()
        
        # Check if user specifically mentioned a route first
        target_route = self._extract_route(query_lower)
        target_year, target_month = self._extract_date_hints(query_lower)

        # If user asked about a specific route, answer for that route directly
        if target_route:
            return self._answer_route_query(target_route, target_year, target_month)

        # Otherwise check if asking for general anomalies / summary
        if any(w in query_lower for w in ["unexplained", "flagged", "all anomalies", "all flags", "summary", "overview", "list all"]):
            return self._summarize_anomalies(query_lower)

        # General / route search across all
        return self._answer_general_query(query_lower, target_year, target_month)

    def _answer_route_query(self, route: str, year: Optional[str], month: Optional[int]) -> str:
        matching_rows = [r for r in self.records if r["route"].lower() == route.lower()]
        if year:
            matching_rows = [r for r in matching_rows if r["week_of"].startswith(year)]
        if month:
            matching_rows = [r for r in matching_rows if f"-{month:02d}-" in r["week_of"]]

        if not matching_rows:
            time_desc = f"in {year or ''} {month or ''}".strip()
            return f"No shipment records were found for route **{route}** {time_desc}."

        spikes = [r for r in matching_rows if r["flagged"] != "No"]
        
        if spikes:
            sections = []
            for s in spikes:
                is_justified = "justified" in s["flagged"].lower()
                status_badge = "✅ JUSTIFIED" if is_justified else "🚨 UNEXPLAINED (ACTION REQUIRED)"
                note_info = f" (Cited Note: **{s['matched_note_id']}**)" if s["matched_note_id"] else ""
                
                card = (
                    f"### 📍 Route: {s['route']} — Week of {s['week_of']}\n\n"
                    f"- **Status**: {status_badge}{note_info}\n"
                    f"- **Freight Rate**: **₹{s['cost_per_tonne_km']}** / tonne-km\n"
                    f"- **Historical Comparison**: `{s['vs_own_history']}`\n"
                    f"- **Peer Route Comparison**: `{s['vs_similar_routes']}`\n\n"
                    f"**Analysis & Findings**:\n"
                    f"{s['reason']}"
                )
                sections.append(card)
            return "\n\n---\n\n".join(sections)
        else:
            avg_cptk = sum(float(r["cost_per_tonne_km"]) for r in matching_rows) / len(matching_rows)
            return (
                f"### 📍 Route: {route}\n\n"
                f"- **Status**: ✅ **Normal Operations (No Spikes)**\n"
                f"- **Average Rate**: **₹{avg_cptk:.2f}** / tonne-km\n"
                f"- **Monitored Weeks**: {len(matching_rows)} weeks\n\n"
                f"Freight rates for **{route}** remained stable within historical baselines and peer group ranges during this period."
            )

    def _summarize_anomalies(self, text: str) -> str:
        flagged_rows = [r for r in self.records if r["flagged"] != "No"]
        if not flagged_rows:
            return "✅ **All Clear**: No cost anomalies detected in the dataset."
            
        unexplained = [r for r in flagged_rows if r["flagged"] == "Yes"]
        justified = [r for r in flagged_rows if "justified" in r["flagged"].lower()]
        
        output = [
            f"## 📊 Shipping Cost Anomaly Summary ({len(flagged_rows)} Events Detected)\n",
            f"### 🚨 Unexplained Price Spikes ({len(unexplained)} items — Action Required)\n"
        ]
        
        for u in unexplained:
            output.append(
                f"• **{u['route']}** (Week of `{u['week_of']}`)\n"
                f"  - **Rate**: ₹{u['cost_per_tonne_km']}/t-km (`{u['vs_own_history']}`)\n"
                f"  - **Finding**: {u['reason']}\n"
            )
            
        output.append(f"\n### ✅ Justified Cost Increases ({len(justified)} items — Context Verified)\n")
        for j in justified:
            output.append(
                f"• **{j['route']}** (Week of `{j['week_of']}`)\n"
                f"  - **Rate**: ₹{j['cost_per_tonne_km']}/t-km (`{j['vs_own_history']}`)\n"
                f"  - **Note**: [`{j['matched_note_id']}`] {j['reason']}\n"
            )
            
        return "\n".join(output)

    def _answer_general_query(self, query: str, year: Optional[str], month: Optional[int]) -> str:
        # Search across context notes
        for note in self.notes:
            if note.note_id.lower() in query or (note.applies_to.lower() in query and note.applies_to != "All Routes"):
                return (
                    f"### 📋 Context Note: {note.note_id} ({note.date_str})\n\n"
                    f"- **Applies To**: {note.applies_to}\n"
                    f"- **Details**: {note.text}\n"
                    f"- **Active Date Window**: {note.start_date.strftime('%Y-%m-%d')} to {note.end_date.strftime('%Y-%m-%d')}"
                )
                
        return self._summarize_anomalies(query)

    def _extract_route(self, text: str) -> Optional[str]:
        known_routes = [
            "Mumbai-Pune", "Delhi-Jaipur", "Chennai-Bangalore",
            "Ahmedabad-Mumbai", "Kolkata-Bhubaneswar", "Bangalore-Hyderabad",
            "Mumbai-Delhi", "Delhi-Kolkata"
        ]
        for r in known_routes:
            cities = r.lower().split("-")
            if r.lower() in text or (cities[0] in text and cities[1] in text):
                return r
        return None

    def _extract_date_hints(self, text: str) -> Tuple[Optional[str], Optional[int]]:
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

    def start_cli_session(self):
        print("\n" + "="*65)
        print("  🚚 FreightTiger Smart Shipping Cost Assistant (Q&A Mode)")
        print("  Type your question or 'exit' / 'quit' to finish.")
        print("="*65 + "\n")
        
        while True:
            try:
                user_input = input("User Question ❯ ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                    
                answer = self.answer_question(user_input)
                print(f"\n{answer}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    assistant = ShippingAssistantQA()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(assistant.answer_question(query))
    else:
        assistant.start_cli_session()
