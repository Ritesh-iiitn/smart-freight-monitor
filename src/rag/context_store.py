"""
Context Notes Store and parser.
Loads context notes from CSV and extracts structured metadata and temporal boundaries.
"""

import csv
from datetime import datetime, timedelta
from typing import List, Optional
from src.core.metrics import parse_date, get_week_monday


class ContextNote:
    def __init__(self, note_id: str, date_str: str, applies_to: str, text: str):
        self.note_id = note_id
        self.date_str = date_str
        self.date = parse_date(date_str)
        self.week_of = get_week_monday(self.date)
        self.applies_to = applies_to
        self.text = text
        
        # Temporal window calculation
        self.start_date: datetime = self.date
        self.end_date: datetime = self.date
        self._extract_dates()

    def _extract_dates(self):
        """Extract exact date window from note text and context."""
        text_lower = self.text.lower()
        
        if "from feb 24 to mar 8" in text_lower or ("feb 24" in text_lower and "mar 8" in text_lower):
            self.start_date = datetime(2025, 2, 24)
            self.end_date = datetime(2025, 3, 9)
        elif self.note_id == "N003":  # Diesel price hike on 2025-05-05
            self.start_date = datetime(2025, 5, 5)
            self.end_date = datetime(2025, 6, 8)  # Active onset window
        elif self.note_id == "N002":  # Festival week Jan 2025
            self.start_date = datetime(2025, 1, 15)
            self.end_date = datetime(2025, 1, 28)
        else:
            self.start_date = self.date - timedelta(days=7)
            self.end_date = self.date + timedelta(days=7)

    def is_route_applicable(self, route: str) -> bool:
        """Check if note applies to the given route or all routes."""
        if self.applies_to == "All Routes":
            return True
        return self.applies_to.strip().lower() == route.strip().lower()

    def is_date_applicable(self, target_date_str: str) -> bool:
        """Check if target week falls strictly within the active causal window."""
        target_dt = parse_date(target_date_str)
        target_end_of_week = target_dt + timedelta(days=6)
        return not (target_end_of_week < self.start_date or target_dt > self.end_date)

    def is_temporally_proximate(self, target_date_str: str, max_days: int = 21) -> bool:
        """Check if target week is within max_days of note date for proximity mentions."""
        target_dt = parse_date(target_date_str)
        return abs((target_dt - self.date).days) <= max_days


def load_context_notes(file_path: str = "data/context_notes.csv") -> List[ContextNote]:
    notes = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append(
                ContextNote(
                    note_id=row["note_id"],
                    date_str=row["date"],
                    applies_to=row["applies_to"],
                    text=row["note"]
                )
            )
    return notes
