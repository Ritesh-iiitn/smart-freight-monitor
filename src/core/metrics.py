"""
Core metrics computation module for FreightTiger Shipping Cost Assistant.
Handles weekly aggregation, cost per tonne-km calculation, trailing 8-week baseline,
and peer route baseline comparisons according to strict competition specifications.
"""

import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


def parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD date string."""
    return datetime.strptime(date_str.strip(), "%Y-%m-%d")


def get_week_monday(dt: datetime) -> str:
    """Return the Monday date string (YYYY-MM-DD) for the given date (Monday to Sunday window)."""
    monday_dt = dt - timedelta(days=dt.weekday())
    return monday_dt.strftime("%Y-%m-%d")


class ShipmentRecord:
    __slots__ = (
        "shipment_id", "origin", "destination", "route", "route_type",
        "material", "quantity_tonnes", "distance_km", "freight_cost_inr",
        "shipment_date", "week_of", "transporter"
    )

    def __init__(
        self,
        shipment_id: str,
        origin: str,
        destination: str,
        route_type: str,
        material: str,
        quantity_tonnes: float,
        distance_km: float,
        freight_cost_inr: float,
        shipment_date: str,
        transporter: str
    ):
        self.shipment_id = shipment_id
        self.origin = origin
        self.destination = destination
        self.route = f"{origin}-{destination}"
        self.route_type = route_type
        self.material = material
        self.quantity_tonnes = quantity_tonnes
        self.distance_km = distance_km
        self.freight_cost_inr = freight_cost_inr
        self.shipment_date = shipment_date
        self.week_of = get_week_monday(parse_date(shipment_date))
        self.transporter = transporter


class WeeklyRouteMetric:
    def __init__(
        self,
        route: str,
        route_type: str,
        week_of: str,
        total_cost: float,
        total_tonne_km: float,
        shipment_count: int
    ):
        self.route = route
        self.route_type = route_type
        self.week_of = week_of
        self.total_cost = total_cost
        self.total_tonne_km = total_tonne_km
        self.shipment_count = shipment_count
        
        # Cost per tonne per km = total freight cost / (quantity in tonnes * distance in km)
        self.cost_per_tonne_km = round(total_cost / total_tonne_km, 2) if total_tonne_km > 0 else 0.0
        
        # Computed baselines
        self.own_history_avg: Optional[float] = None
        self.own_history_weeks_count: int = 0
        self.vs_own_history_pct: Optional[float] = None
        self.vs_own_history_str: str = "N/A"
        
        self.peer_avg: Optional[float] = None
        self.peer_routes_count: int = 0
        self.vs_similar_routes_pct: Optional[float] = None
        self.vs_similar_routes_str: str = "N/A"
        
        # Flagging status & AI Reason
        self.flagged: str = "No"  # "Yes", "No (justified)", "No"
        self.matched_note_id: str = ""
        self.reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "week_of": self.week_of,
            "cost_per_tonne_km": f"{self.cost_per_tonne_km:.2f}",
            "vs_own_history": self.vs_own_history_str,
            "vs_similar_routes": self.vs_similar_routes_str,
            "flagged": self.flagged,
            "matched_note_id": self.matched_note_id,
            "reason": self.reason
        }


def load_shipment_records(file_path: str) -> List[ShipmentRecord]:
    """Load shipment records from CSV file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ShipmentRecord(
                    shipment_id=row["shipment_id"],
                    origin=row["origin"],
                    destination=row["destination"],
                    route_type=row["route_type"],
                    material=row.get("material", ""),
                    quantity_tonnes=float(row["quantity_tonnes"]),
                    distance_km=float(row["distance_km"]),
                    freight_cost_inr=float(row["freight_cost_inr"]),
                    shipment_date=row["shipment_date"],
                    transporter=row.get("transporter", "")
                )
            )
    return records


def aggregate_weekly_metrics(records: List[ShipmentRecord]) -> Dict[Tuple[str, str], WeeklyRouteMetric]:
    """
    Aggregate shipments by (route, week_of).
    Returns mapping from (route, week_of) to WeeklyRouteMetric.
    """
    agg = defaultdict(lambda: {"total_cost": 0.0, "total_tonne_km": 0.0, "count": 0, "route_type": ""})
    
    for rec in records:
        key = (rec.route, rec.week_of)
        agg[key]["total_cost"] += rec.freight_cost_inr
        agg[key]["total_tonne_km"] += (rec.quantity_tonnes * rec.distance_km)
        agg[key]["count"] += 1
        agg[key]["route_type"] = rec.route_type
        
    metrics = {}
    for (route, week_of), data in agg.items():
        metrics[(route, week_of)] = WeeklyRouteMetric(
            route=route,
            route_type=data["route_type"],
            week_of=week_of,
            total_cost=data["total_cost"],
            total_tonne_km=data["total_tonne_km"],
            shipment_count=data["count"]
        )
    return metrics


def compute_baselines(metrics: Dict[Tuple[str, str], WeeklyRouteMetric]) -> List[WeeklyRouteMetric]:
    """
    Compute comparison baselines strictly according to challenge specification:
    1. vs. own history: Trailing 8-week rolling average cost per tonne-km, strictly prior to current week.
       If <8 prior weeks exist, use all prior available weeks.
    2. vs. similar routes: Average cost per tonne-km in that same week across all other routes sharing
       the same route_type (Short / Medium / Long), excluding the route itself.
    """
    # Group metrics by route sorted chronologically by week_of
    route_history: Dict[str, List[WeeklyRouteMetric]] = defaultdict(list)
    week_type_metrics: Dict[Tuple[str, str], List[WeeklyRouteMetric]] = defaultdict(list)
    
    # Sort all entries by week_of
    all_metrics = sorted(metrics.values(), key=lambda m: (m.week_of, m.route))
    
    for m in all_metrics:
        route_history[m.route].append(m)
        week_type_metrics[(m.week_of, m.route_type)].append(m)
        
    # 1. Compute vs. own history
    for route, history in route_history.items():
        # Ensure chronological order
        history.sort(key=lambda x: x.week_of)
        for i, curr in enumerate(history):
            if i == 0:
                # First available week: no strictly prior history
                curr.own_history_avg = None
                curr.own_history_weeks_count = 0
                curr.vs_own_history_pct = 0.0
                curr.vs_own_history_str = "+0.0% vs this route's past average"
            else:
                # Trailing up to 8 strictly prior weeks
                start_idx = max(0, i - 8)
                prior_slice = history[start_idx:i]
                curr.own_history_weeks_count = len(prior_slice)
                prior_avg = sum(p.cost_per_tonne_km for p in prior_slice) / len(prior_slice)
                curr.own_history_avg = round(prior_avg, 4)
                
                pct_diff = ((curr.cost_per_tonne_km - prior_avg) / prior_avg) * 100.0
                curr.vs_own_history_pct = round(pct_diff, 1)
                sign = "+" if pct_diff >= 0 else ""
                curr.vs_own_history_str = f"{sign}{pct_diff:.1f}% vs this route's past average"

    # 2. Compute vs. similar routes (same week, same route_type, excluding self)
    for (week_of, route_type), peer_list in week_type_metrics.items():
        for curr in peer_list:
            peers = [p for p in peer_list if p.route != curr.route]
            if not peers:
                curr.peer_avg = None
                curr.peer_routes_count = 0
                curr.vs_similar_routes_pct = 0.0
                curr.vs_similar_routes_str = "+0.0% vs similar-length routes this week"
            else:
                curr.peer_routes_count = len(peers)
                peer_avg = sum(p.cost_per_tonne_km for p in peers) / len(peers)
                curr.peer_avg = round(peer_avg, 4)
                
                pct_diff = ((curr.cost_per_tonne_km - peer_avg) / peer_avg) * 100.0
                curr.vs_similar_routes_pct = round(pct_diff, 1)
                sign = "+" if pct_diff >= 0 else ""
                curr.vs_similar_routes_str = f"{sign}{pct_diff:.1f}% vs similar-length routes this week"

    return all_metrics
