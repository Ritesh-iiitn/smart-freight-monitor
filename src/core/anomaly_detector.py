"""
Anomaly detection engine for identifying rising and unusual shipping costs.
Flags routes exceeding historical trailing averages or peer group baselines.
"""

from typing import List
from src.core.metrics import WeeklyRouteMetric


class AnomalyDetector:
    def __init__(
        self,
        own_history_threshold_pct: float = 8.0,
        similar_routes_threshold_pct: float = 15.0,
        min_combined_threshold_pct: float = 5.0
    ):
        """
        :param own_history_threshold_pct: Minimum % increase over 8-week history to consider flagged.
        :param similar_routes_threshold_pct: Minimum % increase over peer routes in same week.
        :param min_combined_threshold_pct: Floor for own history when peer route threshold is met.
        """
        self.own_history_threshold_pct = own_history_threshold_pct
        self.similar_routes_threshold_pct = similar_routes_threshold_pct
        self.min_combined_threshold_pct = min_combined_threshold_pct

    def is_cost_spike(self, metric: WeeklyRouteMetric) -> bool:
        """
        Determine if the metric indicates a noteworthy cost spike requiring verification.
        A cost is flagged as a candidate spike if:
        1. It is rising significantly vs its own trailing history (e.g. >= +8%), OR
        2. It is significantly higher than similar routes in the same week (e.g. >= +15%) and not falling vs history.
        """
        own_pct = metric.vs_own_history_pct if metric.vs_own_history_pct is not None else 0.0
        peer_pct = metric.vs_similar_routes_pct if metric.vs_similar_routes_pct is not None else 0.0
        
        # Primary condition: Strong spike over historical trailing baseline
        if own_pct >= self.own_history_threshold_pct:
            return True
            
        # Secondary condition: Substantially higher than peers while also creeping up vs own history
        if peer_pct >= self.similar_routes_threshold_pct and own_pct >= self.min_combined_threshold_pct:
            return True
            
        return False

    def identify_anomalies(self, metrics: List[WeeklyRouteMetric]) -> List[WeeklyRouteMetric]:
        """Return list of metrics that meet the cost spike criteria."""
        return [m for m in metrics if self.is_cost_spike(m)]
