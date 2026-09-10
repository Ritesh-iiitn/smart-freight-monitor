"""
Unit tests for metrics computation and baseline logic.
"""

import unittest
from datetime import datetime
from src.core.metrics import (
    get_week_monday, parse_date, ShipmentRecord, WeeklyRouteMetric,
    aggregate_weekly_metrics, compute_baselines
)
from src.core.anomaly_detector import AnomalyDetector


class TestMetricsLogic(unittest.TestCase):
    def test_week_grouping(self):
        # 2024-01-01 was a Monday
        self.assertEqual(get_week_monday(datetime(2024, 1, 1)), "2024-01-01")
        # 2024-01-06 was a Saturday -> Monday was 2024-01-01
        self.assertEqual(get_week_monday(datetime(2024, 1, 6)), "2024-01-01")
        # 2024-01-07 was a Sunday -> Monday was 2024-01-01
        self.assertEqual(get_week_monday(datetime(2024, 1, 7)), "2024-01-01")
        # 2024-01-08 was next Monday
        self.assertEqual(get_week_monday(datetime(2024, 1, 8)), "2024-01-08")

    def test_cost_per_tonne_km_and_trailing_history(self):
        records = []
        # Create 10 consecutive weeks for Route A
        for i in range(10):
            day_str = f"2024-0{i+1}-01" if i < 9 else "2024-10-01"
            monday_str = get_week_monday(parse_date(day_str))
            
            # Base cost 3.00 for weeks 0..8, spike to 4.50 in week 9
            cptk = 4.50 if i == 9 else 3.00
            qty = 10.0
            dist = 100.0
            cost = cptk * qty * dist
            
            records.append(
                ShipmentRecord(
                    shipment_id=f"S{i}",
                    origin="Mumbai",
                    destination="Pune",
                    route_type="Short",
                    material="Steel",
                    quantity_tonnes=qty,
                    distance_km=dist,
                    freight_cost_inr=cost,
                    shipment_date=monday_str,
                    transporter="Shivam"
                )
            )
            
        metrics_dict = aggregate_weekly_metrics(records)
        processed = compute_baselines(metrics_dict)
        
        # Sort by week_of
        processed.sort(key=lambda x: x.week_of)
        
        # Check first week (no prior history)
        self.assertEqual(processed[0].own_history_weeks_count, 0)
        self.assertEqual(processed[0].vs_own_history_str, "+0.0% vs this route's past average")
        
        # Check week 9 (should use exactly 8 trailing weeks strictly prior)
        week9 = processed[9]
        self.assertEqual(week9.own_history_weeks_count, 8)
        self.assertAlmostEqual(week9.own_history_avg, 3.00, places=2)
        self.assertAlmostEqual(week9.cost_per_tonne_km, 4.50, places=2)
        self.assertEqual(week9.vs_own_history_str, "+50.0% vs this route's past average")

    def test_peer_routes_exclusion(self):
        # 3 Short routes in the same week
        week_str = "2024-02-05"
        records = [
            ShipmentRecord("S1", "Mumbai", "Pune", "Short", "Steel", 10.0, 100.0, 3000.0, week_str, "T1"), # cptk = 3.0
            ShipmentRecord("S2", "Delhi", "Jaipur", "Short", "Steel", 10.0, 100.0, 4000.0, week_str, "T2"), # cptk = 4.0
            ShipmentRecord("S3", "Chennai", "Bangalore", "Short", "Steel", 10.0, 100.0, 2000.0, week_str, "T3"), # cptk = 2.0
        ]
        metrics_dict = aggregate_weekly_metrics(records)
        processed = compute_baselines(metrics_dict)
        
        mumbai = next(m for m in processed if m.route == "Mumbai-Pune")
        # Peers for Mumbai are Delhi-Jaipur (4.0) and Chennai-Bangalore (2.0) -> avg = 3.0
        self.assertEqual(mumbai.peer_routes_count, 2)
        self.assertAlmostEqual(mumbai.peer_avg, 3.0, places=2)
        self.assertEqual(mumbai.vs_similar_routes_str, "+0.0% vs similar-length routes this week")

        delhi = next(m for m in processed if m.route == "Delhi-Jaipur")
        # Peers for Delhi are Mumbai-Pune (3.0) and Chennai-Bangalore (2.0) -> avg = 2.5
        # Delhi is 4.0 -> diff = (4.0 - 2.5) / 2.5 = +60.0%
        self.assertEqual(delhi.peer_routes_count, 2)
        self.assertAlmostEqual(delhi.peer_avg, 2.5, places=2)
        self.assertEqual(delhi.vs_similar_routes_str, "+60.0% vs similar-length routes this week")

    def test_anomaly_detector(self):
        detector = AnomalyDetector(own_history_threshold_pct=8.0, similar_routes_threshold_pct=15.0)
        
        m1 = WeeklyRouteMetric("RouteA", "Short", "2024-01-01", 1000, 100, 1)
        m1.vs_own_history_pct = 12.0
        m1.vs_similar_routes_pct = 2.0
        self.assertTrue(detector.is_cost_spike(m1))
        
        m2 = WeeklyRouteMetric("RouteB", "Short", "2024-01-01", 1000, 100, 1)
        m2.vs_own_history_pct = 2.0
        m2.vs_similar_routes_pct = 3.0
        self.assertFalse(detector.is_cost_spike(m2))


if __name__ == "__main__":
    unittest.main()
