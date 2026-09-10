#!/usr/bin/env python3
"""
Synthetic Shipment Records Generator
Generates realistic shipment records for FreightTiger cost monitoring.
Covers 2024-01-01 to 2025-11-30 across Short, Medium, and Long routes.
Includes controlled price events that map directly to context_notes.csv as well as unexplained anomalies.
"""

import csv
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Ensure deterministic generation
random.seed(42)

ROUTES_CONFIG = [
    {"origin": "Mumbai", "destination": "Pune", "route_type": "Short", "distance_km": 150.0, "base_cptk": 3.00},
    {"origin": "Delhi", "destination": "Jaipur", "route_type": "Short", "distance_km": 280.0, "base_cptk": 2.95},
    {"origin": "Chennai", "destination": "Bangalore", "route_type": "Short", "distance_km": 350.0, "base_cptk": 2.75},
    {"origin": "Ahmedabad", "destination": "Mumbai", "route_type": "Medium", "distance_km": 530.0, "base_cptk": 2.45},
    {"origin": "Kolkata", "destination": "Bhubaneswar", "route_type": "Medium", "distance_km": 440.0, "base_cptk": 2.50},
    {"origin": "Bangalore", "destination": "Hyderabad", "route_type": "Medium", "distance_km": 570.0, "base_cptk": 2.40},
    {"origin": "Mumbai", "destination": "Delhi", "route_type": "Long", "distance_km": 1420.0, "base_cptk": 2.05},
    {"origin": "Delhi", "destination": "Kolkata", "route_type": "Long", "distance_km": 1510.0, "base_cptk": 2.00},
]

MATERIALS = ["Packaged Foods", "Steel Coils", "Textiles", "Automobile Parts", "Chemicals", "Electronics", "FMCG"]
TRANSPORTERS = ["Shivam Freight Co", "Om Sai Transport", "VRL Logistics", "SafeExpress", "TCI Freight", "Western Carriers"]

def get_monday(dt: datetime) -> datetime:
    return dt - timedelta(days=dt.weekday())

def generate_shipments(start_date: str = "2024-01-01", end_date: str = "2025-11-30", output_file: str = "data/shipment_records.csv") -> None:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    shipments: List[Dict[str, Any]] = []
    shipment_counter = 1
    
    current_dt = start_dt
    while current_dt <= end_dt:
        week_monday = get_monday(current_dt)
        week_str = week_monday.strftime("%Y-%m-%d")
        
        for route in ROUTES_CONFIG:
            origin = route["origin"]
            dest = route["destination"]
            route_name = f"{origin}-{dest}"
            base_cptk = route["base_cptk"]
            
            # Default organic weekly fluctuation
            multiplier = 1.0 + random.uniform(-0.02, 0.02)
            
            # Route-specific events checked FIRST
            if route_name == "Delhi-Jaipur" and week_str == "2024-11-11":
                multiplier = 1.41  # +35-40% spike (Unexplained)
                
            elif route_name == "Ahmedabad-Mumbai" and week_str == "2025-01-20":
                multiplier = 1.34  # +29-34% festival spike (N002)
                
            elif route_name == "Chennai-Bangalore" and week_str in ["2025-02-24", "2025-03-03"]:
                multiplier = 1.38  # +30-38% flood disruption (N001)
                
            elif route_name == "Mumbai-Pune" and week_str == "2025-09-15":
                multiplier = 1.32  # Spike on week 2025-09-15, close to N006 report (Unexplained)
                
            elif route_name == "Mumbai-Delhi" and week_str == "2024-07-29":
                multiplier = 1.01  # N005 minor maintenance (costs unaffected)
                
            elif week_str >= "2025-05-05":
                multiplier += 0.06  # Nationwide diesel price hike (N003)
                
            num_shipments = random.randint(1, 3)
            for _ in range(num_shipments):
                dist = round(route["distance_km"] + random.uniform(-4.0, 4.0), 1)
                qty = round(random.uniform(10.0, 25.0), 1)
                
                shipment_cptk = base_cptk * multiplier * (1.0 + random.uniform(-0.015, 0.015))
                freight_cost = int(round(shipment_cptk * qty * dist))
                
                shipment_id = f"SHP{shipment_counter:05d}"
                shipment_counter += 1
                
                shipments.append({
                    "shipment_id": shipment_id,
                    "origin": origin,
                    "destination": dest,
                    "route_type": route["route_type"],
                    "material": random.choice(MATERIALS),
                    "quantity_tonnes": qty,
                    "distance_km": dist,
                    "freight_cost_inr": freight_cost,
                    "shipment_date": current_dt.strftime("%Y-%m-%d"),
                    "transporter": random.choice(TRANSPORTERS)
                })
        
        current_dt += timedelta(days=1)
        
    fieldnames = [
        "shipment_id", "origin", "destination", "route_type",
        "material", "quantity_tonnes", "distance_km", "freight_cost_inr",
        "shipment_date", "transporter"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shipments)
        
    print(f"Generated {len(shipments)} shipment records -> {output_file}")

if __name__ == "__main__":
    generate_shipments()
