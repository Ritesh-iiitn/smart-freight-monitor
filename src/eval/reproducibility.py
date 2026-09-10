"""
Reproducibility Verification Suite.
Executes 3 independent, untouched runs on the same input dataset,
compares all numeric fields, verdicts, matched note IDs, and explanations,
and outputs a diff report confirming deterministic output.
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import csv
import filecmp
from src.pipeline import run_pipeline


def verify_reproducibility(
    shipments_path: str = "data/shipment_records.csv",
    notes_path: str = "data/context_notes.csv",
    temp_dir: str = "output/reproducibility_test"
) -> bool:
    os.makedirs(temp_dir, exist_ok=True)
    
    run_files = []
    print("="*60)
    print("      STARTING REPRODUCIBILITY AUDIT (3 INDEPENDENT RUNS)")
    print("="*60)
    
    for run_idx in range(1, 4):
        out_file = os.path.join(temp_dir, f"run_{run_idx}.csv")
        print(f"Executing Run #{run_idx}...")
        run_pipeline(
            shipments_path=shipments_path,
            notes_path=notes_path,
            output_path=out_file
        )
        run_files.append(out_file)

    print("\nComparing Run 1, Run 2, and Run 3 results...")
    
    # 1. Byte-level file comparison
    match_1_2 = filecmp.cmp(run_files[0], run_files[1], shallow=False)
    match_1_3 = filecmp.cmp(run_files[0], run_files[2], shallow=False)
    
    if match_1_2 and match_1_3:
        print(" [PASS] Exact Byte-for-Byte Match across all 3 runs.")
    else:
        print(" [WARN] Byte-level mismatch detected, performing deep field-by-field inspection...")

    # 2. Detailed field-by-field verification
    rows_run1 = list(csv.DictReader(open(run_files[0], encoding="utf-8")))
    rows_run2 = list(csv.DictReader(open(run_files[1], encoding="utf-8")))
    rows_run3 = list(csv.DictReader(open(run_files[2], encoding="utf-8")))
    
    assert len(rows_run1) == len(rows_run2) == len(rows_run3), "Row count mismatch across runs!"
    
    diff_count = 0
    checked_fields = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id", "reason"]
    
    for i in range(len(rows_run1)):
        r1, r2, r3 = rows_run1[i], rows_run2[i], rows_run3[i]
        for field in checked_fields:
            v1, v2, v3 = r1[field], r2[field], r3[field]
            if not (v1 == v2 == v3):
                diff_count += 1
                print(f" Mismatch at row {i} field '{field}':\n  Run 1: {v1}\n  Run 2: {v2}\n  Run 3: {v3}")

    print("="*60)
    if diff_count == 0:
        print(f" [SUCCESS] 100% REPRODUCIBILITY VERIFIED across {len(rows_run1)} rows x {len(checked_fields)} fields.")
        print(f" - Numbers, verdicts, cited notes, and explanations are 100% identical.")
        print(f" - Total Diff Count: {diff_count}")
        print("="*60)
        return True
    else:
        print(f" [FAILED] Found {diff_count} discrepancies across runs.")
        print("="*60)
        return False


if __name__ == "__main__":
    success = verify_reproducibility()
    if not success:
        sys.exit(1)
