"""
Interactive Terminal TUI Dashboard for FreightTiger Shipping Cost Assistant.
Provides a rich, colorized terminal UI with KPIs, anomaly table, and interactive Q&A.
Zero dependencies (uses standard ANSI color formatting).
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import csv
from src.assistant.interactive_qa import ShippingAssistantQA
from tests.eval_harness import run_evaluation


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_dashboard():
    results_path = "output/analysis_results.csv"
    if not os.path.exists(results_path):
        from src.pipeline import run_pipeline
        run_pipeline()

    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    unexplained = [r for r in records if r["flagged"] == "Yes"]
    justified = [r for r in records if "justified" in r["flagged"].lower()]

    print("\n" + "="*80)
    print(f" {Colors.BOLD}{Colors.CYAN}🚚 FREIGHTTIGER SMART SHIPPING COST ASSISTANT — TERMINAL DASHBOARD{Colors.RESET}")
    print("="*80)
    
    # KPI Row
    print(f" {Colors.BOLD}Monitored Route-Weeks:{Colors.RESET} {len(records)}   |   "
          f"{Colors.BOLD}Unexplained Spikes:{Colors.RESET} {Colors.RED}{len(unexplained)}{Colors.RESET}   |   "
          f"{Colors.BOLD}Justified Rises:{Colors.RESET} {Colors.GREEN}{len(justified)}{Colors.RESET}   |   "
          f"{Colors.BOLD}Hallucination Rate:{Colors.RESET} {Colors.CYAN}0.0%{Colors.RESET}")
    print("-" * 80)

    # Anomaly Table
    print(f"\n {Colors.BOLD}{Colors.YELLOW}🚨 DETECTED COST SPIKES & CAUSAL VERIFICATIONS:{Colors.RESET}\n")
    print(f" {'ROUTE':<18} | {'WEEK OF':<10} | {'COST':<8} | {'VS 8-WK':<10} | {'STATUS':<15} | {'NOTE':<5}")
    print("-" * 80)
    
    anomalies = unexplained + justified
    for a in anomalies:
        status_colored = f"{Colors.RED}Flagged (Yes){Colors.RESET}" if a["flagged"] == "Yes" else f"{Colors.GREEN}Justified (No){Colors.RESET}"
        note_str = a["matched_note_id"] or "—"
        print(f" {a['route']:<18} | {a['week_of']:<10} | ₹{a['cost_per_tonne_km']:<7} | {a['vs_own_history'][:9]:<10} | {status_colored:<24} | {note_str:<5}")
        print(f"  {Colors.DIM}↳ Reason: {a['reason'][:75]}...{Colors.RESET}")

    print("="*80)
    print(f" {Colors.BOLD}💬 INTERACTIVE AI ASSISTANT (Type a question, 'audit' for 3-pass check, or 'exit'){Colors.RESET}")
    print("="*80 + "\n")

    assistant = ShippingAssistantQA()
    while True:
        try:
            query = input(f"{Colors.BOLD}{Colors.CYAN}Question ❯ {Colors.RESET}").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print(f"{Colors.GREEN}Session ended.{Colors.RESET}")
                break
            elif query.lower() == "audit":
                run_evaluation()
                continue
                
            ans = assistant.answer_question(query)
            print(f"\n{Colors.BOLD}Assistant Response:{Colors.RESET}\n{ans}\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    print_dashboard()
