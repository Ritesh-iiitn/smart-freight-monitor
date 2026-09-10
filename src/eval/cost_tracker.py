"""
Token and Cost Tracking Module.
Logs LLM / RAG calls, tokens consumed, latency, and estimated pricing across model providers.
"""

import time
from typing import Dict, Any, List


# Reference pricing per 1M tokens (USD)
PRICING_TABLE = {
    "local_opensource": {"input": 0.00, "output": 0.00},
    "gemini_flash": {"input": 0.075, "output": 0.30},
    "gpt_4o_mini": {"input": 0.15, "output": 0.60},
}


class CostTracker:
    def __init__(self, model_name: str = "local_opensource"):
        self.model_name = model_name
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0
        self.start_time: float = time.time()
        self.logs: List[Dict[str, Any]] = []

    def record_call(self, prompt_text: str, response_text: str, latency_ms: float = 0.0):
        """Estimate token counts and record call."""
        # Standard rough estimate: ~4 chars per token for English text
        in_tokens = max(1, len(prompt_text) // 4)
        out_tokens = max(1, len(response_text) // 4)
        
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        self.call_count += 1
        
        self.logs.append({
            "call_index": self.call_count,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "latency_ms": latency_ms
        })

    def get_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        rates = PRICING_TABLE.get(self.model_name, PRICING_TABLE["local_opensource"])
        
        cost_input = (self.total_input_tokens / 1_000_000) * rates["input"]
        cost_output = (self.total_output_tokens / 1_000_000) * rates["output"]
        total_cost = cost_input + cost_output
        
        return {
            "model_name": self.model_name,
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(total_cost, 6),
            "estimated_cost_inr": round(total_cost * 87.0, 4),
            "elapsed_seconds": round(elapsed, 3)
        }

    def print_report(self):
        summary = self.get_summary()
        print("\n" + "="*50)
        print("         TOKEN AND COST AUDIT REPORT")
        print("="*50)
        print(f" Model Engine          : {summary['model_name']}")
        print(f" Total LLM/RAG Calls   : {summary['total_calls']}")
        print(f" Input Tokens          : {summary['total_input_tokens']:,}")
        print(f" Output Tokens         : {summary['total_output_tokens']:,}")
        print(f" Total Tokens          : {summary['total_tokens']:,}")
        print(f" Estimated Cost (USD)  : ${summary['estimated_cost_usd']:.6f}")
        print(f" Estimated Cost (INR)  : ₹{summary['estimated_cost_inr']:.4f}")
        print(f" Execution Duration    : {summary['elapsed_seconds']}s")
        print("="*50 + "\n")
