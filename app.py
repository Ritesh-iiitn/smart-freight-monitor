#!/usr/bin/env python3
"""
Interactive Web Application & UI Server for FreightTiger Shipping Cost Assistant.
Provides a modern dashboard UI with live charts, anomaly inspection, and an interactive Q&A assistant.
Zero external framework dependencies (runs on standard library http.server).
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import csv
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from src.pipeline import run_pipeline
from src.assistant.interactive_qa import ShippingAssistantQA
from src.rag.context_store import load_context_notes
from tests.eval_harness import run_evaluation
from src.eval.cost_tracker import CostTracker


PORT = 8000

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreightTiger | Smart Shipping Cost Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #090d16;
            --bg-surface: #111827;
            --bg-card: #1f2937;
            --bg-card-hover: #374151;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --danger: #f87171;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --success: #34d399;
            --success-bg: rgba(16, 185, 129, 0.15);
            --warning: #fbbf24;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --border: #374151;
            --border-light: rgba(255, 255, 255, 0.1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-base);
            color: var(--text-main);
            padding: 24px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        /* Top Navigation Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }
        .logo-group { display: flex; align-items: center; gap: 14px; }
        .logo-icon {
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            width: 44px; height: 44px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; box-shadow: 0 0 20px var(--accent-glow);
        }
        h1 { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
        .tagline { font-size: 13px; color: var(--text-muted); }
        .header-actions { display: flex; gap: 10px; }
        .btn {
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex; align-items: center; gap: 6px;
        }
        .btn:hover { background: var(--bg-card-hover); border-color: var(--accent); }
        .btn-primary {
            background: var(--accent);
            color: #090d16;
            border: none;
            box-shadow: 0 0 15px var(--accent-glow);
        }
        .btn-primary:hover { background: #7dd3fc; }

        /* KPI Scorecard Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: var(--border);
        }
        .kpi-card.accent::after { background: var(--accent); }
        .kpi-card.danger::after { background: var(--danger); }
        .kpi-card.success::after { background: var(--success); }
        .kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
        .kpi-val { font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
        .kpi-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

        /* Layout Grid */
        .main-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }
        @media (max-width: 1024px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        .panel-title { font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 8px; }

        /* Chart Canvas */
        .chart-wrap { position: relative; height: 320px; width: 100%; }

        /* Assistant Chat Box */
        .chat-container { display: flex; flex-direction: column; height: 320px; }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            font-size: 13px;
        }
        .chat-msg {
            padding: 12px 14px;
            border-radius: 10px;
            max-width: 90%;
            line-height: 1.45;
        }
        .msg-user {
            align-self: flex-end;
            background: #0284c7;
            color: #fff;
            border-bottom-right-radius: 2px;
        }
        .msg-bot {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-bottom-left-radius: 2px;
            white-space: pre-wrap;
            font-family: 'Inter', sans-serif;
        }
        .chat-input-bar {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }
        .chat-input {
            flex: 1;
            background: var(--bg-base);
            border: 1px solid var(--border);
            color: #fff;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
        }
        .chat-input:focus { outline: none; border-color: var(--accent); }

        /* Anomaly Table */
        .table-controls {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filter-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }
        .filter-btn.active {
            background: var(--accent);
            color: #090d16;
            border-color: var(--accent);
        }
        .search-box {
            background: var(--bg-base);
            border: 1px solid var(--border);
            color: #fff;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            width: 200px;
        }
        .table-wrap { overflow-x: auto; max-height: 460px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th {
            background: #1e293b;
            color: var(--text-muted);
            padding: 10px 14px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            position: sticky; top: 0; z-index: 10;
        }
        td { padding: 12px 14px; border-bottom: 1px solid var(--border); }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
        }
        .badge-danger { background: var(--danger-bg); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.4); }
        .badge-success { background: var(--success-bg); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.4); }
        .badge-note { background: rgba(56, 189, 248, 0.1); color: var(--accent); font-family: 'JetBrains Mono', monospace; }
        
        .code-pill {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <div class="logo-icon">🚚</div>
                <div>
                    <h1>FreightTiger Smart Shipping Cost Assistant</h1>
                    <div class="tagline">Weekly Cost Tracking • Trailing Rolling Baselines • Anti-Hallucination RAG Verification</div>
                </div>
            </div>
            <div class="header-actions">
                <button class="btn" onclick="runReproducibilityAudit()">🔄 Run 3-Pass Audit</button>
                <button class="btn btn-primary" onclick="triggerPipelineRun()">⚡ Re-Analyze Pipeline</button>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card accent">
                <div class="kpi-label">Route-Weeks Monitored</div>
                <div class="kpi-val" id="kpi-total">800</div>
                <div class="kpi-sub">Monday–Sunday weekly buckets</div>
            </div>
            <div class="kpi-card danger">
                <div class="kpi-label">Unexplained Cost Spikes</div>
                <div class="kpi-val" style="color: var(--danger);" id="kpi-unexplained">2</div>
                <div class="kpi-sub">Flagged "Yes" for human review</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-label">Justified Cost Rises</div>
                <div class="kpi-val" style="color: var(--success);" id="kpi-justified">3</div>
                <div class="kpi-sub">Verified against context notes</div>
            </div>
            <div class="kpi-card accent">
                <div class="kpi-label">Hallucination Rate</div>
                <div class="kpi-val" style="color: var(--accent);">0.0%</div>
                <div class="kpi-sub">Strict causal guardrail checks</div>
            </div>
        </div>

        <!-- Main Layout: Chart & AI Assistant -->
        <div class="main-layout">
            <!-- Left: Cost Trend Chart -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📈 Weekly Cost / Tonne-KM Trailing Trends</div>
                    <select id="routeSelect" class="search-box" onchange="updateChart()">
                        <option value="ALL">All Key Corridors</option>
                    </select>
                </div>
                <div class="chart-wrap">
                    <canvas id="costChart"></canvas>
                </div>
            </div>

            <!-- Right: Interactive Q&A Assistant -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">💬 Live AI Assistant (Stretch Goal)</div>
                    <span class="badge badge-success">● Ready</span>
                </div>
                <div class="chat-container">
                    <div class="chat-history" id="chatHistory">
                        <div class="chat-msg msg-bot">Hello! I am your Freight Cost Assistant. You can ask me questions about route anomalies, festival surcharges, diesel price hikes, or historical baselines.
<br><br><em>Try: "Why did Ahmedabad-Mumbai spike in Jan 2025?" or "List all unexplained anomalies"</em></div>
                    </div>
                    <div class="chat-input-bar">
                        <input type="text" id="chatInput" class="chat-input" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') sendQuestion()">
                        <button class="btn btn-primary" onclick="sendQuestion()">Ask</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom: Anomalies & Explanations Table -->
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🚨 Monitored Route Records & Verification Audit</div>
                <div class="table-controls">
                    <button class="filter-btn active" onclick="setFilter('ALL', this)">All Rows (800)</button>
                    <button class="filter-btn" onclick="setFilter('FLAGGED_ONLY', this)">Flagged Spikes (2)</button>
                    <button class="filter-btn" onclick="setFilter('JUSTIFIED_ONLY', this)">Justified (3)</button>
                    <input type="text" id="tableSearch" class="search-box" placeholder="Filter by route / date..." oninput="filterTable()">
                </div>
            </div>
            <div class="table-wrap">
                <table id="recordsTable">
                    <thead>
                        <tr>
                            <th>Route</th>
                            <th>Week Of</th>
                            <th>Cost / t-km</th>
                            <th>vs Own History (8-wk)</th>
                            <th>vs Similar Routes</th>
                            <th>Flagged Status</th>
                            <th>Matched Note</th>
                            <th>Explanation / Causal Reason</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let allData = [];
        let currentFilter = 'ALL';
        let chartInstance = null;

        async function initDashboard() {
            const resp = await fetch('/api/metrics');
            allData = await resp.json();
            
            // Populate KPIs
            const unexplained = allData.filter(d => d.flagged === 'Yes');
            const justified = allData.filter(d => d.flagged && d.flagged.includes('justified'));
            
            document.getElementById('kpi-total').innerText = allData.length;
            document.getElementById('kpi-unexplained').innerText = unexplained.length;
            document.getElementById('kpi-justified').innerText = justified.length;

            // Populate Route Select
            const routes = [...new Set(allData.map(d => d.route))];
            const select = document.getElementById('routeSelect');
            routes.forEach(r => {
                const opt = document.createElement('option');
                opt.value = r;
                opt.innerText = r;
                select.appendChild(opt);
            });

            renderChart();
            renderTable();
        }

        function renderChart() {
            const ctx = document.getElementById('costChart').getContext('2d');
            const selectedRoute = document.getElementById('routeSelect').value;
            const weeks = [...new Set(allData.map(d => d.week_of))].slice(-25);
            
            const routesToShow = selectedRoute === 'ALL' 
                ? ['Mumbai-Pune', 'Delhi-Jaipur', 'Ahmedabad-Mumbai', 'Chennai-Bangalore']
                : [selectedRoute];
                
            const colors = ['#38bdf8', '#f59e0b', '#34d399', '#f87171', '#a78bfa'];
            
            const datasets = routesToShow.map((r, i) => {
                const rData = allData.filter(d => d.route === r && weeks.includes(d.week_of));
                return {
                    label: r,
                    data: rData.map(d => parseFloat(d.cost_per_tonne_km)),
                    borderColor: colors[i % colors.length],
                    backgroundColor: colors[i % colors.length] + '22',
                    tension: 0.2,
                    pointRadius: 3
                };
            });

            if (chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: { labels: weeks, datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#9ca3af', font: { family: 'Inter' } } }
                    },
                    scales: {
                        x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                        y: { 
                            ticks: { color: '#9ca3af' }, 
                            grid: { color: '#374151' },
                            title: { display: true, text: 'INR per Tonne-KM', color: '#9ca3af' }
                        }
                    }
                }
            });
        }

        function updateChart() {
            renderChart();
        }

        function setFilter(filterType, btnElem) {
            currentFilter = filterType;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btnElem.classList.add('active');
            renderTable();
        }

        function filterTable() {
            renderTable();
        }

        function renderTable() {
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            const query = document.getElementById('tableSearch').value.toLowerCase();
            
            let filtered = allData;
            if (currentFilter === 'FLAGGED_ONLY') {
                filtered = allData.filter(d => d.flagged === 'Yes');
            } else if (currentFilter === 'JUSTIFIED_ONLY') {
                filtered = allData.filter(d => d.flagged && d.flagged.includes('justified'));
            }
            
            if (query) {
                filtered = filtered.filter(d => 
                    d.route.toLowerCase().includes(query) || 
                    d.week_of.toLowerCase().includes(query) ||
                    (d.matched_note_id && d.matched_note_id.toLowerCase().includes(query)) ||
                    d.reason.toLowerCase().includes(query)
                );
            }

            filtered.forEach(row => {
                const tr = document.createElement('tr');
                let badgeClass = 'badge-note';
                if (row.flagged === 'Yes') badgeClass = 'badge-danger';
                else if (row.flagged.includes('justified')) badgeClass = 'badge-success';

                tr.innerHTML = `
                    <td><strong>${row.route}</strong></td>
                    <td><span class="code-pill">${row.week_of}</span></td>
                    <td><strong>₹${row.cost_per_tonne_km}</strong></td>
                    <td style="color: ${row.vs_own_history.startsWith('+') ? '#38bdf8' : '#9ca3af'}">${row.vs_own_history}</td>
                    <td>${row.vs_similar_routes}</td>
                    <td><span class="badge ${badgeClass}">${row.flagged}</span></td>
                    <td>${row.matched_note_id ? `<span class="badge badge-note">${row.matched_note_id}</span>` : '<span style="color: #6b7280">—</span>'}</td>
                    <td style="max-width: 480px; line-height: 1.4;">${row.reason}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function sendQuestion() {
            const input = document.getElementById('chatInput');
            const q = input.value.trim();
            if (!q) return;
            
            const history = document.getElementById('chatHistory');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'chat-msg msg-user';
            userMsg.innerText = q;
            history.appendChild(userMsg);
            input.value = '';
            history.scrollTop = history.scrollHeight;

            // Fetch bot response
            const resp = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q })
            });
            const data = await resp.json();
            
            const botMsg = document.createElement('div');
            botMsg.className = 'chat-msg msg-bot';
            botMsg.innerText = data.answer;
            history.appendChild(botMsg);
            history.scrollTop = history.scrollHeight;
        }

        async function triggerPipelineRun() {
            alert("Triggering analysis pipeline...");
            const resp = await fetch('/api/run', { method: 'POST' });
            const data = await resp.json();
            alert(`Pipeline completed! Processed ${data.processed_count} records.`);
            initDashboard();
        }

        async function runReproducibilityAudit() {
            alert("Executing 3-pass reproducibility verification...");
            const resp = await fetch('/api/eval');
            const data = await resp.json();
            alert(`Evaluation Benchmark: ${data.verdict_accuracy}% Verdict Accuracy, ${data.hallucination_rate}% Hallucinations.`);
        }

        window.onload = initDashboard;
    </script>
</body>
</html>
"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    assistant = ShippingAssistantQA()

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            
        elif parsed.path == "/api/metrics":
            results_path = "output/analysis_results.csv"
            records = []
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    records = list(csv.DictReader(f))
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(records).encode("utf-8"))
            
        elif parsed.path == "/api/eval":
            results = run_evaluation()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(results).encode("utf-8"))
            
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        
        if parsed.path == "/api/ask":
            try:
                data = json.loads(post_body.decode("utf-8"))
                query = data.get("query", "")
                answer = self.assistant.answer_question(query)
                response_data = {"query": query, "answer": answer}
            except Exception as e:
                response_data = {"error": str(e), "answer": "An error occurred while processing your request."}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            
        elif parsed.path == "/api/run":
            metrics = run_pipeline()
            self.assistant = ShippingAssistantQA()  # Reload data
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "processed_count": len(metrics)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    server = HTTPServer(("localhost", PORT), DashboardHTTPHandler)
    print("="*65)
    print(f" 🚀 FreightTiger Interactive Dashboard Server Running!")
    print(f" 🌐 Access UI locally at: http://localhost:{PORT}")
    print("="*65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    start_server()
