
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# =============================================
# D-WAVE QUANTUM FINANCE V9
# FASTAPI + INTERACTIVE DASHBOARD
# =============================================

BASE = Path(__file__).resolve().parent

V7_FILE = BASE / "results_v7.json"
V8_FILE = BASE / "multi_agent_report_v8.json"
EQUITY_FILE = BASE / "equity_curve_v7.csv"

app = FastAPI(
    title="D-Wave Quantum Finance API",
    version="9.0",
    description="Quantum-inspired portfolio research dashboard"
)


def load_json(path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Missing file: {path.name}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


def load_equity():
    if not EQUITY_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Missing equity_curve_v7.csv"
        )

    df = pd.read_csv(EQUITY_FILE)

    required = {"Date", "SA", "Exact"}

    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Invalid equity curve columns"
        )

    df = df.dropna(subset=["SA", "Exact"])

    return df.to_dict(orient="records")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "V9",
        "files": {
            "v7": V7_FILE.exists(),
            "v8": V8_FILE.exists(),
            "equity": EQUITY_FILE.exists()
        }
    }


@app.get("/api/v7")
def api_v7():
    return load_json(V7_FILE)


@app.get("/api/v8")
def api_v8():
    return load_json(V8_FILE)


@app.get("/api/equity")
def api_equity():
    return load_equity()


@app.get("/api/summary")
def api_summary():
    v7 = load_json(V7_FILE)
    v8 = load_json(V8_FILE)

    agents = v8.get("agents", {})

    return {
        "version": "V9",
        "solver": "Simulated Annealing",
        "completed_windows": v7.get(
            "completed_windows", 0
        ),
        "sa": v7.get("sa_metrics", {}),
        "exact": v7.get("exact_metrics", {}),
        "market": agents.get("Market Agent", {}),
        "risk": agents.get("Risk Agent", {}),
        "portfolio": agents.get(
            "Portfolio Agent", {}
        ),
        "findings": v8.get(
            "coordinator_findings", []
        )
    }


DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>D-Wave Quantum Finance V9</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #080e1d;
    color: #eaf0ff;
    font-family: Arial, sans-serif;
}

header {
    padding: 28px 5%;
    background: linear-gradient(
        110deg, #111d3c, #102e46
    );
    border-bottom: 1px solid #273b5a;
}

h1 {
    margin: 0 0 8px;
    font-size: 30px;
}

.subtitle {
    color: #a5b6d2;
}

main {
    max-width: 1300px;
    margin: auto;
    padding: 24px;
}

.cards {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(190px, 1fr));
    gap: 16px;
}

.card, .panel {
    background: #111c32;
    border: 1px solid #263653;
    border-radius: 15px;
    padding: 20px;
}

.label {
    color: #a4b4d0;
    font-size: 13px;
}

.value {
    font-size: 27px;
    font-weight: bold;
    margin-top: 10px;
    color: #69dfb3;
}

section {
    margin-top: 24px;
}

h2 {
    font-size: 20px;
}

.chart-wrap {
    height: 370px;
    position: relative;
}

.agent-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(270px, 1fr));
    gap: 16px;
}

.agent-title {
    color: #8fc6ff;
    font-weight: bold;
}

ul {
    padding-left: 20px;
    line-height: 1.7;
}

.note {
    color: #a4b4d0;
    font-size: 13px;
    line-height: 1.6;
}

.error {
    color: #ff9b9b;
}

footer {
    text-align: center;
    padding: 30px;
    color: #8a9bb8;
}
</style>
</head>

<body>

<header>
<h1>⚛ Quantum Finance Research Lab</h1>
<div class="subtitle">
D-Wave V9 | Portfolio Optimization |
Risk Management | Multi-Agent Research
</div>
</header>

<main>

<div class="cards">
    <div class="card">
        <div class="label">SA Final Equity</div>
        <div class="value" id="saEquity">...</div>
    </div>

    <div class="card">
        <div class="label">Exact Final Equity</div>
        <div class="value" id="exactEquity">...</div>
    </div>

    <div class="card">
        <div class="label">SA Sharpe Ratio</div>
        <div class="value" id="sharpe">...</div>
    </div>

    <div class="card">
        <div class="label">SA Max Drawdown</div>
        <div class="value" id="drawdown">...</div>
    </div>

    <div class="card">
        <div class="label">Backtest Windows</div>
        <div class="value" id="windows">...</div>
    </div>
</div>

<section class="panel">
<h2>Portfolio Equity Curve</h2>
<div class="chart-wrap">
<canvas id="equityChart"></canvas>
</div>

<p class="note">
Historical walk-forward simulation.
Values start from an initial equity of 1.
No future performance is guaranteed.
</p>
</section>

<section>
<h2>Multi-Agent Research</h2>

<div class="agent-grid">
    <div class="panel">
        <div class="agent-title">📈 Market Agent</div>
        <p id="marketAgent">Loading...</p>
    </div>

    <div class="panel">
        <div class="agent-title">🛡 Risk Agent</div>
        <p id="riskAgent">Loading...</p>
    </div>

    <div class="panel">
        <div class="agent-title">🧠 Portfolio Agent</div>
        <p id="portfolioAgent">Loading...</p>
    </div>
</div>
</section>

<section class="panel">
<h2>Research Coordinator Findings</h2>
<ul id="findings"></ul>
</section>

<p class="note">
Research only. This system uses local classical
simulated annealing, not quantum hardware.
Data comes from saved historical backtests.
</p>

<p id="error" class="error"></p>

</main>

<footer>
D-Wave Quantum Finance Research Engine V9
</footer>

<script>
const el = id => document.getElementById(id);

const fixed = (value, digits=2) =>
    value == null
        ? "N/A"
        : Number(value).toFixed(digits);

const percent = value =>
    value == null
        ? "N/A"
        : (Number(value) * 100).toFixed(2) + "%";

async function getJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(
            url + ": HTTP " + response.status
        );
    }

    return response.json();
}

async function start() {
    try {
        const [summary, equity] = await Promise.all([
            getJSON("/api/summary"),
            getJSON("/api/equity")
        ]);

        el("saEquity").textContent =
            fixed(summary.sa.final_equity) + "×";

        el("exactEquity").textContent =
            fixed(summary.exact.final_equity) + "×";

        el("sharpe").textContent =
            fixed(summary.sa.sharpe, 3);

        el("drawdown").textContent =
            percent(summary.sa.max_drawdown);

        el("windows").textContent =
            summary.completed_windows;

        el("marketAgent").textContent =
            "Latest historical portfolio: " +
            (summary.market.latest_portfolio || [])
                .join(", ");

        el("riskAgent").textContent =
            "Risk level: " +
            (summary.risk.risk_level || "N/A") +
            " | Annual volatility: " +
            percent(summary.risk.annual_volatility);

        el("portfolioAgent").textContent =
            "Optimal windows: " +
            summary.portfolio.optimal_windows +
            "/" +
            summary.portfolio.total_windows +
            " | Higher historical equity: " +
            summary.portfolio.better_realized_backtest;

        const list = el("findings");
        list.replaceChildren();

        for (const finding of summary.findings) {
            const li = document.createElement("li");
            li.textContent = finding;
            list.appendChild(li);
        }

        if (typeof Chart === "undefined") {
            throw new Error(
                "Chart.js could not load. Check internet."
            );
        }

        new Chart(el("equityChart"), {
            type: "line",
            data: {
                labels: equity.map(x => x.Date),
                datasets: [
                    {
                        label: "Simulated Annealing",
                        data: equity.map(x => x.SA),
                        borderColor: "#4de0b0",
                        backgroundColor: "#4de0b0",
                        pointRadius: 0,
                        borderWidth: 2,
                        tension: 0.15
                    },
                    {
                        label: "Exact Benchmark",
                        data: equity.map(x => x.Exact),
                        borderColor: "#72a7ff",
                        backgroundColor: "#72a7ff",
                        pointRadius: 0,
                        borderWidth: 2,
                        tension: 0.15
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: "#eaf0ff"
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: "#a4b4d0",
                            maxTicksLimit: 8
                        },
                        grid: {
                            color: "#25324b"
                        }
                    },
                    y: {
                        ticks: {
                            color: "#a4b4d0"
                        },
                        grid: {
                            color: "#25324b"
                        }
                    }
                }
            }
        });

    } catch (error) {
        el("error").textContent = error.message;
        console.error(error);
    }
}

start();
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
