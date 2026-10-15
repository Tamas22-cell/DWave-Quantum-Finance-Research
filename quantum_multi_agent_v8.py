
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# D-WAVE QUANTUM FINANCE V8
# MULTI-AGENT RESEARCH SYSTEM
# ==========================================

BASE = Path(__file__).resolve().parent

V7_FILE = BASE / "results_v7.json"
CURVE_FILE = BASE / "equity_curve_v7.csv"

OUTPUT = BASE / "multi_agent_report_v8.json"
MARKDOWN = BASE / "multi_agent_report_v8.md"

CRYPTO = {
    "BTC-USD", "ETH-USD", "SOL-USD",
    "ADA-USD", "SUI20947-USD",
    "AVAX-USD", "LINK-USD", "XRP-USD",
    "DOGE-USD", "LTC-USD"
}

QUANTUM = {"IONQ", "QBTS", "RGTI", "QUBT"}


def safe_number(value):
    if value is None:
        return None
    try:
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    except (ValueError, TypeError):
        return None


def pct(value):
    value = safe_number(value)
    return "N/A" if value is None else f"{value:.2%}"


def number(value):
    value = safe_number(value)
    return "N/A" if value is None else f"{value:.4f}"


# ==========================================
# AGENT 1: MARKET ANALYST
# ==========================================

class MarketAgent:

    def analyze(self, results):
        print("\n[MARKET AGENT] Analyzing portfolios...")

        windows = results.get("windows", [])
        frequencies = {}
        crypto_exposure = []
        quantum_exposure = []

        for window in windows:
            assets = window.get("sa_assets", [])

            for asset in assets:
                frequencies[asset] = (
                    frequencies.get(asset, 0) + 1
                )

            crypto_exposure.append(
                sum(a in CRYPTO for a in assets)
            )
            quantum_exposure.append(
                sum(a in QUANTUM for a in assets)
            )

        ranking = sorted(
            frequencies.items(),
            key=lambda x: (-x[1], x[0])
        )

        return {
            "agent": "Market Agent",
            "analyzed_windows": len(windows),
            "most_selected_assets": [
                {"asset": asset, "windows": count}
                for asset, count in ranking[:10]
            ],
            "average_crypto_positions": (
                float(np.mean(crypto_exposure))
                if crypto_exposure else 0.0
            ),
            "average_quantum_positions": (
                float(np.mean(quantum_exposure))
                if quantum_exposure else 0.0
            ),
            "latest_portfolio": (
                windows[-1].get("sa_assets", [])
                if windows else []
            )
        }


# ==========================================
# AGENT 2: RISK ANALYST
# ==========================================

class RiskAgent:

    def analyze(self, results):
        print("[RISK AGENT] Evaluating risk...")

        metrics = results.get("sa_metrics", {})
        windows = results.get("windows", [])

        volatility = safe_number(
            metrics.get("annual_volatility")
        )
        drawdown = safe_number(
            metrics.get("max_drawdown")
        )
        sharpe = safe_number(
            metrics.get("sharpe")
        )

        limit = safe_number(
            results.get("max_train_volatility")
        )

        violations = []

        for window in windows:
            vol = safe_number(
                window.get("sa_train_volatility")
            )

            if (
                vol is not None
                and limit is not None
                and vol > limit + 1e-9
            ):
                violations.append(window["window"])

        if volatility is None or drawdown is None:
            risk_level = "UNKNOWN"
        elif volatility > 0.50 or drawdown < -0.35:
            risk_level = "HIGH"
        elif volatility > 0.30 or drawdown < -0.20:
            risk_level = "ELEVATED"
        else:
            risk_level = "MODERATE"

        return {
            "agent": "Risk Agent",
            "risk_level": risk_level,
            "annual_volatility": volatility,
            "max_drawdown": drawdown,
            "sharpe_ratio": sharpe,
            "training_volatility_limit": limit,
            "training_limit_violations": violations,
            "risk_note": (
                "Training volatility limits do not "
                "guarantee future realized volatility."
            )
        }


# ==========================================
# AGENT 3: PORTFOLIO ANALYST
# ==========================================

class PortfolioAgent:

    def analyze(self, results):
        print(
            "[PORTFOLIO AGENT] Comparing SA and Exact..."
        )

        windows = results.get("windows", [])
        sa = results.get("sa_metrics", {})
        exact = results.get("exact_metrics", {})

        gaps = [
            float(w["objective_gap"])
            for w in windows
            if safe_number(
                w.get("objective_gap")
            ) is not None
        ]

        optimal_windows = sum(
            abs(gap) <= 1e-8 for gap in gaps
        )

        sa_equity = safe_number(
            sa.get("final_equity")
        )
        exact_equity = safe_number(
            exact.get("final_equity")
        )

        if sa_equity is None or exact_equity is None:
            better = "UNKNOWN"
        elif sa_equity > exact_equity:
            better = "SA"
        elif sa_equity < exact_equity:
            better = "EXACT"
        else:
            better = "TIE"

        return {
            "agent": "Portfolio Agent",
            "total_windows": len(windows),
            "optimal_windows": optimal_windows,
            "optimal_rate": (
                optimal_windows / len(windows)
                if windows else 0.0
            ),
            "average_objective_gap": (
                float(np.mean(gaps))
                if gaps else None
            ),
            "sa_final_equity": sa_equity,
            "exact_final_equity": exact_equity,
            "better_realized_backtest": better,
            "note": (
                "Exact minimizes the training objective; "
                "it does not guarantee better future returns."
            )
        }


# ==========================================
# COORDINATOR
# ==========================================

class ResearchCoordinator:

    def __init__(self):
        self.agents = [
            MarketAgent(),
            RiskAgent(),
            PortfolioAgent()
        ]

    def run(self, results):
        print("\n=== AI MULTI-AGENT RESEARCH V8 ===")

        agent_results = {}

        for agent in self.agents:
            report = agent.analyze(results)
            agent_results[report["agent"]] = report

        risk = agent_results["Risk Agent"]
        portfolio = agent_results["Portfolio Agent"]

        findings = []

        if risk["risk_level"] in ("HIGH", "ELEVATED"):
            findings.append(
                "Portfolio risk requires continued monitoring."
            )

        if risk["training_limit_violations"]:
            findings.append(
                "Some training windows violated "
                "the volatility constraint."
            )
        else:
            findings.append(
                "All reported training windows satisfy "
                "the volatility constraint."
            )

        if portfolio["better_realized_backtest"] == "SA":
            findings.append(
                "SA achieved higher realized equity "
                "in this historical backtest."
            )
        elif portfolio["better_realized_backtest"] == "EXACT":
            findings.append(
                "Exact achieved higher realized equity "
                "in this historical backtest."
            )

        findings.append(
            "Results are historical simulations, "
            "not live investment recommendations."
        )

        return {
            "project": "D-Wave Quantum Finance V8",
            "architecture": "Rule-based Multi-Agent",
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "source": "results_v7.json",
            "agents": agent_results,
            "coordinator_findings": findings
        }


# ==========================================
# REPORT GENERATION
# ==========================================

def create_markdown(report):
    agents = report["agents"]

    market = agents["Market Agent"]
    risk = agents["Risk Agent"]
    portfolio = agents["Portfolio Agent"]

    lines = [
        "# D-Wave Quantum Finance V8",
        "",
        "## Multi-Agent Research Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "### Market Agent",
        "",
        f"Analyzed windows: {market['analyzed_windows']}",
        "",
        "Most frequently selected assets:",
        ""
    ]

    for item in market["most_selected_assets"]:
        lines.append(
            f"- {item['asset']}: "
            f"{item['windows']} windows"
        )

    lines.extend([
        "",
        "Latest historical portfolio: "
        + ", ".join(market["latest_portfolio"]),
        "",
        "### Risk Agent",
        "",
        f"Risk level: **{risk['risk_level']}**",
        "",
        "Annual volatility: "
        + pct(risk["annual_volatility"]),
        "",
        "Maximum drawdown: "
        + pct(risk["max_drawdown"]),
        "",
        "Sharpe ratio: "
        + number(risk["sharpe_ratio"]),
        "",
        "Training volatility limit: "
        + pct(risk["training_volatility_limit"]),
        "",
        "Training limit violations: "
        + str(risk["training_limit_violations"]),
        "",
        "### Portfolio Agent",
        "",
        "Optimal windows: "
        + str(portfolio["optimal_windows"])
        + "/"
        + str(portfolio["total_windows"]),
        "",
        "Optimal rate: "
        + pct(portfolio["optimal_rate"]),
        "",
        "Average objective gap: "
        + number(portfolio["average_objective_gap"]),
        "",
        "SA final equity: "
        + number(portfolio["sa_final_equity"]),
        "",
        "Exact final equity: "
        + number(portfolio["exact_final_equity"]),
        "",
        "Better historical equity: "
        + portfolio["better_realized_backtest"],
        "",
        "### Coordinator Findings",
        ""
    ])

    for finding in report["coordinator_findings"]:
        lines.append(f"- {finding}")

    lines.extend([
        "",
        "---",
        "",
        "Research only. Not financial advice.",
        ""
    ])

    return "\n".join(lines)


# ==========================================
# MAIN
# ==========================================

def main():
    if not V7_FILE.exists():
        raise FileNotFoundError(
            f"V7 results missing: {V7_FILE}"
        )

    with V7_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        results = json.load(file)

    coordinator = ResearchCoordinator()
    report = coordinator.run(results)

    with OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    MARKDOWN.write_text(
        create_markdown(report),
        encoding="utf-8"
    )

    print("\n=== COORDINATOR FINDINGS ===")

    for finding in report["coordinator_findings"]:
        print("-", finding)

    print("\nJSON saved:", OUTPUT)
    print("Markdown report:", MARKDOWN)
    print("\nV8 completed successfully.")


if __name__ == "__main__":
    main()
