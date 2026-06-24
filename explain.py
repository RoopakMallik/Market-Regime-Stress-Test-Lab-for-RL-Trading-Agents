"""
Generate interview-ready battle narratives (Groq API or rule-based fallback).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from config import GROQ_MODEL, SCENARIO_LABELS, TICKERS

load_dotenv()

AGENT_DESCRIPTIONS = {
    "PPO": "PPO learns a trading policy with stable, clipped gradient updates.",
    "DQN": "DQN estimates the value of hold/buy/sell actions from past experience.",
    "A2C": "A2C combines a policy network with a value network for on-policy learning.",
    "Buy & Hold": "Buy & Hold buys on day 1 of the eval window and never trades again.",
}


def _format_leaderboard(results: list[dict]) -> str:
    lines = []
    for row in results:
        ci = ""
        if "sharpe_ci_low" in row and "sharpe_ci_high" in row:
            ci = f" CI [{row['sharpe_ci_low']}, {row['sharpe_ci_high']}]"
        low = " [low activity]" if row.get("low_activity") else ""
        trades = f" trades={row['trade_count']}" if "trade_count" in row else ""
        lines.append(
            f"- {row['agent']}{low}: Sharpe={row['sharpe']:.3f}{ci}, "
            f"Return={row['total_return']:.1%}, MaxDD={row['max_drawdown']:.1%}, "
            f"Final=${row['final_value']:,.0f}{trades}"
        )
    return "\n".join(lines)


def _yield_change_text(macro: dict) -> str:
    change = macro.get("yield_10y_change")
    if change is None:
        return "n/a"
    return f"{change:+.2f} percentage points"


def explain_results_rule_based(battle: dict) -> str:
    """Template narrative when Groq is unavailable."""
    winner = battle["sharpe_winner"]
    return_winner = battle["return_winner"]
    scenario = battle["scenario"]
    ticker = battle["ticker"]
    all_results = battle["results"]
    macro = battle.get("macro_summary") or {}
    short_window = battle.get("short_window_warning", False)

    agent = winner["agent"]
    label = SCENARIO_LABELS.get(scenario, scenario)
    desc = AGENT_DESCRIPTIONS.get(agent, agent)

    lines = [
        f"**Interview summary — {ticker} in {label} (eval split)**",
        "",
        f"**Sharpe winner:** {agent} — {desc}",
        f"Risk-adjusted score: Sharpe {winner['sharpe']:.2f} "
        f"({'annualized' if winner.get('sharpe_annualized') else 'raw, short window'}), "
        f"return {winner['total_return']:.1%}, max drawdown {winner['max_drawdown']:.1%}.",
    ]

    if winner.get("low_activity"):
        lines.append(
            "⚠️ This agent barely moved capital (return under 1%). "
            "High Sharpe here often means 'sat in cash' rather than skilled trading."
        )

    if not battle.get("same_winner"):
        lines.append(
            f"**Highest return:** {return_winner['agent']} at {return_winner['total_return']:.1%} "
            f"(final ${return_winner['final_value']:,.0f}) — "
            "rankings by Sharpe and by dollars differ."
        )

    bh = next((r for r in all_results if r["agent"] == "Buy & Hold"), None)
    if bh:
        lines.append(
            f"**vs Buy & Hold:** passive benchmark returned {bh['total_return']:.1%} "
            f"with Sharpe {bh['sharpe']:.2f} and max drawdown {bh['max_drawdown']:.1%}."
        )

    if macro:
        lines.append(
            f"**Macro backdrop:** avg VIX {macro.get('avg_vix', 'n/a')}, "
            f"10Y yield change {_yield_change_text(macro)}, "
            f"SPY above 200d MA {macro.get('spy_above_sma200_pct', 'n/a')}% of days."
        )

    lines.append(
        f"**Setup note:** RL models train on a basket ({', '.join(TICKERS)}) "
        f"but this battle replays **{ticker}** only on the held-out 30% of the window."
    )

    if short_window:
        lines.append("⚠️ Short eval window — treat Sharpe as noisy.")

    if battle.get("missing_models"):
        lines.append(f"Missing models: {', '.join(battle['missing_models'])}.")

    lines.append(
        "\n*Historical replay only — not a forecast. "
        "Say in interviews: 'labeled regime backtest with out-of-sample eval slice.'*"
    )
    return "\n".join(lines)


def explain_results_groq(battle: dict) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return explain_results_rule_based(battle)

    winner = battle["sharpe_winner"]
    return_winner = battle["return_winner"]
    scenario = battle["scenario"]
    ticker = battle["ticker"]
    label = SCENARIO_LABELS.get(scenario, scenario)
    macro = battle.get("macro_summary") or {}

    system = (
        "You are a cautious quant reviewer helping a candidate explain a portfolio project. "
        "Never predict future returns. Distinguish Sharpe winner from return leader. "
        "Mention when an agent had low activity (near-zero return). "
        "Use plain English suitable for a data-science interview."
    )

    user_prompt = f"""Summarize this trading experiment in 4-5 short sentences for an interview.

Ticker: {ticker}
Scenario: {label} ({scenario})
Eval days: {battle['eval_days']} (held-out 30%, not used in training)
Short-window warning: {battle.get('short_window_warning')}
Training note: models trained on multi-ticker basket {TICKERS}, evaluated on {ticker} only

Macro: avg VIX {macro.get('avg_vix')}, 10Y yield change {_yield_change_text(macro)}, SPY above 200d MA {macro.get('spy_above_sma200_pct')}%

Leaderboard (Sharpe rank):
{_format_leaderboard(battle['results'])}

Sharpe winner: {winner['agent']} (Sharpe {winner['sharpe']}, return {winner['total_return']:.1%}, low_activity={winner.get('low_activity')})
Return leader: {return_winner['agent']} (return {return_winner['total_return']:.1%})
Same winner for both: {battle.get('same_winner')}

Cover: (1) who won on risk-adjusted metrics and why that can happen, (2) Buy & Hold comparison, (3) one honest limitation."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=450,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text + "\n\n*Historical replay only — not a forecast.*"
    except Exception:
        pass

    return explain_results_rule_based(battle)


def explain_results(battle: dict[str, Any]) -> str:
    """Main entry — pass the full battle dict from run_battle()."""
    if os.getenv("GROQ_API_KEY"):
        return explain_results_groq(battle)
    return explain_results_rule_based(battle)
