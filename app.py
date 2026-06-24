"""
Streamlit dashboard — Market Regime Stress-Test Lab for RL Trading Agents.

This file is UI only. Core logic lives elsewhere:
  evaluate.run_battle()  → runs competitors, returns metrics + portfolio histories
  config.INITIAL_BALANCE → starting $ shown as "Final value" baseline ($10,000 default)
  explain.explain_results() → Groq / rule-based Analysis text

Run:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import INITIAL_BALANCE, SCENARIO_LABELS, TICKERS
from data import macro_summary
from evaluate import models_cache_key, run_battle
from explain import explain_results
from scenarios import SCENARIOS

# ── Palette ──────────────────────────────────────────────────────────────────
PURPLE_50 = "#faf5ff"
PURPLE_100 = "#f3e8ff"
PURPLE_200 = "#e9d5ff"
PURPLE_300 = "#c4b5fd"
PURPLE_400 = "#c084fc"
PURPLE_500 = "#a855f7"
PURPLE_600 = "#9333ea"
PURPLE_700 = "#7e22ce"
PURPLE_900 = "#4c1d95"
SLATE_500 = "#64748b"
SLATE_700 = "#334155"

AGENT_COLORS = {
    "Buy & Hold": "#94a3b8",
    "DQN": "#7c3aed",
    "PPO": "#a855f7",
    "A2C": "#c026d3",
}

# Plain-language tooltips shown under each metric in the UI.
METRIC_HINTS = {
    "sharpe_ann": "Return earned per unit of risk, scaled to a full year. Higher is better. Used to rank the leaderboard.",
    "sharpe_raw": "Return per unit of risk (not annualized). Used when the eval window is short.",
    "total_return": "Percent gain or loss from start to end of the eval window. Higher = more money made.",
    "final_value": f"Portfolio dollars at the end. Everyone starts at ${INITIAL_BALANCE:,} (see config.INITIAL_BALANCE).",
    "max_drawdown": "Largest peak-to-trough drop during the eval window. Lower = less scary dip.",
    "eval_days": "Number of trading days in the test slice (last 30% of the scenario — not used for training).",
    "sharpe_ci": "Rough range where Sharpe might fall if we resampled daily returns. Wider = less certain.",
    "avg_vix": "Average fear gauge. Higher values mean a more volatile, stressful market.",
    "yield_change": "How much the 10-year Treasury yield moved. Rising yields often pressure bond prices.",
    "spy_trend": "Share of days the broad market traded above its 200-day average — a simple uptrend signal.",
    "dollar_chart": "Shows actual account balance in dollars over the eval period.",
    "return_chart": "All strategies start at 0% on day one — easiest way to see who gained the most.",
    "sharpe_winner_card": "Best risk-adjusted score. Can differ from who made the most money.",
    "return_winner_card": "Largest percentage profit. The strategy you'd pick if only raw gains matter.",
}

st.set_page_config(
    page_title="Market Regime Stress-Test Lab for RL Trading Agents",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Inject purple/white CSS — colors, cards, sidebar, buttons."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: linear-gradient(160deg, #ffffff 0%, {PURPLE_50} 45%, #ffffff 100%);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, {PURPLE_50} 100%);
            border-right: 1px solid {PURPLE_200};
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}

        #MainMenu, footer, header[data-testid="stHeader"] {{
            visibility: hidden;
        }}

        .hero {{
            background: linear-gradient(135deg, {PURPLE_600} 0%, {PURPLE_400} 100%);
            border-radius: 20px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 40px rgba(124, 58, 237, 0.18);
            color: white;
        }}

        .hero h1 {{
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: white !important;
        }}

        .hero p {{
            margin: 0;
            opacity: 0.92;
            font-size: 1.05rem;
            color: #f5f3ff !important;
        }}

        .section-title {{
            font-size: 1.15rem;
            font-weight: 600;
            color: {PURPLE_900};
            margin: 1.75rem 0 1rem 0;
            padding-left: 0.75rem;
            border-left: 4px solid {PURPLE_500};
        }}

        .context-bar {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
            color: {SLATE_700};
            font-size: 0.95rem;
        }}

        .winner-card {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            height: 100%;
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.06);
        }}

        .winner-card.sharpe {{
            border-top: 4px solid {PURPLE_600};
        }}

        .winner-card.return {{
            border-top: 4px solid {PURPLE_400};
        }}

        .card-eyebrow {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {PURPLE_600};
            margin-bottom: 0.35rem;
        }}

        .card-agent {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {PURPLE_900};
            margin-bottom: 1rem;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }}

        .stat-box {{
            background: {PURPLE_50};
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
        }}

        .stat-label {{
            font-size: 0.72rem;
            color: {SLATE_500};
            font-weight: 500;
            margin-bottom: 0.15rem;
        }}

        .stat-value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {PURPLE_900};
        }}

        .stat-hint {{
            font-size: 0.68rem;
            color: {SLATE_500};
            line-height: 1.4;
            margin-top: 0.4rem;
        }}

        .card-eyebrow-hint {{
            font-size: 0.78rem;
            color: {SLATE_500};
            font-weight: 400;
            text-transform: none;
            letter-spacing: 0;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }}

        .section-note {{
            font-size: 0.88rem;
            color: {SLATE_500};
            margin: -0.35rem 0 1rem 0;
            line-height: 1.5;
        }}

        .context-hint {{
            font-size: 0.8rem;
            color: {SLATE_500};
            margin-top: 0.35rem;
            line-height: 1.45;
        }}

        .macro-pill .hint {{
            font-size: 0.68rem;
            color: {SLATE_500};
            margin-top: 0.35rem;
            line-height: 1.35;
        }}

        .chart-hint {{
            font-size: 0.82rem;
            color: {SLATE_500};
            margin: 0 0 0.75rem 0;
            line-height: 1.45;
        }}

        .control-hint {{
            font-size: 0.72rem;
            color: {SLATE_500};
            line-height: 1.4;
            margin-top: 0.5rem;
        }}

        .card-note {{
            margin-top: 0.85rem;
            font-size: 0.82rem;
            color: {SLATE_500};
            line-height: 1.45;
        }}

        .info-box {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-left: 4px solid {PURPLE_500};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin: 1rem 0;
            font-size: 0.8rem;
            color: {SLATE_700};
            line-height: 1.5;
        }}

        .info-box h4 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.85rem;
            font-weight: 600;
            color: {PURPLE_900};
        }}

        .info-box ul {{
            margin: 0.35rem 0 0 0;
            padding-left: 1.1rem;
        }}

        .info-box li {{
            margin-bottom: 0.35rem;
        }}

        .info-box li:last-child {{
            margin-bottom: 0;
        }}

        .info-box .tag {{
            font-weight: 600;
            color: {PURPLE_700};
        }}

        .mini-stat-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
            margin-bottom: 0.5rem;
        }}

        .mini-stat {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-radius: 12px;
            padding: 0.85rem 1rem;
            text-align: center;
        }}

        .mini-stat .stat-label {{ text-align: center; }}
        .mini-stat .stat-value {{ font-size: 1.05rem; text-align: center; }}
        .mini-stat .stat-hint {{ text-align: center; font-size: 0.65rem; }}

        .chart-panel {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-radius: 16px;
            padding: 1.25rem 1.25rem 0.5rem 1.25rem;
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.05);
        }}

        .chart-controls {{
            background: {PURPLE_50};
            border: 1px solid {PURPLE_200};
            border-radius: 14px;
            padding: 1rem;
        }}

        .analysis-box {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-left: 4px solid {PURPLE_500};
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            line-height: 1.65;
            color: {SLATE_700};
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.05);
        }}

        .empty-state {{
            text-align: center;
            background: white;
            border: 2px dashed {PURPLE_200};
            border-radius: 20px;
            padding: 3.5rem 2rem;
            margin-top: 1rem;
        }}

        .empty-state h3 {{
            color: {PURPLE_900};
            margin-bottom: 0.5rem;
        }}

        .empty-state p {{
            color: {SLATE_500};
            max-width: 420px;
            margin: 0 auto;
            line-height: 1.6;
        }}

        .sidebar-brand {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {PURPLE_900};
            margin-bottom: 0.25rem;
        }}

        .sidebar-sub {{
            font-size: 0.82rem;
            color: {SLATE_500};
            margin-bottom: 1.25rem;
        }}

        .macro-pill {{
            background: white;
            border: 1px solid {PURPLE_200};
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }}

        .macro-pill .label {{
            font-size: 0.72rem;
            color: {SLATE_500};
            font-weight: 500;
        }}

        .macro-pill .value {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {PURPLE_900};
        }}

        .badge-warn {{
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        div[data-testid="stButton"] > button[kind="primary"] {{
            background: linear-gradient(135deg, {PURPLE_600}, {PURPLE_500});
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.65rem 1rem;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
        }}

        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {PURPLE_700}, {PURPLE_600});
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.4);
        }}

        .stDataFrame {{
            border: 1px solid {PURPLE_200};
            border-radius: 12px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Market Regime Stress-Test Lab for RL Trading Agents</h1>
            <p>Compare RL trading agents vs buy-and-hold across historically labeled market stress tests.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_note(text: str) -> None:
    st.markdown(f'<p class="section-note">{text}</p>', unsafe_allow_html=True)


def render_competitors_guide(compact: bool = False) -> None:
    """Explain Buy & Hold vs RL agents in the sidebar or above the leaderboard."""
    if compact:
        st.markdown(
            """
            <div class="info-box">
                <h4>Who is competing?</h4>
                <ul>
                    <li><span class="tag">Buy &amp; Hold</span> — buys on day 1, never trades again (passive benchmark).</li>
                    <li><span class="tag">DQN / PPO / A2C</span> — RL agents that choose hold, buy, or sell each day using learned models.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="info-box">
            <h4>Buy &amp; Hold vs the algorithms</h4>
            <p style="margin:0 0 0.5rem 0;">
                Every battle compares one <strong>passive baseline</strong> against three
                <strong>active RL strategies</strong> on the same eval window and starting cash.
            </p>
            <ul>
                <li>
                    <span class="tag">Buy &amp; Hold</span> — invests (almost) all cash on the
                    first eval day and holds until the end. No AI, one trade, the sanity check:
                    “was active trading worth it?”
                </li>
                <li>
                    <span class="tag">DQN</span> — learns which action (hold / buy / sell) is
                    best using value estimates from past experience.
                </li>
                <li>
                    <span class="tag">PPO</span> — learns a trading policy directly with stable,
                    gradual updates (common default in RL).
                </li>
                <li>
                    <span class="tag">A2C</span> — combines a policy (“what to do”) with a
                    value estimate (“how good is this situation”).
                </li>
            </ul>
            <p style="margin:0.5rem 0 0 0; font-size:0.75rem; color:#64748b;">
                Sharpe ranks risk-adjusted performance; highest return shows who made the most money.
                They can disagree — e.g. Buy &amp; Hold may earn more in a bull market while an RL
                agent wins Sharpe by barely trading.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_winner_card(
    card_type: str,
    eyebrow: str,
    eyebrow_hint: str,
    agent: str,
    stats: list[tuple[str, str, str]],
    note: str = "",
) -> None:
    """Render a winner card. Each stat is (label, value, hint_text)."""
    stats_html = "".join(
        f'<div class="stat-box"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-hint">{hint}</div></div>'
        for label, value, hint in stats
    )
    note_html = f'<div class="card-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="winner-card {card_type}">
            <div class="card-eyebrow">{eyebrow}</div>
            <div class="card-eyebrow-hint">{eyebrow_hint}</div>
            <div class="card-agent">{agent}</div>
            <div class="stat-grid">{stats_html}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_stats(items: list[tuple[str, str, str]]) -> None:
    """Mini stat tiles: (label, value, hint)."""
    cells = "".join(
        f'<div class="mini-stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-hint">{hint}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="mini-stat-row">{cells}</div>', unsafe_allow_html=True)


def render_macro_pill(label: str, value: str, hint: str) -> None:
    st.markdown(
        f'<div class="macro-pill"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def md_to_html_block(text: str) -> str:
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    html = html.replace("\n\n", "<br><br>")
    return f'<div class="analysis-box">{html}</div>'


def style_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_return"] = out["total_return"].map(lambda x: f"{x:.1%}")
    out["max_drawdown"] = out["max_drawdown"].map(lambda x: f"{x:.1%}")
    out["final_value"] = out["final_value"].map(lambda x: f"${x:,.0f}")
    out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.3f}")
    rename = {
        "agent": "Agent",
        "sharpe": "Sharpe",
        "sharpe_ci": "Sharpe CI",
        "total_return": "Return",
        "max_drawdown": "Max DD",
        "final_value": "Final Value",
    }
    return out.rename(columns=rename)


@st.cache_data(show_spinner=False)
def cached_battle(ticker: str, scenario: str, models_version: str) -> dict:
    return run_battle(ticker, scenario)


def build_equity_figure(
    battle: dict,
    winner: dict,
    view_mode: str = "Dollar value",
) -> go.Figure:
    """Plot daily portfolio values from battle['histories'] (built in evaluate.py)."""
    dates = battle["dates"]
    winner_name = winner["agent"]
    use_return_view = view_mode == "Return %"
    ordered_names = [
        row["agent"] for row in battle["results"] if row["agent"] in battle["histories"]
    ]

    fig = go.Figure()
    start_value = INITIAL_BALANCE

    for name in ordered_names:
        history = battle["histories"][name]
        x = dates[: len(history)]
        values = [float(v) for v in history]
        if values:
            start_value = values[0]

        y = (
            [((v / values[0]) - 1) * 100 if values else 0 for v in values]
            if use_return_view
            else values
        )

        customdata = []
        for i, val in enumerate(values):
            ret_pct = ((val / values[0]) - 1) * 100 if values else 0.0
            daily_pct = (
                ((values[i] - values[i - 1]) / values[i - 1]) * 100 if i > 0 else 0.0
            )
            pnl = val - values[0]
            customdata.append([val, ret_pct, daily_pct, pnl])

        is_winner = name == winner_name
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"{name}{'  ★' if is_winner else ''}",
                mode="lines",
                line={
                    "color": AGENT_COLORS.get(name, PURPLE_500),
                    "width": 3.5 if is_winner else 2.2,
                },
                customdata=customdata,
                hovertemplate=(
                    f"<b>{name}</b>"
                    + (" (Sharpe winner)" if is_winner else "")
                    + "<br>Date: %{x}<br>"
                    + "Portfolio: $%{customdata[0]:,.2f}<br>"
                    + "P/L vs start: $%{customdata[3]:+,.2f}<br>"
                    + "Return: %{customdata[1]:+.2f}%<br>"
                    + "Daily: %{customdata[2]:+.2f}%<extra></extra>"
                ),
            )
        )

    y_title = "Return since eval start (%)" if use_return_view else "Portfolio value ($)"
    if not use_return_view:
        fig.add_hline(
            y=start_value,
            line_dash="dot",
            line_color=PURPLE_300,
            line_width=1.5,
            annotation_text=f"Start ${start_value:,.0f}",
            annotation_font_color=PURPLE_600,
            annotation_font_size=11,
        )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#faf8ff",
        height=440,
        margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family="Inter, sans-serif", color=PURPLE_900, size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PURPLE_200,
            borderwidth=1,
            title_text="",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_color=PURPLE_900, font_size=13),
        xaxis_title="Eval period (held-out 30%)",
        yaxis_title=y_title,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(167, 139, 250, 0.15)",
        linecolor=PURPLE_200,
        showspikes=True,
        spikemode="across",
        spikecolor="rgba(124, 58, 237, 0.35)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(167, 139, 250, 0.15)",
        linecolor=PURPLE_200,
        showspikes=True,
        spikemode="across",
        spikecolor="rgba(124, 58, 237, 0.35)",
        ticksuffix="%" if use_return_view else "",
    )
    return fig


# ── App ──────────────────────────────────────────────────────────────────────
inject_theme()
render_hero()

# ── Sidebar: user picks scenario + ticker, clicks Run Battle ─────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Battle setup</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Pick a regime window and ticker, then run the eval replay.</div>',
        unsafe_allow_html=True,
    )

    scenario = st.selectbox(
        "Market scenario",
        list(SCENARIOS.keys()),
        format_func=lambda key: SCENARIO_LABELS.get(key, key),
        help="A fixed historical date range labeled by market regime (e.g. 2008 recession, 2017 bull run).",
    )

    ticker = st.selectbox(
        "Stock ticker",
        TICKERS,
        help="Which stock to replay on the eval window. Must be one of the trained tickers in config.TICKERS.",
    )

    run_clicked = st.button("Run Battle", type="primary", use_container_width=True)

    st.markdown(
        '<div class="section-title" style="margin-top:1.25rem;">Who is competing?</div>',
        unsafe_allow_html=True,
    )
    render_competitors_guide(compact=False)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Macro context</div>', unsafe_allow_html=True)
    section_note("Background conditions for the full scenario window (not just the eval slice).")
    macro = macro_summary(scenario)
    if macro:
        render_macro_pill(
            "Avg VIX",
            f"{macro.get('avg_vix', 0):.1f}",
            METRIC_HINTS["avg_vix"],
        )
        render_macro_pill(
            "10Y yield change",
            f"{macro.get('yield_10y_change', 0):+.2f}",
            METRIC_HINTS["yield_change"],
        )
        render_macro_pill(
            "SPY above 200d MA",
            f"{macro.get('spy_above_sma200_pct', 0):.1f}%",
            METRIC_HINTS["spy_trend"],
        )
        st.caption(f"Window: {macro.get('start')} → {macro.get('end')}")
    else:
        st.info("Macro data unavailable for this window.")

    st.caption(
        "Eval uses the held-out **30%** of each scenario. "
        "Train first: `.venv/bin/python train.py`"
    )

# Remember last battle across reruns (e.g. when toggling Chart view Return %).
if "battle_ticker" not in st.session_state:
    st.session_state.battle_ticker = None
    st.session_state.battle_scenario = None

if run_clicked:
    st.session_state.battle_ticker = ticker
    st.session_state.battle_scenario = scenario

show_battle = (
    st.session_state.battle_ticker is not None
    and st.session_state.battle_scenario is not None
)

if show_battle:
    # Use stored params so chart toggles / reruns do not wipe results.
    battle_ticker = st.session_state.battle_ticker
    battle_scenario = st.session_state.battle_scenario

    if not run_clicked and (ticker != battle_ticker or scenario != battle_scenario):
        st.info("Selection changed in the sidebar — click **Run Battle** to refresh results.")

    # ── Backend: replay all agents on eval split (see evaluate.py) ───────────
    models_ver = models_cache_key()
    try:
        if run_clicked:
            with st.spinner("Replaying agents on eval split..."):
                battle = cached_battle(battle_ticker, battle_scenario, models_ver)
        else:
            battle = cached_battle(battle_ticker, battle_scenario, models_ver)
    except Exception as exc:
        st.error(f"Battle failed: {exc}")
        st.stop()

    results = battle["results"]
    if not results:
        st.warning("No results returned.")
        st.stop()

    if battle["missing_models"]:
        st.warning(
            "Missing models: "
            + ", ".join(battle["missing_models"])
            + " — run `.venv/bin/python train.py`"
        )

    # Sharpe winner = rank #1; return winner = highest total_return (may differ).
    sharpe_winner = battle.get("sharpe_winner") or results[0]
    return_winner = battle.get("return_winner") or max(results, key=lambda row: row["total_return"])
    same_winner = battle.get("same_winner", sharpe_winner["agent"] == return_winner["agent"])
    label = SCENARIO_LABELS.get(battle_scenario, battle_scenario)

    warn_html = (
        '<span class="badge-warn">Short window — Sharpe is noisy</span>'
        if battle["short_window_warning"]
        else ""
    )
    st.markdown(
        f"""
        <div class="context-bar">
            <strong>{battle_ticker}</strong> &nbsp;·&nbsp; <strong>{label}</strong>
            &nbsp;·&nbsp; {battle['eval_days']} eval days &nbsp; {warn_html}
            <div class="context-hint">{METRIC_HINTS["eval_days"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sharpe_winner.get("low_activity"):
        st.warning(
            "Sharpe winner had very low return (<1%) — likely sat mostly in cash. "
            "Check **Highest return** and Buy & Hold for economic performance."
        )

    w1, w2 = st.columns(2)
    sharpe_label = "Sharpe (ann.)" if sharpe_winner.get("sharpe_annualized") else "Sharpe (raw)"
    sharpe_hint = (
        METRIC_HINTS["sharpe_ann"]
        if sharpe_winner.get("sharpe_annualized")
        else METRIC_HINTS["sharpe_raw"]
    )
    with w1:
        render_winner_card(
            "sharpe",
            "Sharpe winner",
            METRIC_HINTS["sharpe_winner_card"],
            sharpe_winner["agent"],
            [
                (sharpe_label, f"{sharpe_winner['sharpe']:.2f}", sharpe_hint),
                (
                    "Total return",
                    f"{sharpe_winner['total_return']:.1%}",
                    METRIC_HINTS["total_return"],
                ),
            ],
        )
    with w2:
        note = (
            "Same agent as Sharpe winner."
            if same_winner
            else (
                f"Ranked by dollars, not Sharpe "
                f"({sharpe_label}: {return_winner['sharpe']:.2f})."
            )
        )
        render_winner_card(
            "return",
            "Highest return",
            METRIC_HINTS["return_winner_card"],
            return_winner["agent"],
            [
                (
                    "Total return",
                    f"{return_winner['total_return']:.1%}",
                    METRIC_HINTS["total_return"],
                ),
                (
                    "Final value",
                    f"${return_winner['final_value']:,.0f}",
                    METRIC_HINTS["final_value"],
                ),
            ],
            note=note,
        )

    mini_items = [
        (
            "Sharpe winner max DD",
            f"{sharpe_winner['max_drawdown']:.1%}",
            METRIC_HINTS["max_drawdown"],
        ),
        (
            "Sharpe winner final $",
            f"${sharpe_winner['final_value']:,.0f}",
            METRIC_HINTS["final_value"],
        ),
    ]
    if not same_winner:
        mini_items.append(
            ("Return leader Sharpe", f"{return_winner['sharpe']:.2f}", sharpe_hint)
        )
    else:
        mini_items.append(
            (
                "Return leader max DD",
                f"{return_winner['max_drawdown']:.1%}",
                METRIC_HINTS["max_drawdown"],
            )
        )
    render_mini_stats(mini_items)

    st.markdown('<div class="section-title">Equity curves</div>', unsafe_allow_html=True)

    chart_main, chart_side = st.columns([4, 1])
    with chart_side:
        st.markdown('<div class="chart-controls">', unsafe_allow_html=True)
        chart_view = st.radio(
            "Chart view",
            ["Dollar value", "Return %"],
            key="chart_view",
            help="Return % rebases all lines to 0% on day one.",
        )
        chart_hint = (
            METRIC_HINTS["return_chart"]
            if chart_view == "Return %"
            else METRIC_HINTS["dollar_chart"]
        )
        st.markdown(f'<p class="control-hint">{chart_hint}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_main:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown(
            '<p class="chart-hint">Hover any date to compare portfolio value, P/L, and daily change for every agent.</p>',
            unsafe_allow_html=True,
        )
        fig = build_equity_figure(battle, sharpe_winner, view_mode=chart_view)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Leaderboard</div>', unsafe_allow_html=True)
    render_competitors_guide(compact=True)
    section_note(
        "All competitors on the eval window, sorted by Sharpe. "
        "Hover column headers for quick definitions."
    )
    table = pd.DataFrame(results)
    display_cols = ["agent", "sharpe", "total_return", "max_drawdown", "final_value"]
    if "sharpe_ci_low" in table.columns:
        table["sharpe_ci"] = table.apply(
            lambda row: (
                f"[{row['sharpe_ci_low']}, {row['sharpe_ci_high']}]"
                if pd.notna(row.get("sharpe_ci_low"))
                else ""
            ),
            axis=1,
        )
        display_cols.insert(2, "sharpe_ci")
    styled = style_leaderboard(table[display_cols])
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Agent": st.column_config.TextColumn(
                "Agent",
                help="Trading strategy: Buy & Hold baseline or an RL algorithm (DQN, PPO, A2C).",
            ),
            "Sharpe": st.column_config.TextColumn(
                "Sharpe",
                help=METRIC_HINTS["sharpe_ann"],
            ),
            "Sharpe CI": st.column_config.TextColumn(
                "Sharpe CI",
                help=METRIC_HINTS["sharpe_ci"],
            ),
            "Return": st.column_config.TextColumn(
                "Return",
                help=METRIC_HINTS["total_return"],
            ),
            "Max DD": st.column_config.TextColumn(
                "Max DD",
                help=METRIC_HINTS["max_drawdown"],
            ),
            "Final Value": st.column_config.TextColumn(
                "Final Value",
                help=METRIC_HINTS["final_value"],
            ),
        },
    )

    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
    section_note("Interview-style summary grounded in the full leaderboard.")
    st.markdown(explain_results(battle))

else:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Ready when you are</h3>
            <p>
                Choose a <strong>market scenario</strong> and <strong>ticker</strong>
                in the sidebar, then click <strong>Run Battle</strong> to compare
                DQN, PPO, A2C, and buy-and-hold on the held-out eval window.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
