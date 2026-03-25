"""
app.py — AI Risk Intelligence Dashboard
========================================
Tabs:
  1. Overview        — KPIs + scatter + histograms
  2. Risk Distribution — pie charts + agreement matrix
  3. Allocation       — current vs suggested + delta bar
  4. Insights         — red flags + sector + persona benchmarks
  5. Investor Drill-Down — per-investor profile, charts, advice
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import run_pipeline

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colours ───────────────────────────────────────────────────────────────────

RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
PALETTE     = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]

# ── minimal dark CSS ──────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* page background */
.stApp { background: #0f0f11; color: #e2e8f0; }
.block-container { padding: 1.5rem 2rem 4rem; max-width: 1400px; }

/* sidebar */
[data-testid="stSidebar"] { background: #0a0a0c; border-right: 1px solid #1e1e24; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* section header */
.sec-head {
    font-size: 13px; font-weight: 600; color: #6366f1;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 24px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid #1e1e24;
}

/* stat card */
.kcard {
    background: #141417; border: 1px solid #1e1e24; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 10px;
}
.kcard .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.kcard .val   { font-size: 26px; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kcard .sub   { font-size: 12px; color: #475569; margin-top: 4px; }

/* flag row */
.flag-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; background: #141417;
    border-left: 3px solid #6366f1; border-radius: 0 6px 6px 0;
    margin-bottom: 6px;
}
.flag-row .flabel { color: #94a3b8; font-size: 13px; }
.flag-row .fval   { color: #f1f5f9; font-size: 13px; font-weight: 600; }

/* risk badge */
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.badge-Low    { background: #052e16; color: #22c55e; border: 1px solid #22c55e; }
.badge-Medium { background: #451a03; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-High   { background: #450a0a; color: #ef4444; border: 1px solid #ef4444; }

/* investor card */
.icard {
    background: #141417; border: 1px solid #1e1e24;
    border-radius: 10px; padding: 20px 22px; margin-bottom: 10px;
}
.icard h4 { font-size: 11px; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 1.5px; margin: 0 0 14px; }
.irow { display: flex; justify-content: space-between; margin-bottom: 8px; }
.irow .ilabel { color: #64748b; font-size: 13px; }
.irow .ivalue { color: #f1f5f9; font-size: 13px; font-weight: 500; }

/* allocation bars */
.abar-wrap { margin: 10px 0; }
.abar-labels { display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-bottom: 4px; }
.abar { height: 7px; border-radius: 99px; background: #1e1e24; overflow: hidden; margin-bottom: 12px; }
.abar-fill { height: 100%; border-radius: 99px; }

/* advice box */
.advice-box {
    margin-top: 10px; padding: 12px 14px;
    background: #052e16; border: 1px solid #22c55e;
    border-radius: 8px; font-size: 13px; color: #22c55e; line-height: 1.6;
}
/* reason box */
.reason-box {
    margin-top: 10px; padding: 12px 14px;
    background: #1a1a2e; border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0; font-size: 13px; color: #94a3b8; line-height: 1.6;
}

/* download button */
.stDownloadButton > button {
    background: #6366f1 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100%;
}

/* metrics */
[data-testid="stMetric"] { background: #141417; border: 1px solid #1e1e24; border-radius: 8px; padding: 14px !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 22px !important; }

/* tab labels */
button[data-baseweb="tab"] { color: #64748b !important; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ── plotly base ───────────────────────────────────────────────────────────────

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#141417",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=14, r=14, t=36, b=14),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
_AXIS = dict(gridcolor="#1e1e24", linecolor="#1e1e24", zerolinecolor="#1e1e24")


def _fig(f: go.Figure, **kw) -> go.Figure:
    layout = {**_LAYOUT, **kw}
    layout["xaxis"] = {**_AXIS, **layout.get("xaxis", {})}
    layout["yaxis"] = {**_AXIS, **layout.get("yaxis", {})}
    f.update_layout(**layout)
    return f


def _sh(text: str):
    st.markdown(f'<div class="sec-head">{text}</div>', unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_realtime  = st.toggle("Real-Time Market Data", value=True)
    manual_market = None
    if not use_realtime:
        manual_market = st.selectbox("Market Condition", ["Bullish", "Neutral", "Bearish"])

    st.divider()
    st.markdown("### 📂 Upload Data")
    with st.expander("Required columns"):
        st.markdown("""
        **Investors CSV**  
        `Investor_ID, Age, Income, Investment, Equity_Percent, Trades_Per_Month`

        **Holdings CSV**  
        `Investor_ID, Ticker, Quantity, Price, Sector`
        """)

    investor_file = st.file_uploader("Investors CSV", type=["csv"])
    holdings_file = st.file_uploader("Holdings CSV",  type=["csv"])


# ── landing ───────────────────────────────────────────────────────────────────

if not (investor_file and holdings_file):
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 30px">
        <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#6366f1;margin-bottom:16px">Portfolio Risk Analytics</div>
        <h1 style="font-size:52px;font-weight:700;color:#f1f5f9;margin:0 0 14px;line-height:1.1">
            AI Risk<br><span style="color:#6366f1">Intelligence</span>
        </h1>
        <p style="color:#64748b;font-size:15px;max-width:440px;margin:0 auto 40px;line-height:1.7">
            Dynamic portfolio risk profiling using ML clustering, live market signals,
            and rule-based scoring. Upload your CSVs to begin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📊", "Risk Profiling",    "Volatility, beta, drawdown & concentration into one risk score."),
        (c2, "🤖", "AI Clustering",     "K-Means segments investors into Conservative, Balanced, Aggressive."),
        (c3, "🔄", "Smart Rebalancing", "Equity/debt targets adjusted for live market conditions."),
        (c4, "💡", "Insights Engine",   "Per-investor narrative risk drivers and actionable advice."),
    ]:
        col.markdown(f"""
        <div style="background:#141417;border:1px solid #1e1e24;border-radius:10px;padding:22px;height:160px">
            <div style="font-size:26px;margin-bottom:10px">{icon}</div>
            <div style="font-size:14px;font-weight:600;color:#e2e8f0;margin-bottom:7px">{title}</div>
            <div style="font-size:13px;color:#64748b;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("⬅️ Upload both CSVs in the sidebar to begin.", icon="💡")
    st.stop()


# ── run pipeline ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running risk pipeline…")
def _run(inv_b: bytes, hld_b: bytes, realtime: bool, manual: str | None):
    inv = pd.read_csv(io.BytesIO(inv_b))
    hld = pd.read_csv(io.BytesIO(hld_b))
    return run_pipeline(inv, hld, realtime, manual)


df, market, summary = _run(
    investor_file.getvalue(),
    holdings_file.getvalue(),
    use_realtime,
    manual_market,
)

holdings = pd.read_csv(io.BytesIO(holdings_file.getvalue()))

# ── KPI strip ─────────────────────────────────────────────────────────────────

rd = summary.get("risk_distribution", {})
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("Total Investors",  summary.get("total_investors", len(df)))
k2.metric("🔴 High Risk",     rd.get("High",   0))
k3.metric("🟡 Medium Risk",   rd.get("Medium", 0))
k4.metric("🟢 Low Risk",      rd.get("Low",    0))
k5.metric("Avg Sharpe Ratio", summary.get("avg_sharpe", "—"))
k6.metric("Avg Alpha (pa)",   f"{summary.get('avg_alpha_pa', 0):.1%}")
k7.metric("Market Condition", market)

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⚠️ Risk Distribution",
    "🔄 Allocation",
    "💡 Insights",
    "🔍 Investor Drill-Down",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    # Scatter: volatility vs Sharpe coloured by persona
    _sh("Portfolio Volatility vs Sharpe Ratio")
    fig_s = _fig(px.scatter(
        df,
        x="Portfolio_Volatility",
        y="Sharpe_Ratio",
        color="Persona",
        symbol="Dynamic_Risk",
        hover_name="Investor_ID",
        size="Portfolio_Value" if "Portfolio_Value" in df.columns else None,
        color_discrete_sequence=PALETTE,
        labels={"Portfolio_Volatility": "Volatility (Daily σ)", "Sharpe_Ratio": "Sharpe Ratio"},
        size_max=22,
    ))
    st.plotly_chart(fig_s, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        _sh("3-Month Return vs Nifty Benchmark")
        fig_ex = _fig(px.histogram(
            df, x="Excess_Return", color="Dynamic_Risk",
            nbins=20, barmode="overlay", opacity=0.8,
            color_discrete_map=RISK_COLORS,
            labels={"Excess_Return": "Excess Return vs Nifty 50"},
        ))
        fig_ex.add_vline(x=0, line_dash="dash", line_color="#334155",
                         annotation_text="Benchmark", annotation_font_color="#64748b")
        st.plotly_chart(fig_ex, use_container_width=True)

    with c2:
        _sh("Beta Distribution by Risk Level")
        fig_b = _fig(px.histogram(
            df, x="Beta", color="Dynamic_Risk",
            nbins=20, barmode="overlay", opacity=0.8,
            color_discrete_map=RISK_COLORS,
            labels={"Beta": "Portfolio Beta"},
        ))
        fig_b.add_vline(x=1, line_dash="dash", line_color="#334155",
                        annotation_text="β = 1", annotation_font_color="#64748b")
        st.plotly_chart(fig_b, use_container_width=True)

    _sh("Max Drawdown by Investor Persona")
    fig_dd = _fig(px.box(
        df, x="Persona", y="Max_Drawdown", color="Persona",
        color_discrete_sequence=PALETTE, points="all",
        labels={"Max_Drawdown": "Max Drawdown", "Persona": "Investor Persona"},
    ))
    st.plotly_chart(fig_dd, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    # Three pie charts: rule / AI / dynamic risk
    pc1, pc2, pc3 = st.columns(3)
    for col, field, title in [
        (pc1, "Rule_Risk",    "Rule-Based Risk"),
        (pc2, "AI_Risk",      "AI (K-Means) Risk"),
        (pc3, "Dynamic_Risk", "Dynamic Risk (Market-Adjusted)"),
    ]:
        counts = df[field].value_counts().reset_index()
        counts.columns = ["Risk", "Count"]
        fig_pie = _fig(px.pie(
            counts, names="Risk", values="Count",
            title=title, color="Risk",
            color_discrete_map=RISK_COLORS, hole=0.45,
        ))
        fig_pie.update_traces(textfont_size=13, textfont_color="#f1f5f9")
        col.plotly_chart(fig_pie, use_container_width=True)

    # Agreement heatmap between rule and AI
    _sh("Rule-Based vs AI Risk Agreement")
    heat = pd.crosstab(df["Rule_Risk"], df["AI_Risk"])
    fig_h = _fig(px.imshow(
        heat, text_auto=True,
        color_continuous_scale=["#141417", "#6366f1"],
        labels={"x": "AI Risk", "y": "Rule-Based Risk", "color": "Count"},
    ))
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("This matrix shows how often Rule-Based and AI risk labels agree on each investor.")

    # Risk score distribution
    _sh("Risk Score Distribution")
    fig_rs = _fig(px.histogram(
        df, x="Risk_Score", color="Rule_Risk", nbins=20,
        barmode="overlay", opacity=0.8,
        color_discrete_map=RISK_COLORS,
        labels={"Risk_Score": "Composite Risk Score", "Rule_Risk": "Risk Level"},
    ))
    st.plotly_chart(fig_rs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALLOCATION
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    # Equity delta bar chart
    _sh("Equity Adjustment Needed per Investor")
    df_sorted   = df.sort_values("Equity_Delta")
    bar_colors  = df_sorted["Equity_Delta"].apply(
        lambda x: "#22c55e" if x > 0 else ("#ef4444" if x < 0 else "#6366f1")
    )
    fig_d = go.Figure(go.Bar(
        x=df_sorted["Investor_ID"].astype(str),
        y=df_sorted["Equity_Delta"],
        marker_color=bar_colors.tolist(),
        hovertemplate="<b>%{x}</b><br>Change needed: %{y:+.1f}pp<extra></extra>",
    ))
    fig_d.add_hline(y=0, line_color="#334155")
    _fig(fig_d,
         xaxis=dict(title="Investor", showticklabels=len(df) < 50, gridcolor="#1e1e24"),
         yaxis=dict(title="Equity Adjustment (percentage points)", gridcolor="#1e1e24"))
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption("Green bars = investor should hold more equity. Red = should reduce equity.")

    c1, c2 = st.columns(2)
    with c1:
        _sh("Rebalance Urgency")
        fig_u = _fig(px.histogram(
            df, x="Rebalance_Urgency", nbins=15, color="Dynamic_Risk",
            barmode="overlay", opacity=0.8, color_discrete_map=RISK_COLORS,
            labels={"Rebalance_Urgency": "Urgency Score (0–100)"},
        ))
        fig_u.add_vline(x=40, line_dash="dash", line_color="#f59e0b",
                        annotation_text="Action needed", annotation_font_color="#f59e0b")
        st.plotly_chart(fig_u, use_container_width=True)

    with c2:
        _sh("Direction Breakdown")
        dir_counts = df["Alloc_Direction"].value_counts().reset_index()
        dir_counts.columns = ["Direction", "Count"]
        fig_dir = _fig(px.pie(
            dir_counts, names="Direction", values="Count", hole=0.45,
            color="Direction",
            color_discrete_map={
                "Increase Equity": "#22c55e",
                "Reduce Equity":   "#ef4444",
                "On Target":       "#6366f1",
            },
        ))
        fig_dir.update_traces(textfont_size=13, textfont_color="#f1f5f9")
        st.plotly_chart(fig_dir, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    # Summary KPI cards
    _sh("Portfolio Summary")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, sub in [
        (m1, "Avg Annual Alpha",       f"{summary.get('avg_alpha_pa', 0):.1%}",          "vs Nifty 50"),
        (m2, "Avg 3-Month Excess",     f"{summary.get('avg_excess_3mo', 0):.1%}",        f"{summary.get('outperformers_3mo', 0)} outperformers"),
        (m3, "Need Rebalancing",       f"{summary.get('pct_need_rebalancing', 0)}%",     "urgency ≥ 40"),
        (m4, "Avg Rebalance Urgency",  f"{summary.get('avg_rebalance_urgency', 0):.0f}/100", f"avg equity Δ: {summary.get('avg_equity_delta', 0):+.1f}pp"),
    ]:
        col.markdown(f"""
        <div class="kcard">
            <div class="label">{label}</div>
            <div class="val">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Red flag panel
    _sh("Portfolio Red Flags")
    flags = [
        ("Deep drawdown (below −20%)",        f"{summary.get('deep_drawdown_pct', 0)}% of investors"),
        ("High concentration (HHI > 0.4)",    f"{summary.get('high_concentration_pct', 0)}% of investors"),
        ("High beta (β > 1.2)",               f"{summary.get('high_beta_pct', 0)}% of investors"),
        ("Negative alpha vs Nifty",           f"{summary.get('negative_alpha_pct', 0)}% of investors"),
        ("Underperforming benchmark (3mo)",   f"{summary.get('underperformers_pct', 0)}% of investors"),
    ]
    for label, val in flags:
        st.markdown(
            f'<div class="flag-row"><span class="flabel">{label}</span><span class="fval">{val}</span></div>',
            unsafe_allow_html=True,
        )

    # Top exposed sectors
    top_sec = summary.get("top_exposed_sectors", {})
    if top_sec:
        _sh("Most Concentrated Sectors")
        sec_df = pd.DataFrame(list(top_sec.items()), columns=["Sector", "Investors"])
        fig_sec = _fig(px.bar(
            sec_df, x="Sector", y="Investors",
            color="Sector", color_discrete_sequence=PALETTE,
            labels={"Investors": "Number of Investors with this as Top Sector"},
        ))
        st.plotly_chart(fig_sec, use_container_width=True)

    # Persona average metrics
    _sh("Average Metrics by Investor Persona")
    persona_avg = df.groupby("Persona")[[
        "Portfolio_Volatility", "Sharpe_Ratio", "Beta",
        "Ann_Alpha", "Max_Drawdown", "Concentration_Index",
    ]].mean().reset_index()

    fig_pa = _fig(px.bar(
        persona_avg.melt(id_vars="Persona"),
        x="variable", y="value",
        color="Persona", barmode="group",
        color_discrete_sequence=PALETTE,
        labels={"variable": "Metric", "value": "Average Value", "Persona": "Persona"},
    ))
    st.plotly_chart(fig_pa, use_container_width=True)

    # Calmar ratio distribution
    _sh("Calmar Ratio Distribution (Return / Max Drawdown)")
    fig_cal = _fig(px.histogram(
        df, x="Calmar_Ratio", color="Dynamic_Risk", nbins=20,
        barmode="overlay", opacity=0.8,
        color_discrete_map=RISK_COLORS,
        labels={"Calmar_Ratio": "Calmar Ratio"},
    ))
    st.plotly_chart(fig_cal, use_container_width=True)
    st.caption("Higher Calmar ratio = better return per unit of drawdown risk.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — INVESTOR DRILL-DOWN
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    investor_id = st.selectbox(
        "Select an Investor",
        df["Investor_ID"].sort_values().tolist(),
    )

    inv      = df[df["Investor_ID"] == investor_id].iloc[0]
    inv_hold = holdings[holdings["Investor_ID"] == investor_id].copy()
    inv_hold["Value"] = inv_hold["Quantity"] * inv_hold["Price"]

    risk_lbl   = inv["Dynamic_Risk"]
    badge_html = f'<span class="badge badge-{risk_lbl}">{risk_lbl}</span>'

    dc1, dc2 = st.columns(2)

    # ── left: profile + risk ─────────────────────────────────────────────
    with dc1:
        st.markdown(f"""
        <div class="icard">
            <h4>Investor Profile</h4>
            <div class="irow"><span class="ilabel">Age</span><span class="ivalue">{int(inv['Age'])}</span></div>
            <div class="irow"><span class="ilabel">Annual Income</span><span class="ivalue">₹{inv['Income']:,.0f}</span></div>
            <div class="irow"><span class="ilabel">Total Investment</span><span class="ivalue">₹{inv['Investment']:,.0f}</span></div>
            <div class="irow"><span class="ilabel">Trades per Month</span><span class="ivalue">{inv['Trades_Per_Month']}</span></div>
            <div class="irow"><span class="ilabel">Number of Holdings</span><span class="ivalue">{int(inv.get('Num_Holdings', 0))}</span></div>
            <div class="irow"><span class="ilabel">Sectors Invested</span><span class="ivalue">{int(inv.get('Num_Sectors', 0))}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="icard" style="margin-top:10px">
            <h4>Risk Assessment</h4>
            <div class="irow">
                <span class="ilabel">Rule-Based Risk</span>
                <span class="badge badge-{inv['Rule_Risk']}">{inv['Rule_Risk']}</span>
            </div>
            <div class="irow">
                <span class="ilabel">AI Persona</span>
                <span class="ivalue">{inv.get('Persona', '—')}</span>
            </div>
            <div class="irow">
                <span class="ilabel">Final (Market-Adjusted)</span>
                {badge_html}
            </div>
            <div class="irow"><span class="ilabel">Risk Score</span><span class="ivalue">{inv.get('Risk_Score', 0):.2f}</span></div>
            <div class="irow"><span class="ilabel">Beta</span><span class="ivalue">{inv['Beta']:.2f}</span></div>
            <div class="irow"><span class="ilabel">Sharpe Ratio</span><span class="ivalue">{inv['Sharpe_Ratio']:.3f}</span></div>
            <div class="irow"><span class="ilabel">Calmar Ratio</span><span class="ivalue">{inv.get('Calmar_Ratio', 0):.2f}</span></div>
            <div class="irow"><span class="ilabel">Annual Alpha</span><span class="ivalue">{inv.get('Ann_Alpha', 0):.2%}</span></div>
            <div class="irow"><span class="ilabel">Max Drawdown</span><span class="ivalue">{inv['Max_Drawdown']:.1%}</span></div>
            <div class="irow"><span class="ilabel">3-Month Return</span><span class="ivalue">{inv.get('Port_3mo_Return', 0):.1%}</span></div>
            <div class="irow"><span class="ilabel">Excess vs Nifty</span><span class="ivalue">{inv.get('Excess_Return', 0):.1%}</span></div>
            <div class="reason-box">💡 {inv.get('Risk_Reason', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── right: allocation + holdings ─────────────────────────────────────
    with dc2:
        eq_curr = float(inv["Equity_Percent"])
        eq_sugg = float(inv["Suggested_Equity"])
        dt_curr = float(inv["Debt_Percent"])
        dt_sugg = float(inv["Suggested_Debt"])
        eq_d    = float(inv["Equity_Delta"])
        urgency = float(inv.get("Rebalance_Urgency", 0))

        if eq_d > 5:
            direction_str = "🟢 Should increase equity"
        elif eq_d < -5:
            direction_str = "🔴 Should reduce equity"
        else:
            direction_str = "✅ On target"

        advice_items = inv.get("Advice", "Portfolio within target parameters").split(" | ")
        advice_html  = "<br>".join([f"• {a}" for a in advice_items])

        st.markdown(f"""
        <div class="icard">
            <h4>Allocation</h4>
            <div class="irow"><span class="ilabel">Recommendation</span><span class="ivalue">{direction_str}</span></div>
            <div class="irow"><span class="ilabel">Rebalance Urgency</span><span class="ivalue">{urgency:.0f} / 100</span></div>
            <div class="abar-wrap">
                <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Current</div>
                <div class="abar-labels"><span>Equity {eq_curr:.0f}%</span><span>Debt {dt_curr:.0f}%</span></div>
                <div class="abar"><div class="abar-fill" style="width:{eq_curr}%;background:#6366f1"></div></div>
                <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Suggested</div>
                <div class="abar-labels"><span>Equity {eq_sugg:.0f}%</span><span>Debt {dt_sugg:.0f}%</span></div>
                <div class="abar"><div class="abar-fill" style="width:{eq_sugg}%;background:#22c55e"></div></div>
            </div>
            <div class="advice-box">{advice_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Holdings charts (only if data exists)
        if not inv_hold.empty:
            sector_grp = inv_hold.groupby("Sector")["Value"].sum().reset_index()
            fig_sv = _fig(px.bar(
                sector_grp, x="Sector", y="Value",
                color="Sector", color_discrete_sequence=PALETTE,
                title="Portfolio by Sector",
                labels={"Value": "Market Value (₹)"},
            ))
            st.plotly_chart(fig_sv, use_container_width=True)

            fig_hw = _fig(px.pie(
                inv_hold, names="Ticker", values="Value",
                title="Holdings Weight", hole=0.4,
                color_discrete_sequence=PALETTE,
            ))
            fig_hw.update_traces(textfont_size=12, textfont_color="#f1f5f9")
            st.plotly_chart(fig_hw, use_container_width=True)

    st.divider()

    # Full table
    with st.expander("📋 Full Data Table — All Investors"):
        show_cols = [
            "Investor_ID", "Age", "Persona", "Rule_Risk", "AI_Risk", "Dynamic_Risk",
            "Risk_Score", "Sharpe_Ratio", "Calmar_Ratio", "Portfolio_Volatility",
            "Beta", "Ann_Alpha", "Max_Drawdown", "Excess_Return", "Port_3mo_Return",
            "Concentration_Index", "Num_Holdings", "Num_Sectors",
            "Equity_Percent", "Suggested_Equity", "Equity_Delta",
            "Debt_Percent", "Suggested_Debt", "Alloc_Direction",
            "Rebalance_Urgency", "Advice",
        ]
        show = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show], use_container_width=True)

    st.download_button(
        "⬇️  Download Full Report (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="risk_report.csv",
        mime="text/csv",
    )