"""
pipeline.py — AI Risk Intelligence Engine
==========================================
Stages:
  1. Data ingestion & validation
  2. Market data fetch (single yfinance call)
  3. Portfolio metrics (volatility, Sharpe, Beta, drawdown, etc.)
  4. Rule-based risk scoring
  5. K-Means persona clustering
  6. Risk label fusion (rule + AI)
  7. Dynamic market adjustment
  8. Allocation comparison & rebalancing suggestions
  9. Advice & risk reason generation
 10. Portfolio-level summary
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────

BENCHMARK = "^NSEI"
PERIOD    = "3mo"
INTERVAL  = "1d"

W_VOL   = 3.0
W_DD    = 3.0
W_CONC  = 2.0
W_BETA  = 2.0
W_TRADE = 0.15

HIGH_CONC       = 0.40
HIGH_BETA       = 1.20
DEEP_DD         = -0.20
POOR_SHARPE     = 0.05
HIGH_SECTOR_EXP = 0.50
HIGH_VOL_PCTILE = 0.60

ALLOC_TARGETS = {"Low": 45, "Medium": 60, "High": 75}
MARKET_DELTA  = {"Bullish": +5, "Neutral": 0, "Bearish": -10}

RISK_MAP   = {"Low": 1, "Medium": 2, "High": 3}
RISK_UNMAP = {1: "Low", 2: "Medium", 3: "High"}

BEAR_MAP = {"Low": "Medium", "Medium": "High",  "High": "High"}
BULL_MAP = {"Low": "Low",    "Medium": "Low",   "High": "Medium"}

CLUSTER_FEATURES = [
    "Age", "Income", "Trades_Per_Month",
    "Portfolio_Volatility", "Sharpe_Ratio", "Beta",
    "Concentration_Index", "Max_Drawdown",
]

REQUIRED_INV = {"Investor_ID", "Age", "Income", "Investment", "Equity_Percent", "Trades_Per_Month"}
REQUIRED_HLD = {"Investor_ID", "Ticker", "Quantity", "Price", "Sector"}


# ── 1. validation ─────────────────────────────────────────────────────────────

def load_and_validate(
    investor_data: pd.DataFrame,
    holdings_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    inv = investor_data.copy()
    hld = holdings_data.copy()

    missing_inv = REQUIRED_INV - set(inv.columns)
    missing_hld = REQUIRED_HLD - set(hld.columns)
    if missing_inv:
        raise ValueError(f"Investor CSV missing columns: {missing_inv}")
    if missing_hld:
        raise ValueError(f"Holdings CSV missing columns: {missing_hld}")

    for col in ["Age", "Income", "Investment", "Equity_Percent", "Trades_Per_Month"]:
        inv[col] = pd.to_numeric(inv[col], errors="coerce")

    hld["Quantity"] = pd.to_numeric(hld["Quantity"], errors="coerce").fillna(0)
    hld["Price"]    = pd.to_numeric(hld["Price"],    errors="coerce").fillna(0)

    if "Debt_Percent" not in inv.columns:
        inv["Debt_Percent"] = 100 - inv["Equity_Percent"]

    return inv, hld


# ── 2. market data ────────────────────────────────────────────────────────────

class MarketData:
    def __init__(self, stock_returns, bench_returns, bench_var, market_state, bench_3mo_ret):
        self.stock_returns  = stock_returns
        self.bench_returns  = bench_returns
        self.bench_var      = bench_var
        self.market_state   = market_state
        self.bench_3mo_ret  = bench_3mo_ret


def fetch_market_data(
    tickers: list[str],
    use_realtime: bool,
    manual_market: Optional[str],
) -> MarketData:

    all_symbols = list(set(tickers + [BENCHMARK]))

    raw = yf.download(
        all_symbols, period=PERIOD, interval=INTERVAL,
        progress=False, auto_adjust=True,
    )["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=all_symbols[0])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    returns = raw.pct_change().dropna(how="all")

    bench_col = BENCHMARK if BENCHMARK in returns.columns else None
    if bench_col:
        bench_returns = returns.pop(bench_col)
    else:
        bench_returns = pd.Series(dtype=float)
    stock_returns = returns

    bench_var = float(bench_returns.dropna().var()) + 1e-9

    if bench_col and bench_col in raw.columns:
        bp = raw[bench_col].dropna()
        bench_3mo_ret = float((bp.iloc[-1] / bp.iloc[0]) - 1) if len(bp) >= 2 else 0.0
    else:
        bench_3mo_ret = 0.0

    if use_realtime:
        try:
            daily_chg = float(bench_returns.dropna().mean())
            if daily_chg > 0.005:
                market_state = "Bullish"
            elif daily_chg < -0.005:
                market_state = "Bearish"
            else:
                market_state = "Neutral"
        except Exception:
            market_state = "Neutral"
    else:
        market_state = manual_market or "Neutral"

    return MarketData(stock_returns, bench_returns, bench_var, market_state, bench_3mo_ret)


# ── 3. portfolio metrics ──────────────────────────────────────────────────────

def _safe_cov(a: pd.Series, b: pd.Series) -> float:
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) < 2:
        return 0.0
    return float(np.cov(a.loc[idx].values, b.loc[idx].values)[0][1])


def _portfolio_returns(ret_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    valid = [t for t in weights.index if t in ret_df.columns]
    if not valid:
        return pd.Series(dtype=float)
    w = weights.reindex(valid).dropna()
    w = w / w.sum()
    return ret_df[w.index].dropna(how="all").dot(w)


def compute_portfolio_metrics(
    investors: pd.DataFrame,
    holdings: pd.DataFrame,
    mkt: MarketData,
) -> pd.DataFrame:

    hld = holdings.copy()
    hld["Value"] = hld["Quantity"] * hld["Price"]
    total_val    = hld.groupby("Investor_ID")["Value"].sum()
    hld["Weight"] = hld["Value"] / hld["Investor_ID"].map(total_val)

    bench_mean = float(mkt.bench_returns.dropna().mean())
    rows = []

    for inv_id, grp in hld.groupby("Investor_ID"):
        weights  = grp.set_index("Ticker")["Weight"]
        port_ret = _portfolio_returns(mkt.stock_returns, weights)

        if port_ret.empty or len(port_ret) < 2:
            continue

        vol    = float(port_ret.std())
        mean_r = float(port_ret.mean())

        sharpe  = mean_r / (vol + 1e-9)
        ann_ret = float((1 + mean_r) ** 252 - 1)

        cum    = (1 + port_ret).cumprod()
        peak   = cum.cummax()
        max_dd = float(((cum - peak) / peak).min())
        calmar = ann_ret / (abs(max_dd) + 1e-9)

        cov         = _safe_cov(port_ret, mkt.bench_returns)
        beta        = cov / mkt.bench_var
        daily_alpha = mean_r - beta * bench_mean
        ann_alpha   = float((1 + daily_alpha) ** 252 - 1)

        port_3mo_ret = float(cum.iloc[-1] - 1)
        excess_ret   = port_3mo_ret - mkt.bench_3mo_ret

        concentration = float((weights ** 2).sum())
        sector_wts    = grp.groupby("Sector")["Weight"].sum()
        top_sector    = sector_wts.idxmax()
        max_sector    = float(sector_wts.max())

        rows.append({
            "Investor_ID":          inv_id,
            "Portfolio_Value":      float(grp["Value"].sum()),
            "Portfolio_Volatility": vol,
            "Ann_Return":           ann_ret,
            "Port_3mo_Return":      port_3mo_ret,
            "Excess_Return":        excess_ret,
            "Sharpe_Ratio":         sharpe,
            "Calmar_Ratio":         calmar,
            "Max_Drawdown":         max_dd,
            "Beta":                 beta,
            "Ann_Alpha":            ann_alpha,
            "Concentration_Index":  concentration,
            "Max_Sector_Exposure":  max_sector,
            "Top_Sector":           top_sector,
            "Num_Sectors":          int(sector_wts.shape[0]),
            "Num_Holdings":         int(grp.shape[0]),
        })

    return investors.merge(pd.DataFrame(rows), on="Investor_ID", how="left")


# ── 4. rule-based risk scoring ────────────────────────────────────────────────

def _normalize(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def apply_rule_based(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    age_adj = np.where(data["Age"] < 30, 1.5,
              np.where(data["Age"] < 45, 1.0, 0.0))

    data["Risk_Score"] = (
        _normalize(data["Portfolio_Volatility"].fillna(0)) * W_VOL
        + _normalize(data["Max_Drawdown"].fillna(0).abs())  * W_DD
        + _normalize(data["Concentration_Index"].fillna(0)) * W_CONC
        + _normalize(data["Beta"].fillna(0).abs())          * W_BETA
        + data["Trades_Per_Month"].fillna(0)                * W_TRADE
        + age_adj
    )

    data["Rule_Risk"] = pd.cut(
        data["Risk_Score"],
        bins=[-np.inf, 4, 7, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return data


# ── 5. k-means clustering ─────────────────────────────────────────────────────

def apply_kmeans(data: pd.DataFrame) -> tuple[pd.DataFrame, KMeans]:
    data   = data.copy()
    feat   = data[CLUSTER_FEATURES].fillna(0)
    scaled = StandardScaler().fit_transform(feat)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(scaled)
    return data, kmeans


def map_clusters(data: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    centers = kmeans.cluster_centers_
    vol_idx = CLUSTER_FEATURES.index("Portfolio_Volatility")
    shp_idx = CLUSTER_FEATURES.index("Sharpe_Ratio")
    order   = sorted(range(3), key=lambda i: centers[i][vol_idx] - centers[i][shp_idx])

    persona_map = {order[0]: "Conservative", order[1]: "Balanced", order[2]: "Aggressive"}
    risk_map    = {"Conservative": "Low", "Balanced": "Medium", "Aggressive": "High"}

    data = data.copy()
    data["Persona"] = data["Cluster"].map(persona_map)
    data["AI_Risk"] = data["Persona"].map(risk_map)
    return data


# ── 6. fuse rule + ai ─────────────────────────────────────────────────────────

def apply_combined(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Final_Risk"] = data.apply(
        lambda r: RISK_UNMAP[round(
            (RISK_MAP.get(r["Rule_Risk"], 2) + RISK_MAP.get(r["AI_Risk"], 2)) / 2
        )],
        axis=1,
    )
    return data


# ── 7. dynamic market adjustment ─────────────────────────────────────────────

def apply_dynamic(data: pd.DataFrame, market: str) -> pd.DataFrame:
    data = data.copy()
    if market == "Bearish":
        data["Dynamic_Risk"] = data["Final_Risk"].map(BEAR_MAP)
    elif market == "Bullish":
        data["Dynamic_Risk"] = data["Final_Risk"].map(BULL_MAP)
    else:
        data["Dynamic_Risk"] = data["Final_Risk"]
    return data


# ── 8. allocation comparison ──────────────────────────────────────────────────

def compute_allocations(data: pd.DataFrame, market: str) -> pd.DataFrame:
    data  = data.copy()
    delta = MARKET_DELTA.get(market, 0)

    def _row(row):
        base_eq      = ALLOC_TARGETS.get(row["Dynamic_Risk"], 60)
        suggested_eq = float(np.clip(base_eq + delta, 20, 90))
        suggested_dt = 100 - suggested_eq
        current_eq   = float(row.get("Equity_Percent", 60))
        current_dt   = float(row.get("Debt_Percent",   40))
        eq_delta     = round(suggested_eq - current_eq, 1)
        dt_delta     = round(suggested_dt - current_dt, 1)

        if eq_delta > 5:
            direction = "Increase Equity"
        elif eq_delta < -5:
            direction = "Reduce Equity"
        else:
            direction = "On Target"

        urgency = float(np.clip(
            abs(eq_delta) * 1.5
            + RISK_MAP.get(row["Dynamic_Risk"], 2) * 10
            + abs(row.get("Max_Drawdown", 0)) * 50,
            0, 100,
        ))

        return pd.Series({
            "Suggested_Equity":  suggested_eq,
            "Suggested_Debt":    suggested_dt,
            "Equity_Delta":      eq_delta,
            "Debt_Delta":        dt_delta,
            "Alloc_Direction":   direction,
            "Rebalance_Urgency": round(urgency, 1),
        })

    return pd.concat([data, data.apply(_row, axis=1)], axis=1)


# ── 9. advice & risk reason ───────────────────────────────────────────────────

def generate_advice(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    def _advice(row):
        tips = []
        eq_d = row.get("Equity_Delta", 0)
        if eq_d > 5:
            tips.append(f"Increase equity by ~{eq_d:.0f}pp (target {row['Suggested_Equity']:.0f}%)")
        elif eq_d < -5:
            tips.append(f"Reduce equity by ~{abs(eq_d):.0f}pp (target {row['Suggested_Equity']:.0f}%)")
        if row.get("Concentration_Index", 0) > HIGH_CONC:
            tips.append("Diversify: single holding dominates portfolio")
        if row.get("Beta", 0) > HIGH_BETA:
            tips.append(f"Trim high-beta exposure (β = {row['Beta']:.2f})")
        if row.get("Max_Drawdown", 0) < DEEP_DD:
            tips.append(f"Add stop-loss discipline (max drawdown = {row['Max_Drawdown']:.1%})")
        if row.get("Max_Sector_Exposure", 0) > HIGH_SECTOR_EXP:
            tips.append(f"Reduce {row.get('Top_Sector', 'sector')} concentration ({row['Max_Sector_Exposure']:.0%})")
        if row.get("Sharpe_Ratio", 1) < POOR_SHARPE:
            tips.append("Poor risk-adjusted return — review underperformers")
        if row.get("Ann_Alpha", 0) < -0.05:
            tips.append(f"Underperforming benchmark by {abs(row['Ann_Alpha']):.1%} pa")
        return " | ".join(tips) if tips else "Portfolio within target parameters"

    data["Advice"] = data.apply(_advice, axis=1)
    return data


def generate_risk_reason(data: pd.DataFrame) -> pd.DataFrame:
    data    = data.copy()
    vol_p60 = data["Portfolio_Volatility"].quantile(HIGH_VOL_PCTILE)

    def _reason(row):
        flags = []
        if row.get("Portfolio_Volatility", 0) > vol_p60:
            flags.append("above-average volatility")
        if row.get("Max_Drawdown", 0) < DEEP_DD:
            flags.append(f"deep drawdown ({row['Max_Drawdown']:.1%})")
        if row.get("Beta", 0) > HIGH_BETA:
            flags.append(f"high market sensitivity (β {row['Beta']:.2f})")
        if row.get("Concentration_Index", 0) > HIGH_CONC:
            flags.append("concentrated holdings")
        if row.get("Max_Sector_Exposure", 0) > HIGH_SECTOR_EXP:
            flags.append(f"heavy {row.get('Top_Sector', '')} exposure")
        if row.get("Ann_Alpha", 0) < -0.05:
            flags.append("negative alpha vs Nifty")
        return ("Driven by: " + ", ".join(flags)) if flags else "Well-balanced portfolio"

    data["Risk_Reason"] = data.apply(_reason, axis=1)
    return data


# ── 10. portfolio summary ─────────────────────────────────────────────────────

def generate_portfolio_summary(data: pd.DataFrame, market: str) -> dict:
    n = len(data)
    if n == 0:
        return {}

    def _pct(mask) -> float:
        return round(int(mask.sum()) / n * 100, 1)

    return {
        "total_investors":        n,
        "market_condition":       market,
        "risk_distribution":      data["Dynamic_Risk"].value_counts().to_dict(),
        "persona_distribution":   data["Persona"].value_counts().to_dict() if "Persona" in data.columns else {},
        "avg_sharpe":             round(float(data["Sharpe_Ratio"].mean()), 3),
        "avg_beta":               round(float(data["Beta"].mean()), 2),
        "avg_alpha_pa":           round(float(data["Ann_Alpha"].mean()), 4),
        "avg_excess_3mo":         round(float(data["Excess_Return"].mean()), 4),
        "avg_ann_return":         round(float(data["Ann_Return"].mean()), 4),
        "avg_equity_delta":       round(float(data["Equity_Delta"].mean()), 1),
        "investors_on_target":    int((data["Alloc_Direction"] == "On Target").sum()),
        "avg_rebalance_urgency":  round(float(data["Rebalance_Urgency"].mean()), 1),
        "pct_need_rebalancing":   _pct(data["Rebalance_Urgency"] >= 40),
        "high_concentration_pct": _pct(data["Concentration_Index"] > HIGH_CONC),
        "high_beta_pct":          _pct(data["Beta"] > HIGH_BETA),
        "negative_alpha_pct":     _pct(data["Ann_Alpha"] < 0),
        "underperformers_pct":    _pct(data["Excess_Return"] < 0),
        "deep_drawdown_pct":      _pct(data["Max_Drawdown"] < DEEP_DD),
        "outperformers_3mo":      int((data["Excess_Return"] > 0).sum()),
        "top_exposed_sectors":    data["Top_Sector"].value_counts().head(3).to_dict()
                                  if "Top_Sector" in data.columns else {},
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    investor_data: pd.DataFrame,
    holdings_data: pd.DataFrame,
    use_realtime: bool = True,
    manual_market: Optional[str] = None,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Returns
    -------
    df      : enriched investor DataFrame
    market  : "Bullish" | "Neutral" | "Bearish"
    summary : portfolio-level insight dict
    """
    investors, holdings = load_and_validate(investor_data, holdings_data)

    mkt = fetch_market_data(
        holdings["Ticker"].unique().tolist(), use_realtime, manual_market
    )

    investors = compute_portfolio_metrics(investors, holdings, mkt)
    investors = apply_rule_based(investors)

    investors, kmeans = apply_kmeans(investors)
    investors = map_clusters(investors, kmeans)

    investors = apply_combined(investors)
    investors = apply_dynamic(investors, mkt.market_state)

    investors = compute_allocations(investors, mkt.market_state)
    investors = generate_advice(investors)

    investors = generate_risk_reason(investors)
    summary   = generate_portfolio_summary(investors, mkt.market_state)

    return investors, mkt.market_state, summary