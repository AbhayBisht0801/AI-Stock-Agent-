"""Portfolio Insights - Streamlit dashboard.

Run:  streamlit run app.py

Reads a portfolio JSON (upload in the sidebar, or drop a `portfolio.json` next to this file).
Everything is driven by the `stocks` list, so adding a new stock to the JSON is all it takes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Streamlit >=1.50 wants width="stretch"; older versions only know use_container_width.
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _v = (0, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Portfolio Insights", page_icon="📈", layout="wide")

DEFAULT_PATH = Path(__file__).parent / "stocks.json"
CUR = "₹"
GREEN, RED, GREY = "#16a34a", "#dc2626", "#6b7280"


# --------------------------------------------------------------------------- helpers
def num(v) -> float:
    try:
        if v is None:
            return math.nan
        f = float(v)
        return f
    except (TypeError, ValueError):
        return math.nan


def money(x: float, decimals: int = 2) -> str:
    """Format in Indian style: Cr / L / plain."""
    if x is None or pd.isna(x):
        return "–"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}{CUR}{a / 1e7:,.{decimals}f} Cr"
    if a >= 1e5:
        return f"{sign}{CUR}{a / 1e5:,.{decimals}f} L"
    return f"{sign}{CUR}{a:,.{decimals}f}"


def pct(x: float, decimals: int = 2) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:+.{decimals}f}%"


def load_payload(raw: str | bytes) -> dict:
    data = json.loads(raw)
    if isinstance(data, list):
        data = {"stocks": data}
    if "stocks" not in data or not isinstance(data["stocks"], list):
        raise ValueError("JSON must contain a 'stocks' list.")
    return data


def build_frame(stocks: list[dict]) -> pd.DataFrame:
    rows = []
    for s in stocks:
        p = s.get("performance") or {}
        f = s.get("fundamentals") or {}
        sc = s.get("score") or {}
        qty, avg, cur = num(s.get("quantity")), num(s.get("avg_buy_price")), num(s.get("current_price"))
        invested, value = qty * avg, qty * cur
        score, max_score = num(sc.get("score")), num(sc.get("max_score"))
        rows.append(
            {
                "ticker": s.get("ticker", "?"),
                "company": f.get("company_name") or s.get("ticker", "?"),
                "sector": f.get("sector") or "Unknown",
                "industry": f.get("industry") or "Unknown",
                "quantity": qty,
                "avg_buy": avg,
                "price": cur,
                "invested": invested,
                "value": value,
                "pnl": value - invested,
                "pnl_pct": (value / invested - 1) * 100 if invested else math.nan,
                # performance
                "ret_1m": num(p.get("return_1m_pct")),
                "ret_3m": num(p.get("return_3m_pct")),
                "ret_6m": num(p.get("return_6m_pct")),
                "period_ret": num(p.get("total_return_pct")),
                "drawdown": num(p.get("drawdown_from_high_pct")),
                "volatility": num(p.get("annualized_volatility_pct")),
                "period_high": num(p.get("period_high")),
                "period_low": num(p.get("period_low")),
                "sma_50": num(p.get("sma_50")),
                "sma_200": num(p.get("sma_200")),
                "trend": p.get("trend") or "–",
                # fundamentals
                "market_cap": num(f.get("market_cap")),
                "pe": num(f.get("pe_ratio_trailing")),
                "pe_fwd": num(f.get("pe_ratio_forward")),
                "pb": num(f.get("price_to_book")),
                "profit_margin": num(f.get("profit_margin_pct")),
                "op_margin": num(f.get("operating_margin_pct")),
                "roe": num(f.get("roe_pct")),
                "rev_growth": num(f.get("revenue_growth_pct")),
                "earn_growth": num(f.get("earnings_growth_pct")),
                "debt_eq": num(f.get("debt_to_equity")),
                "current_ratio": num(f.get("current_ratio")),
                "div_yield": num(f.get("dividend_yield_pct")),
                "beta": num(f.get("beta")),
                "high_52w": num(f.get("52w_high")),
                "low_52w": num(f.get("52w_low")),
                # score
                "score": score,
                "max_score": max_score,
                "score_pct": score / max_score * 100 if max_score else math.nan,
                "verdict": sc.get("verdict") or "–",
                "notes": sc.get("notes") or [],
            }
        )
    df = pd.DataFrame(rows)
    total = df["value"].sum(skipna=True)
    df["weight"] = df["value"] / total * 100 if total else math.nan
    return df


def weighted(df: pd.DataFrame, col: str) -> float:
    d = df[[col, "value"]].dropna()
    return (d[col] * d["value"]).sum() / d["value"].sum() if d["value"].sum() else math.nan


# --------------------------------------------------------------------------- insights
def generate_insights(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (level, text) where level in {good, warn, bad, info}."""
    out: list[tuple[str, str]] = []
    tot_inv, tot_val = df["invested"].sum(), df["value"].sum()
    tot_pnl = tot_val - tot_inv
    tot_pct = (tot_val / tot_inv - 1) * 100 if tot_inv else math.nan

    out.append(
        ("good" if tot_pnl >= 0 else "bad",
         f"Portfolio is **{'up' if tot_pnl >= 0 else 'down'} {money(abs(tot_pnl))}** ({pct(tot_pct)}) on {money(tot_inv)} invested.")
    )

    # winners / losers
    n_loss = int((df["pnl"] < 0).sum())
    n_gain = int((df["pnl"] > 0).sum())
    out.append(("info", f"**{n_gain}** position(s) in profit, **{n_loss}** in loss vs. your buy price."))
    best, worst = df.loc[df["pnl_pct"].idxmax()], df.loc[df["pnl_pct"].idxmin()]
    if len(df) > 1:
        out.append(("good" if best.pnl_pct > 0 else "info", f"Best performer vs cost: **{best.ticker}** ({pct(best.pnl_pct)})."))
        out.append(("bad" if worst.pnl_pct < 0 else "info", f"Weakest vs cost: **{worst.ticker}** ({pct(worst.pnl_pct)}, {money(worst.pnl)})."))

    # concentration
    ds = df.sort_values("weight", ascending=False)
    top = ds.iloc[0]
    if len(df) > 1 and top.weight > 30:
        out.append(("warn", f"**Concentration:** {top.ticker} is {top.weight:.1f}% of the portfolio."))
    if len(df) >= 4 and ds["weight"].head(3).sum() > 70:
        out.append(("warn", f"Top 3 holdings make up {ds['weight'].head(3).sum():.1f}% of the portfolio."))
    sec = df.groupby("sector")["value"].sum().sort_values(ascending=False)
    if len(sec) and sec.iloc[0] / tot_val * 100 > 50 and len(df) > 1:
        out.append(("warn", f"**Sector tilt:** {sec.index[0]} is {sec.iloc[0] / tot_val * 100:.1f}% of the portfolio."))
    if len(df) == 1:
        out.append(("warn", "Only one holding - there is no diversification yet."))

    # deep losers / near lows
    deep = df[df["pnl_pct"] <= -20]
    if len(deep):
        out.append(("bad", "Down **20%+** from your cost: " + ", ".join(f"{r.ticker} ({pct(r.pnl_pct, 1)})" for r in deep.itertuples())))
    near_low = df[(df["low_52w"] > 0) & (df["price"] <= df["low_52w"] * 1.1)]
    if len(near_low):
        out.append(("warn", "Trading within 10% of the 52-week low: " + ", ".join(near_low.ticker)))
    dd = df[df["drawdown"] <= -25]
    if len(dd):
        out.append(("warn", "Drawdown of 25%+ from recent high: " + ", ".join(f"{r.ticker} ({r.drawdown:.1f}%)" for r in dd.itertuples())))
    hv = df[df["volatility"] >= 40]
    if len(hv):
        out.append(("warn", "High volatility (40%+ annualised): " + ", ".join(f"{r.ticker} ({r.volatility:.0f}%)" for r in hv.itertuples())))

    # price vs fundamentals divergence
    strong_down = df[(df["score_pct"] >= 70) & (df["pnl_pct"] < -10)]
    if len(strong_down):
        out.append(("info", "Strong fundamentals score but price below your cost - worth reviewing whether the thesis is intact: "
                    + ", ".join(strong_down.ticker)))
    weak_down = df[(df["score_pct"] <= 50) & (df["pnl_pct"] < 0)]
    if len(weak_down):
        out.append(("bad", "Weak fundamentals score **and** in loss: " + ", ".join(weak_down.ticker)))
    margin_press = df[(df["rev_growth"] > 0) & (df["earn_growth"] < 0)]
    if len(margin_press):
        out.append(("info", "Revenue growing but earnings shrinking (margin pressure?): " + ", ".join(margin_press.ticker)))

    # trend
    below50 = df[(df["sma_50"] > 0) & (df["price"] < df["sma_50"])]
    if len(below50):
        out.append(("info", "Trading below 50-day average: " + ", ".join(below50.ticker)))

    wscore = weighted(df, "score_pct")
    if not pd.isna(wscore):
        out.append(("info", f"Value-weighted fundamentals score: **{wscore:.0f}/100**."))
    return out


# --------------------------------------------------------------------------- load data
st.title("📈 Portfolio Insights")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload portfolio JSON", type=["json"])
    pasted = st.text_area("…or paste JSON", height=100, placeholder='{"stocks": [...]}')
    st.caption(f"Fallback: `{DEFAULT_PATH.name}` next to app.py")

raw = None
source = ""
if up is not None:
    raw, source = up.getvalue(), up.name
elif pasted.strip():
    raw, source = pasted, "pasted JSON"
elif DEFAULT_PATH.exists():
    raw, source = DEFAULT_PATH.read_text(encoding="utf-8"), DEFAULT_PATH.name

if raw is None:
    st.info("Upload a portfolio JSON in the sidebar to get started.")
    st.stop()

try:
    payload = load_payload(raw)
    full = build_frame(payload["stocks"])
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the JSON: {e}")
    st.stop()

if full.empty:
    st.warning("No stocks found in the file.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    sectors = sorted(full["sector"].unique())
    chosen = st.multiselect("Sector", sectors, default=sectors)
    only_loss = st.checkbox("Only positions in loss")

df = full[full["sector"].isin(chosen)]
if only_loss:
    df = df[df["pnl"] < 0]
if df.empty:
    st.warning("No holdings match the current filters.")
    st.stop()
df = df.copy()
df["weight"] = df["value"] / df["value"].sum() * 100

gen = payload.get("generated_at", "–")
st.caption(f"Source: {source} · generated {gen} · lookback {payload.get('months', '–')} months · {len(df)} holding(s)")

# --------------------------------------------------------------------------- KPIs
inv, val = df["invested"].sum(), df["value"].sum()
pnl = val - inv
pnl_pct = (val / inv - 1) * 100 if inv else math.nan

c = st.columns(5)
c[0].metric("Invested", money(inv))
c[1].metric("Current value", money(val))
c[2].metric("Total P&L", money(pnl), pct(pnl_pct))
c[3].metric("Holdings", len(df), f"{int((df.pnl > 0).sum())} up / {int((df.pnl < 0).sum())} down", delta_color="off")
ws = weighted(df, "score_pct")
c[4].metric("Fundamentals score", "–" if pd.isna(ws) else f"{ws:.0f}/100", help="Value-weighted average of each stock's score")

tab_over, tab_hold, tab_risk, tab_fund, tab_detail = st.tabs(
    ["Overview", "Holdings", "Performance & risk", "Fundamentals", "Stock detail"]
)

# --------------------------------------------------------------------------- Overview
with tab_over:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Key insights")
        icons = {"good": "🟢", "warn": "🟠", "bad": "🔴", "info": "🔵"}
        for level, text in generate_insights(df):
            st.markdown(f"{icons[level]} {text}")
        st.caption("Rule-based observations from your data - not investment advice.")
    with right:
        st.subheader("P&L by stock")
        d = df.sort_values("pnl")
        fig = px.bar(d, x="pnl", y="ticker", orientation="h",
                     color=d["pnl"] >= 0, color_discrete_map={True: GREEN, False: RED},
                     text=d["pnl_pct"].map(lambda v: pct(v, 1)))
        fig.update_layout(showlegend=False, height=max(300, 40 * len(d) + 100), xaxis_title=f"P&L ({CUR})", yaxis_title=None,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, **STRETCH)

    a, b = st.columns(2)
    with a:
        st.subheader("Allocation by stock")
        fig = px.pie(df, names="ticker", values="value", hole=0.55)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=340)
        st.plotly_chart(fig, **STRETCH)
    with b:
        st.subheader("Allocation by sector")
        sec = df.groupby("sector", as_index=False)["value"].sum()
        fig = px.pie(sec, names="sector", values="value", hole=0.55)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=340)
        st.plotly_chart(fig, **STRETCH)

    st.subheader("Invested vs current value")
    m = df.melt(id_vars="ticker", value_vars=["invested", "value"], var_name="type", value_name="amount")
    m["type"] = m["type"].map({"invested": "Invested", "value": "Current value"})
    fig = px.bar(m, x="ticker", y="amount", color="type", barmode="group",
                 color_discrete_map={"Invested": GREY, "Current value": "#2563eb"})
    fig.update_layout(xaxis_title=None, yaxis_title=CUR, legend_title=None, margin=dict(l=0, r=0, t=10, b=0), height=340)
    st.plotly_chart(fig, **STRETCH)

# --------------------------------------------------------------------------- Holdings
with tab_hold:
    st.subheader("All holdings")
    t = df[["ticker", "company", "sector", "quantity", "avg_buy", "price", "invested", "value",
            "pnl", "pnl_pct", "weight", "score", "max_score", "verdict"]].sort_values("value", ascending=False)

    def _color(v):
        if pd.isna(v):
            return ""
        return f"color: {GREEN}" if v > 0 else f"color: {RED}" if v < 0 else ""

    styler = t.style
    styler = (styler.map if hasattr(styler, "map") else styler.applymap)(_color, subset=["pnl", "pnl_pct"])
    st.dataframe(
        styler,
        **STRETCH,
        hide_index=True,
        column_config={
            "ticker": "Ticker", "company": "Company", "sector": "Sector",
            "quantity": st.column_config.NumberColumn("Qty", format="%.0f"),
            "avg_buy": st.column_config.NumberColumn("Avg buy", format=f"{CUR}%.2f"),
            "price": st.column_config.NumberColumn("Price", format=f"{CUR}%.2f"),
            "invested": st.column_config.NumberColumn("Invested", format=f"{CUR}%.0f"),
            "value": st.column_config.NumberColumn("Value", format=f"{CUR}%.0f"),
            "pnl": st.column_config.NumberColumn("P&L", format=f"{CUR}%.0f"),
            "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
            "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%", min_value=0, max_value=100),
            "score": st.column_config.NumberColumn("Score", format="%.0f"),
            "max_score": st.column_config.NumberColumn("Max", format="%.0f"),
            "verdict": "Verdict",
        },
    )
    st.download_button("Download table (CSV)", t.to_csv(index=False), "holdings.csv", "text/csv")

# --------------------------------------------------------------------------- Performance & risk
with tab_risk:
    st.subheader("Returns over time")
    pc = st.columns(4)
    pc[0].metric("1M (weighted)", pct(weighted(df, "ret_1m")))
    pc[1].metric("3M (weighted)", pct(weighted(df, "ret_3m")))
    pc[2].metric("6M (weighted)", pct(weighted(df, "ret_6m")))
    pc[3].metric("Volatility (weighted)", "–" if pd.isna(weighted(df, "volatility")) else f"{weighted(df, 'volatility'):.1f}%")

    mm = df.melt(id_vars="ticker", value_vars=["ret_1m", "ret_3m", "ret_6m"], var_name="period", value_name="ret")
    mm["period"] = mm["period"].map({"ret_1m": "1M", "ret_3m": "3M", "ret_6m": "6M"})
    fig = px.bar(mm, x="ticker", y="ret", color="period", barmode="group",
                 color_discrete_sequence=["#93c5fd", "#3b82f6", "#1e3a8a"])
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(xaxis_title=None, yaxis_title="Return %", legend_title=None, margin=dict(l=0, r=0, t=10, b=0), height=340)
    st.plotly_chart(fig, **STRETCH)

    a, b = st.columns(2)
    with a:
        st.subheader("Risk vs return")
        sc = df.dropna(subset=["volatility", "pnl_pct"]).copy()
        sc["size"] = sc["value"].clip(lower=1)
        if len(sc):
            fig = px.scatter(sc, x="volatility", y="pnl_pct", size="size", color="sector", text="ticker",
                             labels={"volatility": "Annualised volatility %", "pnl_pct": "P&L vs cost %"})
            fig.add_hline(y=0, line_color=GREY, line_dash="dot")
            fig.update_traces(textposition="top center")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
            st.plotly_chart(fig, **STRETCH)
        else:
            st.caption("Not enough data.")
    with b:
        st.subheader("Drawdown from period high")
        d = df.dropna(subset=["drawdown"]).sort_values("drawdown")
        fig = px.bar(d, x="drawdown", y="ticker", orientation="h", color_discrete_sequence=[RED])
        fig.update_layout(xaxis_title="%", yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0), height=360)
        st.plotly_chart(fig, **STRETCH)

    st.subheader("Price vs buy price & moving averages")
    cmp = df[["ticker", "avg_buy", "price", "sma_50", "sma_200", "period_low", "period_high", "trend"]].copy()
    cmp["vs_sma50_pct"] = (cmp["price"] / cmp["sma_50"] - 1) * 100
    st.dataframe(cmp, **STRETCH, hide_index=True, column_config={
        "ticker": "Ticker",
        "avg_buy": st.column_config.NumberColumn("Avg buy", format="%.2f"),
        "price": st.column_config.NumberColumn("Price", format="%.2f"),
        "sma_50": st.column_config.NumberColumn("SMA 50", format="%.2f"),
        "sma_200": st.column_config.NumberColumn("SMA 200", format="%.2f"),
        "period_low": st.column_config.NumberColumn("Period low", format="%.2f"),
        "period_high": st.column_config.NumberColumn("Period high", format="%.2f"),
        "vs_sma50_pct": st.column_config.NumberColumn("vs SMA50", format="%.1f%%"),
        "trend": "Trend",
    })

# --------------------------------------------------------------------------- Fundamentals
with tab_fund:
    st.subheader("Fundamentals score")
    d = df.sort_values("score_pct")
    fig = px.bar(d, x="score_pct", y="ticker", orientation="h", text=d.apply(lambda r: f"{r.score:.0f}/{r.max_score:.0f}" if pd.notna(r.score) else "", axis=1),
                 color="score_pct", color_continuous_scale=["#dc2626", "#f59e0b", "#16a34a"], range_color=[0, 100])
    fig.update_layout(xaxis_title="Score %", yaxis_title=None, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0), height=max(240, 40 * len(d) + 80))
    st.plotly_chart(fig, **STRETCH)

    st.subheader("Side-by-side")
    ft = df[["ticker", "sector", "market_cap", "pe", "pe_fwd", "pb", "roe", "profit_margin", "op_margin",
             "rev_growth", "earn_growth", "debt_eq", "current_ratio", "div_yield", "beta"]].copy()
    ft["market_cap"] = ft["market_cap"].map(money)
    st.dataframe(ft, **STRETCH, hide_index=True, column_config={
        "ticker": "Ticker", "sector": "Sector", "market_cap": "Mkt cap",
        "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        "pe_fwd": st.column_config.NumberColumn("Fwd P/E", format="%.1f"),
        "pb": st.column_config.NumberColumn("P/B", format="%.2f"),
        "roe": st.column_config.NumberColumn("ROE %", format="%.1f"),
        "profit_margin": st.column_config.NumberColumn("Net margin %", format="%.1f"),
        "op_margin": st.column_config.NumberColumn("Op margin %", format="%.1f"),
        "rev_growth": st.column_config.NumberColumn("Rev growth %", format="%.1f"),
        "earn_growth": st.column_config.NumberColumn("Earn growth %", format="%.1f"),
        "debt_eq": st.column_config.NumberColumn("D/E", format="%.2f"),
        "current_ratio": st.column_config.NumberColumn("Current ratio", format="%.2f"),
        "div_yield": st.column_config.NumberColumn("Div yield %", format="%.2f"),
        "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
    })

    a, b = st.columns(2)
    with a:
        st.subheader("Growth: revenue vs earnings")
        g = df.melt(id_vars="ticker", value_vars=["rev_growth", "earn_growth"], var_name="m", value_name="v")
        g["m"] = g["m"].map({"rev_growth": "Revenue", "earn_growth": "Earnings"})
        fig = px.bar(g, x="ticker", y="v", color="m", barmode="group", color_discrete_sequence=["#3b82f6", "#f59e0b"])
        fig.add_hline(y=0, line_color=GREY)
        fig.update_layout(xaxis_title=None, yaxis_title="%", legend_title=None, margin=dict(l=0, r=0, t=10, b=0), height=320)
        st.plotly_chart(fig, **STRETCH)
    with b:
        st.subheader("Valuation vs profitability")
        v = df.dropna(subset=["pe", "roe"]).copy()
        v["size"] = v["value"].clip(lower=1)
        if len(v):
            fig = px.scatter(v, x="pe", y="roe", size="size", color="sector", text="ticker",
                             labels={"pe": "Trailing P/E", "roe": "ROE %"})
            fig.update_traces(textposition="top center")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig, **STRETCH)
        else:
            st.caption("Not enough data.")

# --------------------------------------------------------------------------- Stock detail
with tab_detail:
    pick = st.selectbox("Select stock", df["ticker"].tolist())
    r = df[df["ticker"] == pick].iloc[0]
    st.subheader(f"{r.company} ({r.ticker})")
    st.caption(f"{r.sector} · {r.industry} · Market cap {money(r.market_cap)}")

    k = st.columns(5)
    k[0].metric("Price", f"{CUR}{r.price:,.2f}", pct((r.price / r.avg_buy - 1) * 100) + " vs buy")
    k[1].metric("Qty × avg buy", f"{r.quantity:,.0f} × {CUR}{r.avg_buy:,.2f}")
    k[2].metric("Invested", money(r.invested))
    k[3].metric("Value", money(r.value))
    k[4].metric("P&L", money(r.pnl), pct(r.pnl_pct))

    a, b = st.columns(2)
    with a:
        st.markdown(f"**Verdict:** {r.verdict} ({r.score:.0f}/{r.max_score:.0f})" if pd.notna(r.score) else f"**Verdict:** {r.verdict}")
        for n in r.notes:
            st.markdown(f"- {n}")
        st.markdown(f"**Trend:** {r.trend}")
    with b:
        st.markdown("**Price levels**")
        lv = pd.DataFrame(
            {"Level": ["Avg buy", "Current", "SMA 50", "SMA 200", "Period low", "Period high", "52W low", "52W high"],
             "Price": [r.avg_buy, r.price, r.sma_50, r.sma_200, r.period_low, r.period_high, r.low_52w, r.high_52w]}
        ).dropna()
        fig = px.bar(lv, x="Price", y="Level", orientation="h", color_discrete_sequence=["#3b82f6"])
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0), height=300, yaxis_title=None)
        st.plotly_chart(fig, **STRETCH)

st.divider()
st.caption("Data-driven summaries only - not investment advice.")