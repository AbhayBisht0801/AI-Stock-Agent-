

import sys
import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import pandas as pd
import yfinance as yf
from langchain.tools import tool
import asyncio
from growwapi import GrowwAPI
from dotenv import load_dotenv
load_dotenv()
import os
import pyotp
api_key = os.environ["GROWW_TOTP_TOKEN"]
totp_secret = os.environ["GROWW_TOTP_SECRET"]


# Generate current TOTP
totp = pyotp.TOTP(totp_secret).now()
access_token = GrowwAPI.get_access_token(
    api_key=api_key,
    totp=totp
)

groww = GrowwAPI(access_token)
# Generate access token
access_token = GrowwAPI.get_access_token(
    api_key=api_key,
    totp=totp
)

# --------------------------------------------------------------------------
# Price performance
# --------------------------------------------------------------------------

def get_price_performance(ticker: str, months: int = 6) -> dict:
    """Fetch price history and compute performance stats over the window."""
    stock = yf.Ticker(ticker)
    end = datetime.today()
    start = end - timedelta(days=months * 31)

    hist = stock.history(start=start, end=end, interval="1d")
    if hist.empty:
        raise ValueError(f"No price data found for '{ticker}'. Check the symbol.")

    close = hist["Close"]
    start_price = close.iloc[0]
    end_price = close.iloc[-1]

    total_return_pct = (end_price / start_price - 1) * 100

    # Rolling windows for 1m / 3m / 6m returns (if enough data exists)
    def pct_change_over_days(days):
        if len(close) < 2:
            return None
        target_date = close.index[-1] - pd.Timedelta(days=days)
        past = close[close.index <= target_date]
        if past.empty:
            return None
        return (end_price / past.iloc[-1] - 1) * 100

    daily_returns = close.pct_change().dropna()
    volatility_annualized = daily_returns.std() * (252 ** 0.5) * 100

    high = close.max()
    low = close.min()
    drawdown_from_high = (end_price / high - 1) * 100

    # Simple moving averages for trend context
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    return {
        "start_date": close.index[0].date(),
        "end_date": close.index[-1].date(),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "total_return_pct": round(total_return_pct, 2),
        "return_1m_pct": _safe_round(pct_change_over_days(30)),
        "return_3m_pct": _safe_round(pct_change_over_days(90)),
        "return_6m_pct": _safe_round(pct_change_over_days(180)),
        "period_high": round(high, 2),
        "period_low": round(low, 2),
        "drawdown_from_high_pct": round(drawdown_from_high, 2),
        "annualized_volatility_pct": round(volatility_annualized, 2),
        "sma_50": round(sma_50, 2) if sma_50 else None,
        "sma_200": round(sma_200, 2) if sma_200 else None,
        "trend": _trend_label(end_price, sma_50, sma_200),
    }


def _safe_round(val, n=2):
    return round(val, n) if val is not None else None


def _trend_label(price, sma_50, sma_200):
    if sma_50 and sma_200:
        if price > sma_50 > sma_200:
            return "Uptrend (price > 50DMA > 200DMA)"
        elif price < sma_50 < sma_200:
            return "Downtrend (price < 50DMA < 200DMA)"
        else:
            return "Mixed / sideways"
    return "Not enough data for trend (need 200+ trading days)"


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    return {
        "company_name": info.get("longName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio_trailing": info.get("trailingPE"),
        "pe_ratio_forward": info.get("forwardPE"),
        "peg_ratio": info.get("trailingPegRatio"),
        "price_to_book": info.get("priceToBook"),
        "profit_margin_pct": _pct(info.get("profitMargins")),
        "operating_margin_pct": _pct(info.get("operatingMargins")),
        "roe_pct": _pct(info.get("returnOnEquity")),
        "roa_pct": _pct(info.get("returnOnAssets")),
        "revenue_growth_pct": _pct(info.get("revenueGrowth")),
        "earnings_growth_pct": _pct(info.get("earningsGrowth")),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "free_cash_flow": info.get("freeCashflow"),
        "dividend_yield_pct": _pct(info.get("dividendYield")),
        "beta": info.get("beta"),
        "eps_trailing": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }


def _pct(val):
    return round(val * 100, 2) if isinstance(val, (int, float)) else None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_fundamentals(f: dict) -> dict:
    """
    Simple rule-of-thumb scoring (0-6). Not investment advice —
    just a quick heuristic screen to flag areas worth digging into.
    """
    score = 0
    notes = []

    pe = f.get("pe_ratio_trailing")
    if pe is not None:
        if 0 < pe <= 25:
            score += 1
            notes.append(f"P/E of {pe} is reasonable (<=25).")
        else:
            notes.append(f"P/E of {pe} is high or negative — richly valued or unprofitable.")

    margin = f.get("profit_margin_pct")
    if margin is not None:
        if margin >= 10:
            score += 1
            notes.append(f"Profit margin {margin}% is healthy (>=10%).")
        else:
            notes.append(f"Profit margin {margin}% is thin/negative.")

    roe = f.get("roe_pct")
    if roe is not None:
        if roe >= 15:
            score += 1
            notes.append(f"ROE {roe}% is strong (>=15%).")
        else:
            notes.append(f"ROE {roe}% is below the 15% strong-quality bar.")

    rev_growth = f.get("revenue_growth_pct")
    if rev_growth is not None:
        if rev_growth >= 5:
            score += 1
            notes.append(f"Revenue growth {rev_growth}% is solid (>=5%).")
        else:
            notes.append(f"Revenue growth {rev_growth}% is sluggish or shrinking.")

    dte = f.get("debt_to_equity")
    if dte is not None:
        if dte <= 100:
            score += 1
            notes.append(f"Debt/Equity {dte} is manageable (<=100).")
        else:
            notes.append(f"Debt/Equity {dte} is elevated — higher leverage risk.")

    cr = f.get("current_ratio")
    if cr is not None:
        if cr >= 1.2:
            score += 1
            notes.append(f"Current ratio {cr} shows solid short-term liquidity (>=1.2).")
        else:
            notes.append(f"Current ratio {cr} is tight — watch short-term liquidity.")

    max_score = 6
    verdict = (
        "Strong fundamentals" if score >= 5 else
        "Decent fundamentals" if score >= 3 else
        "Weak / mixed fundamentals"
    )

    return {"score": score, "max_score": max_score, "verdict": verdict, "notes": notes}
@tool
async def stock_price_budget_check(symbol:str,quantity:int,price:float,balance_available:float)->bool:
    """Checks if the total cost of buying a stock is within the available balance."""
    total_cost = quantity * price
    if total_cost > balance_available:
        return f'There is not enough balance to buy the stock {symbol} with quantity {quantity} at price {price}. Total cost is {total_cost} and available balance is {balance_available}.'
    return f'There is enough balance to buy the stock {symbol} with quantity {quantity} at price {price}. Total cost is {total_cost} and available balance is {balance_available}.'
# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
@tool
async def print_report(ticker: str, months: int):
    """Take input Stock name Capitalized stock name input like Adani Power is the stock name it should be passed as ADANIPOWER.NS
    with the duration in which Ananlysis need to been done like 1 month or 6  month so on the the input is number of months.
    This Tool Generates the Stock performance report how good the stock is """
    print("\n" + "=" * 70)
    print(f"  STOCK ANALYSIS: {ticker.upper()}")
    print("=" * 70)

    try:
        perf = get_price_performance(ticker, months)
    except Exception as e:
        print(f"[Price data error] {e}")
        return

    try:
        fund = get_fundamentals(ticker)
        print(fund.keys())
    except Exception as e:
        print(f"[Fundamentals data error] {e}")
        fund = {}

    print(f"\n{fund.get('company_name', ticker)} | {fund.get('sector', 'N/A')} / {fund.get('industry', 'N/A')}")

    print(f"\n--- Price Performance ({perf['start_date']} to {perf['end_date']}) ---")
    print(f"  Start price:        ₹{perf['start_price']}")
    print(f"  Current price:      ₹{perf['end_price']}")
    print(f"  Total return:       {perf['total_return_pct']}%")
    print(f"  1-month return:     {perf['return_1m_pct']}%")
    print(f"  3-month return:     {perf['return_3m_pct']}%")
    print(f"  6-month return:     {perf['return_6m_pct']}%")
    print(f"  Period high/low:    ₹{perf['period_high']} / ₹{perf['period_low']}")
    print(f"  Drawdown from high: {perf['drawdown_from_high_pct']}%")
    print(f"  Annualized vol:     {perf['annualized_volatility_pct']}%")
    print(f"  50DMA / 200DMA:     {perf['sma_50']} / {perf['sma_200']}")
    print(f"  Trend:              {perf['trend']}")

    if fund:
        print(f"\n--- Fundamentals ---")
        print(f"  Market cap:         {fund.get('market_cap')}")

        print(f"  P/E (trailing/fwd): {fund.get('pe_ratio_trailing')} / {fund.get('pe_ratio_forward')}")
        print(f"  PEG ratio:          {fund.get('peg_ratio')}")
        print(f"  Price/Book:         {fund.get('price_to_book')}")
        print(f"  Profit margin:      {fund.get('profit_margin_pct')}%")
        print(f"  Operating margin:   {fund.get('operating_margin_pct')}%")
        print(f"  ROE:                {fund.get('roe_pct')}%")
        print(f"  ROA:                {fund.get('roa_pct')}%")
        print(f"  Revenue growth:     {fund.get('revenue_growth_pct')}%")
        print(f"  Earnings growth:    {fund.get('earnings_growth_pct')}%")
        print(f"  Debt/Equity:        {fund.get('debt_to_equity')}")
        print(f"  Current ratio:      {fund.get('current_ratio')}")
        print(f"  Dividend yield:     {fund.get('dividend_yield_pct')}%")
        print(f"  Beta:               {fund.get('beta')}")
        print(f"  52w range:          ₹{fund.get('52w_low')} - ₹{fund.get('52w_high')}")

        result = score_fundamentals(fund)
        print(f"\n--- Fundamental Score: {result['score']}/{result['max_score']} -> {result['verdict']} ---")
        for note in result["notes"]:
            print(f"  - {note}")
        

    print("\n(Data via Yahoo Finance / yfinance. This is a heuristic screen, not investment advice.)")
    print("=" * 70 + "\n")
    return result






# Your holdings: ticker -> average buy price and quantity
# HOLDINGS = {
#     "ADANIPOWER.NS": {"avg_buy_price": 520.0, "quantity": 200},

# }

OUTPUT_FILE = "stocks.json"


def _clean(obj):
    """Make values JSON-safe: numpy/pandas scalars, Timestamps, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):  # numpy scalar
        try:
            obj = obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)  # dates, Timestamps, anything else


def get_stock_report(ticker: str, months: int, avg_buy_price=None, quantity=1) -> dict:
    """Same data print_report prints, returned as one dict (no printing)."""
    ticker = ticker.upper()
    perf = get_price_performance(ticker, months)  # raises if price data is missing

    try:
        fund = get_fundamentals(ticker) or {}
    except Exception as e:
        print(f"[{ticker}] fundamentals error: {e}")
        fund = {}

    try:
        score = score_fundamentals(fund) if fund else {}
    except Exception as e:
        print(f"[{ticker}] scoring error: {e}")
        score = {}

    return _clean({
        "ticker": ticker,
        "quantity": quantity,
        "avg_buy_price": avg_buy_price,
        "current_price": perf.get("end_price"),
        "performance": perf,
        "fundamentals": fund,
        "score": score,
    })


def export_stocks_json(holdings: dict = None, months: int = 6, path: str = OUTPUT_FILE) -> dict:
    """Run the report for every holding and write the dashboard JSON."""
    holding_result = groww.get_holdings_for_user()
    holdings=dict()
    
    for i in holding_result['holdings']:
        holdings[f"{i['trading_symbol']}.NS"] = {
    "quantity": i["quantity"],
    "avg_buy_price": i["average_price"],
}
    print(holdings)
    stocks, failed = [], []

    for ticker, h in holdings.items():
        try:
            stocks.append(get_stock_report(
                ticker, months,
                avg_buy_price=h.get("avg_buy_price"),
                quantity=h.get("quantity", 1),
            ))
            print(f"OK   {ticker}")
        except Exception as e:
            failed.append({"ticker": ticker, "error": str(e)})
            print(f"FAIL {ticker}: {e}")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "months": months,
        "stocks": stocks,
        "failed": failed,
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(stocks)} stocks to {path}" + (f" ({len(failed)} failed)" if failed else ""))
    return payload


# Async wrapper so it can sit next to your @tool functions.
# If you use it as a tool, add the @tool decorator above it.
async def export_stock_json( months: int = 6, path: str = OUTPUT_FILE):
    holding_result = groww.get_holdings_for_user()
    r
    holdings=dict()
    
    for i in holding_result['holdings']:
        holdings[f"{i['trading_symbol']}.NS"] = {
    "quantity": i["quantity"],
    "avg_buy_price": i["average_price"],
}
    print(holdings)
    return await asyncio.to_thread(export_stocks_json, holdings, months, path)





if __name__ == "__main__":
    export_stocks_json(months=6)