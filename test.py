"""
Indian sector performance screener — async / concurrent version.

Changes vs. the original synchronous script:
  1. Sector screening + price downloads run CONCURRENTLY (via asyncio +
     a thread pool, since yfinance itself is a blocking/sync library).
  2. Each sector's stocks are additionally split into market-cap buckets
     (Large / Mid / Small Cap, using SEBI-style INR crore thresholds).
  3. For every (sector, market-cap bucket) combination we return the
     TOP 5 stocks by 1-month return.

NOTE ON CONCURRENCY: Yahoo Finance will rate-limit/throttle if you hit it
with too many simultaneous requests. A semaphore caps how many sectors
are processed in parallel (SECTOR_CONCURRENCY) — raise it if you have
a paid/otherwise more permissive data source, lower it if you see
errors like "Too Many Requests".

NOTE ON JUPYTER: If you run this inside Jupyter/Colab, there's already
a running event loop, so `asyncio.run(main())` will raise
"asyncio.run() cannot be called from a running event loop". In that
case just do:  results = await main()   in its own cell (remove the
`if __name__ == "__main__"` block), or `pip install nest_asyncio` and
call `nest_asyncio.apply()` before `asyncio.run(main())`.
"""

import asyncio
import concurrent.futures

import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================



PERIOD = "1mo"
INTERVAL = "1d"

TOP_N = 5

# How many sectors to screen/download in parallel at once.
# Keep this modest to avoid Yahoo Finance rate limiting.


# Thread pool for running yfinance's blocking calls off the event loop.


# --------------------------------------------------------------
# Market cap buckets (INR). 1 crore = 1e7.
#   Large Cap : >= 20,000 crore
#   Mid Cap   :  5,000 - 20,000 crore
#   Small Cap :  < 5,000 crore
# Adjust these thresholds to whatever convention you want to use.
# --------------------------------------------------------------
CRORE = 10_000_000  # 1e7

MARKET_CAP_BUCKETS = {
    "Large Cap": (20_000 * CRORE, float("inf")),
    "Mid Cap": (5_000 * CRORE, 20_000 * CRORE),
    "Small Cap": (0, 5_000 * CRORE),
}


def classify_market_cap(market_cap):
    """Return the bucket name for a given market cap value, or None."""
    if market_cap is None:
        return None
    try:
        market_cap = float(market_cap)
    except (TypeError, ValueError):
        return None

    for bucket, (low, high) in MARKET_CAP_BUCKETS.items():
        if low <= market_cap < high:
            return bucket
    return None


# ============================================================
# 1. GET STOCKS FOR A SECTOR (blocking — runs in a thread)
# ============================================================

def get_sector_stocks(sector):

    print(f"[{sector}] Screening stocks...")

    # query = EquityQuery(
    #     "and",
    #     [
    #         EquityQuery("eq", ["region", "in"]),
    #         EquityQuery("eq", ["sector", sector]),
    #     ],
    # )
    query = EquityQuery(
    "and",
    [
        EquityQuery("eq", ["exchange", "NSI"]),
        EquityQuery("eq", ["sector", sector]),
    ],
)
    try:
        result = yf.screen(
            query,
            size=250,
            sortField="ticker",
            sortAsc=True,
        )
    except Exception as e:
        print(f"[{sector}] Screen error: {e}")
        return []

    quotes = result.get("quotes", [])

    stocks = []

    for q in quotes:
        ticker = q.get("symbol")
        name = q.get("longName") or q.get("shortName")
        market_cap = q.get("marketCap")

        if ticker is None:
            continue

        stocks.append(
            {
                "ticker": ticker,
                "name": name,
                "market_cap": market_cap,
                "sector": sector,
                "market_cap_bucket": classify_market_cap(market_cap),
            }
        )

    print(f"[{sector}] Found {len(stocks)} stocks")

    return stocks


# ============================================================
# 2. DOWNLOAD PRICE DATA (blocking — runs in a thread)
# ============================================================

def download_prices(sector, tickers):

    if not tickers:
        return pd.DataFrame()

    print(f"[{sector}] Downloading prices for {len(tickers)} stocks...")

    try:
        prices = yf.download(
            tickers=tickers,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if prices.empty:
            return pd.DataFrame()

        close = prices["Close"]

        # yfinance returns a Series instead of a DataFrame when only
        # one ticker was requested.
        if isinstance(close, pd.Series):
            close = close.to_frame()
            close.columns = [tickers[0]]

        return close

    except Exception as e:
        print(f"[{sector}] Price download error: {e}")
        return pd.DataFrame()


# ============================================================
# 3. CALCULATE STOCK RETURNS
# ============================================================

def calculate_stock_returns(prices):

    if prices.empty:
        return pd.Series(dtype=float)

    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        return pd.Series(dtype=float)

    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]

    returns = ((last_price / first_price) - 1) * 100

    return returns.dropna()


# ============================================================
# 4. ASYNC WRAPPERS
# Run the blocking yfinance calls in a thread pool so multiple
# sectors can be in-flight (network I/O) at the same time.
# ============================================================

async def get_sector_stocks_async(sector, executor):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, get_sector_stocks, sector)


async def download_prices_async(sector, tickers, executor):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, download_prices, sector, tickers)


# ============================================================
# 5. PROCESS ONE SECTOR END-TO-END
# ============================================================

async def process_sector(sector, executor, semaphore):

    async with semaphore:

        stocks = await get_sector_stocks_async(sector, executor)

        if not stocks:
            return sector, {"stocks": [], "sector_summary": None, "top_by_bucket": {}}

        tickers = [s["ticker"] for s in stocks]

        prices = await download_prices_async(sector, tickers, executor)

        returns = calculate_stock_returns(prices)

    # -- everything below is pure CPU / pandas work, no need to hold
    #    the semaphore (which is there to cap concurrent network I/O)

    if returns.empty:
        return sector, {"stocks": stocks, "sector_summary": None, "top_by_bucket": {}}

    # Attach the 1-month return to each stock record
    for s in stocks:
        s["return_1m"] = returns.get(s["ticker"], np.nan)

    stocks_df = pd.DataFrame(stocks)
    priced_df = stocks_df.dropna(subset=["return_1m"]).copy()

    # -------------------- sector-level summary (equal + cap weighted) --------------------
    sector_summary = None

    if not priced_df.empty:

        equal_weighted_return = priced_df["return_1m"].mean()

        cap_df = priced_df.dropna(subset=["market_cap"]).copy()
        cap_df = cap_df[cap_df["market_cap"] > 0]

        if not cap_df.empty and cap_df["market_cap"].sum() > 0:
            weights = cap_df["market_cap"] / cap_df["market_cap"].sum()
            weighted_return = (cap_df["return_1m"] * weights).sum()
        else:
            weighted_return = np.nan

        best_row = priced_df.loc[priced_df["return_1m"].idxmax()]
        worst_row = priced_df.loc[priced_df["return_1m"].idxmin()]

        sector_summary = {
            "sector": sector,
            "stocks_found": len(stocks),
            "stocks_with_price_data": len(priced_df),
            "market_cap_weighted_return_1m": round(weighted_return, 2)
            if pd.notna(weighted_return)
            else np.nan,
            "equal_weighted_return_1m": round(equal_weighted_return, 2),
            "best_stock": best_row["ticker"],
            "best_stock_name": best_row["name"],
            "best_stock_return_1m": round(best_row["return_1m"], 2),
            "worst_stock": worst_row["ticker"],
            "worst_stock_name": worst_row["name"],
            "worst_stock_return_1m": round(worst_row["return_1m"], 2),
        }

    # -------------------- top 5 per market-cap bucket --------------------
    top_by_bucket = {}

    for bucket in MARKET_CAP_BUCKETS:

        bucket_df = priced_df[priced_df["market_cap_bucket"] == bucket]

        if bucket_df.empty:
            top_by_bucket[bucket] = pd.DataFrame()
            continue

        top5 = (
            bucket_df.sort_values("return_1m", ascending=False)
            .head(TOP_N)[["ticker", "name", "market_cap", "return_1m"]]
            .reset_index(drop=True)
        )
        top5["return_1m"] = top5["return_1m"].round(2)

        top_by_bucket[bucket] = top5

    return sector, {
        "stocks": stocks,
        "sector_summary": sector_summary,
        "top_by_bucket": top_by_bucket,
    }


# ============================================================
# 6. MAIN — fan out over all sectors concurrently
# ============================================================
SECTORS = [
    "Basic Materials"
]
# SECTORS = [
#     "Basic Materials",
#     "Communication Services",
#     "Consumer Cyclical",
#     "Consumer Defensive",
#     "Energy",
#     "Financial Services",
#     "Healthcare",
#     "Industrials",
#     "Real Estate",
#     "Technology",
#     "Utilities",
# ]

async def deepresearch(MAX_WORKERS = 10,SECTOR_CONCURRENCY = 5):

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    semaphore = asyncio.Semaphore(SECTOR_CONCURRENCY)

    tasks = [
        process_sector(sector, executor, semaphore)
        for sector in SECTORS
    ]

    results = await asyncio.gather(*tasks)

    executor.shutdown(wait=False)

    return dict(results)


# ============================================================
# 7. DISPLAY HELPERS
# ============================================================

def build_sector_summary_df(results):

    summaries = [
        data["sector_summary"]
        for data in results.values()
        if data["sector_summary"] is not None
    ]

    if not summaries:
        return pd.DataFrame()

    df = pd.DataFrame(summaries).sort_values(
        "market_cap_weighted_return_1m", ascending=False
    ).reset_index(drop=True)

    df.insert(0, "rank", range(1, len(df) + 1))

    return df


def build_top_stocks_df(results):
    """
    Flatten every sector's top-5-per-bucket tables into one long
    DataFrame: sector | market_cap_bucket | rank | ticker | name |
    market_cap | return_1m
    """
    rows = []

    for sector, data in results.items():
        for bucket, top5 in data["top_by_bucket"].items():
            if top5.empty:
                continue
            for i, row in top5.iterrows():
                rows.append(
                    {
                        "sector": sector,
                        "market_cap_bucket": bucket,
                        "rank": i + 1,
                        "ticker": row["ticker"],
                        "name": row["name"],
                        "market_cap": row["market_cap"],
                        "return_1m": row["return_1m"],
                    }
                )

    # return pd.DataFrame(rows)
    return rows


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # In a plain .py script / terminal, asyncio.run() works fine.
    # In Jupyter/Colab, replace this block with:  results = await main()
    results = asyncio.run(deepresearch())

    print("\n" + "=" * 100)
    print("SECTOR SUMMARY (market-cap weighted 1M return)")
    print("=" * 100)
    sector_summary_df = build_sector_summary_df(results)
    print(sector_summary_df)

    print("\n" + "=" * 100)
    print(f"TOP {TOP_N} STOCKS PER SECTOR, PER MARKET-CAP BUCKET")
    print("=" * 100)
    top_stocks_df = build_top_stocks_df(results)

    if top_stocks_df.empty:
        print("No results.")
    else:
        for sector in SECTORS:
            sector_rows = top_stocks_df[top_stocks_df["sector"] == sector]
            if sector_rows.empty:
                continue
            print(f"\n--- {sector} ---")
            for bucket in MARKET_CAP_BUCKETS:
                bucket_rows = sector_rows[sector_rows["market_cap_bucket"] == bucket]
                if bucket_rows.empty:
                    continue
                print(f"  {bucket}:")
                print(
                    bucket_rows[["rank", "ticker", "name", "market_cap", "return_1m"]]
                    .to_string(index=False)
                )
    breakpoint()