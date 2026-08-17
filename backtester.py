"""
Moving Average Crossover Backtester
=====================================
A beginner-friendly quant trading project.

STRATEGY LOGIC:
- Compute a short-term Simple Moving Average (SMA) and a long-term SMA on price.
- BUY signal: when short SMA crosses ABOVE long SMA (bullish momentum, "Golden Cross").
- SELL signal: when short SMA crosses BELOW long SMA (bearish momentum, "Death Cross").
- We simulate holding 1 unit of the asset when in a "long" position, flat otherwise.

This file contains no UI code — it's pure logic, which is what interviewers
care about most. app.py is a thin Streamlit wrapper around this engine.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical daily OHLCV data for a ticker from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol/date range.")
    # yfinance sometimes returns MultiIndex columns for a single ticker — flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    return df


def generate_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """
    Add SMA columns and a position column (1 = long, 0 = flat) to the price dataframe.
    Position flips to 1 the day AFTER a golden cross, and 0 the day AFTER a death cross —
    this avoids lookahead bias (you can't trade on today's close using today's signal).
    """
    out = df.copy()
    out["sma_short"] = out["close"].rolling(window=short_window, min_periods=short_window).mean()
    out["sma_long"] = out["close"].rolling(window=long_window, min_periods=long_window).mean()

    # Raw signal: 1 when short SMA is above long SMA, else 0
    out["signal"] = 0
    out.loc[out["sma_short"] > out["sma_long"], "signal"] = 1

    # Shift by 1 day to simulate acting on yesterday's signal at today's open/close
    out["position"] = out["signal"].shift(1).fillna(0)

    return out


def run_backtest(df: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    """
    Simulate the strategy vs. a buy-and-hold benchmark.
    Returns a dataframe with daily returns, cumulative returns, and portfolio value
    for both the strategy and the benchmark.
    """
    out = df.copy()
    out["daily_return"] = out["close"].pct_change().fillna(0)

    # Strategy only earns the daily return on days it's in a "long" position
    out["strategy_return"] = out["daily_return"] * out["position"]

    out["cum_return_strategy"] = (1 + out["strategy_return"]).cumprod()
    out["cum_return_buyhold"] = (1 + out["daily_return"]).cumprod()

    out["portfolio_value_strategy"] = initial_capital * out["cum_return_strategy"]
    out["portfolio_value_buyhold"] = initial_capital * out["cum_return_buyhold"]

    return out


def compute_metrics(results: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Compute standard quant performance metrics for the strategy.
    These are the numbers you should be ready to explain in an interview.
    """
    strat_returns = results["strategy_return"]
    bh_returns = results["daily_return"]

    trading_days_per_year = 252

    def annualized_return(returns):
        cum = (1 + returns).prod()
        n_years = len(returns) / trading_days_per_year
        return cum ** (1 / n_years) - 1 if n_years > 0 else np.nan

    def annualized_vol(returns):
        return returns.std() * np.sqrt(trading_days_per_year)

    def sharpe_ratio(returns):
        ann_ret = annualized_return(returns)
        ann_vol = annualized_vol(returns)
        return (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    def max_drawdown(cum_returns):
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def win_rate(returns):
        active = returns[returns != 0]
        if len(active) == 0:
            return np.nan
        return (active > 0).mean()

    metrics = {
        "Strategy Total Return (%)": (results["cum_return_strategy"].iloc[-1] - 1) * 100,
        "Buy & Hold Total Return (%)": (results["cum_return_buyhold"].iloc[-1] - 1) * 100,
        "Strategy Annualized Return (%)": annualized_return(strat_returns) * 100,
        "Strategy Annualized Volatility (%)": annualized_vol(strat_returns) * 100,
        "Strategy Sharpe Ratio": sharpe_ratio(strat_returns),
        "Strategy Max Drawdown (%)": max_drawdown(results["cum_return_strategy"]) * 100,
        "Strategy Win Rate (%)": win_rate(strat_returns) * 100,
        "Number of Trades": int((results["position"].diff().abs() == 1).sum()),
    }
    return metrics


if __name__ == "__main__":
    # Quick sanity-check run from the command line
    prices = fetch_price_data("RELIANCE.NS", start="2019-01-01", end="2024-01-01")
    signals = generate_signals(prices, short_window=20, long_window=50)
    results = run_backtest(signals)
    metrics = compute_metrics(results)

    print("\n--- Backtest Results: RELIANCE.NS | SMA 20/50 Crossover ---\n")
    for k, v in metrics.items():
        print(f"{k:40s}: {v:,.2f}")
