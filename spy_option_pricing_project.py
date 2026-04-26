"""
SPY Stochastic Option Pricing Project
=====================================

This script implements an end-to-end stochastic option pricing framework:

1. Defines a European call/put option on SPY
2. Simulates Brownian Motion and Geometric Brownian Motion
3. Implements Black-Scholes analytical call/put pricing
4. Prices European calls using Monte Carlo simulation
5. Computes Greeks: Delta, Gamma, Vega, Theta, Rho
6. Estimates historical volatility from SPY data using yfinance
7. Uses calibrated volatility in Black-Scholes pricing
8. Extracts implied volatility from market option prices
9. Produces model risk diagnostics

Run locally:
    pip install -r requirements.txt
    python spy_option_pricing_project.py

Note:
    Historical data and option-chain sections require internet access.
"""

from math import log, sqrt, exp, erf, pi
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project parameters
# ============================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

S0 = 500.0
K = 500.0
T = 1.0
r = 0.05
mu = 0.08
sigma = 0.20

N_STEPS = 252
N_PATHS = 50
N_SIMULATIONS = 100_000
SEED = 42


# ============================================================
# Basic normal distribution utilities
# ============================================================

def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    """Standard normal probability density function."""
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x ** 2)


# ============================================================
# Step 1: Payoff functions
# ============================================================

def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0.0)


def put_payoff(S_T, K):
    return np.maximum(K - S_T, 0.0)


def plot_call_payoff(K=K):
    S_T_grid = np.linspace(0.6 * K, 1.4 * K, 400)
    payoff = call_payoff(S_T_grid, K)

    plt.figure(figsize=(8, 5))
    plt.plot(S_T_grid, payoff)
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("SPY price at maturity (S_T)")
    plt.ylabel("Call option payoff")
    plt.title("European Call Option Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_call_payoff.png", dpi=150)
    plt.close()


# ============================================================
# Step 2: Brownian Motion and GBM simulation
# ============================================================

def simulate_brownian_motion(n_paths=N_PATHS, n_steps=N_STEPS, T=T, seed=SEED):
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))
    dW = np.sqrt(dt) * Z
    W = np.cumsum(dW, axis=1)
    W = np.column_stack([np.zeros(n_paths), W])
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, W


def simulate_gbm_paths(S0=S0, mu=mu, sigma=sigma, T=T, n_paths=N_PATHS, n_steps=N_STEPS, seed=SEED):
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    S_paths = S0 * np.exp(log_paths)
    S_paths = np.column_stack([np.full(n_paths, S0), S_paths])
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, S_paths


def plot_brownian_and_gbm():
    time_grid, W = simulate_brownian_motion()

    plt.figure(figsize=(8, 5))
    for i in range(W.shape[0]):
        plt.plot(time_grid, W[i])
    plt.xlabel("Time (years)")
    plt.ylabel("W_t")
    plt.title("Simulated Brownian Motion Paths")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_brownian_motion_paths.png", dpi=150)
    plt.close()

    time_grid, S_paths = simulate_gbm_paths()

    plt.figure(figsize=(8, 5))
    for i in range(S_paths.shape[0]):
        plt.plot(time_grid, S_paths[i])
    plt.xlabel("Time (years)")
    plt.ylabel("SPY price")
    plt.title("Simulated Geometric Brownian Motion Paths for SPY")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_gbm_paths.png", dpi=150)
    plt.close()

    terminal_prices = S_paths[:, -1]

    plt.figure(figsize=(8, 5))
    plt.hist(terminal_prices, bins=15)
    plt.xlabel("Terminal SPY price (S_T)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Simulated Terminal Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_terminal_price_histogram.png", dpi=150)
    plt.close()


# ============================================================
# Step 3: Black-Scholes analytical pricing
# ============================================================

def black_scholes_d1(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive.")
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))


def black_scholes_d2(S, K, T, r, sigma):
    return black_scholes_d1(S, K, T, r, sigma) - sigma * sqrt(T)


def black_scholes_call(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def plot_black_scholes_sensitivities():
    S_grid = np.linspace(300, 700, 250)
    call_prices = np.array([black_scholes_call(S, K, T, r, sigma) for S in S_grid])
    put_prices = np.array([black_scholes_put(S, K, T, r, sigma) for S in S_grid])

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, call_prices, label="Call price")
    plt.plot(S_grid, put_prices, label="Put price")
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Option price")
    plt.title("Black-Scholes Call and Put Prices vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_bs_price_vs_underlying.png", dpi=150)
    plt.close()

    sigma_grid = np.linspace(0.05, 0.60, 200)
    call_by_sigma = np.array([black_scholes_call(S0, K, T, r, sig) for sig in sigma_grid])

    plt.figure(figsize=(8, 5))
    plt.plot(sigma_grid, call_by_sigma)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Call option price")
    plt.title("Black-Scholes Call Price vs Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_bs_call_price_vs_volatility.png", dpi=150)
    plt.close()

    T_grid = np.linspace(0.02, 2.0, 200)
    call_by_T = np.array([black_scholes_call(S0, K, tau, r, sigma) for tau in T_grid])

    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, call_by_T)
    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Call option price")
    plt.title("Black-Scholes Call Price vs Time to Maturity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_bs_call_price_vs_time.png", dpi=150)
    plt.close()


# ============================================================
# Step 4: Monte Carlo option pricing
# ============================================================

def monte_carlo_call_price(S0, K, T, r, sigma, n_simulations=N_SIMULATIONS, seed=SEED):
    np.random.seed(seed)
    Z = np.random.normal(size=n_simulations)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = discounted_payoffs.mean()
    standard_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_simulations)
    ci_lower = price - 1.96 * standard_error
    ci_upper = price + 1.96 * standard_error

    return price, standard_error, ci_lower, ci_upper, S_T, discounted_payoffs


def plot_monte_carlo_outputs():
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    mc_price, mc_se, ci_lower, ci_upper, S_T, discounted_payoffs = monte_carlo_call_price(
        S0, K, T, r, sigma
    )

    plt.figure(figsize=(8, 5))
    plt.hist(S_T, bins=60)
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Terminal SPY price (S_T)")
    plt.ylabel("Frequency")
    plt.title("Risk-Neutral Distribution of Simulated Terminal SPY Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_mc_terminal_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(discounted_payoffs, bins=60)
    plt.xlabel("Discounted payoff")
    plt.ylabel("Frequency")
    plt.title("Distribution of Discounted Call Option Payoffs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_mc_discounted_payoff_distribution.png", dpi=150)
    plt.close()

    np.random.seed(SEED)
    max_sim = 100_000
    Z = np.random.normal(size=max_sim)
    S_T_all = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs_all = np.exp(-r * T) * np.maximum(S_T_all - K, 0.0)
    running_estimate = np.cumsum(payoffs_all) / np.arange(1, max_sim + 1)

    checkpoints = np.arange(100, max_sim + 1, 100)
    running_at_checkpoints = running_estimate[checkpoints - 1]

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, running_at_checkpoints, label="Monte Carlo estimate")
    plt.axhline(bs_price, linestyle="--", linewidth=1, label="Black-Scholes price")
    plt.xlabel("Number of simulations")
    plt.ylabel("Call option price")
    plt.title("Monte Carlo Convergence to Black-Scholes Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_mc_convergence_to_bs.png", dpi=150)
    plt.close()

    return {
        "bs_price": bs_price,
        "mc_price": mc_price,
        "mc_standard_error": mc_se,
        "mc_ci_lower": ci_lower,
        "mc_ci_upper": ci_upper
    }


# ============================================================
# Step 5: Greeks
# ============================================================

def call_delta(S, K, T, r, sigma):
    return norm_cdf(black_scholes_d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    return norm_cdf(black_scholes_d1(S, K, T, r, sigma)) - 1.0


def gamma(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    return S * norm_pdf(d1) * sqrt(T)


def call_theta(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = black_scholes_d2(S, K, T, r, sigma)
    return -(S * norm_pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm_cdf(d2)


def put_theta(S, K, T, r, sigma):
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = black_scholes_d2(S, K, T, r, sigma)
    return -(S * norm_pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(-d2)


def call_rho(S, K, T, r, sigma):
    d2 = black_scholes_d2(S, K, T, r, sigma)
    return K * T * exp(-r * T) * norm_cdf(d2)


def put_rho(S, K, T, r, sigma):
    d2 = black_scholes_d2(S, K, T, r, sigma)
    return -K * T * exp(-r * T) * norm_cdf(-d2)


def plot_greeks():
    S_grid = np.linspace(300, 700, 250)

    call_delta_grid = np.array([call_delta(S, K, T, r, sigma) for S in S_grid])
    put_delta_grid = np.array([put_delta(S, K, T, r, sigma) for S in S_grid])
    gamma_grid = np.array([gamma(S, K, T, r, sigma) for S in S_grid])
    vega_grid = np.array([vega(S, K, T, r, sigma) for S in S_grid])
    call_theta_grid = np.array([call_theta(S, K, T, r, sigma) for S in S_grid])
    put_theta_grid = np.array([put_theta(S, K, T, r, sigma) for S in S_grid])
    call_rho_grid = np.array([call_rho(S, K, T, r, sigma) for S in S_grid])
    put_rho_grid = np.array([put_rho(S, K, T, r, sigma) for S in S_grid])

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, call_delta_grid, label="Call Delta")
    plt.plot(S_grid, put_delta_grid, label="Put Delta")
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Delta")
    plt.title("Black-Scholes Delta vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_delta_vs_underlying.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, gamma_grid)
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Gamma")
    plt.title("Black-Scholes Gamma vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_gamma_vs_underlying.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, vega_grid)
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Vega")
    plt.title("Black-Scholes Vega vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_vega_vs_underlying.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, call_theta_grid, label="Call Theta")
    plt.plot(S_grid, put_theta_grid, label="Put Theta")
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Theta per year")
    plt.title("Black-Scholes Theta vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_theta_vs_underlying.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, call_rho_grid, label="Call Rho")
    plt.plot(S_grid, put_rho_grid, label="Put Rho")
    plt.axvline(K, linestyle="--", linewidth=1, label="Strike price")
    plt.xlabel("Underlying SPY price (S)")
    plt.ylabel("Rho")
    plt.title("Black-Scholes Rho vs Underlying Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_rho_vs_underlying.png", dpi=150)
    plt.close()


# ============================================================
# Step 6 and Step 7: Historical volatility calibration
# ============================================================

def download_spy_data(start="2020-01-01", end=None):
    import yfinance as yf
    data = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError("No SPY data downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data[["Close"]].dropna()
    data.rename(columns={"Close": "SPY_Close"}, inplace=True)
    return data


def estimate_historical_volatility(data, trading_days=252, rolling_window=21):
    data = data.copy()
    data["Log_Return"] = np.log(data["SPY_Close"] / data["SPY_Close"].shift(1))
    data.dropna(inplace=True)

    daily_vol = data["Log_Return"].std(ddof=1)
    annualized_vol = daily_vol * np.sqrt(trading_days)

    data["Rolling_Annualized_Vol"] = (
        data["Log_Return"].rolling(rolling_window).std(ddof=1) * np.sqrt(trading_days)
    )

    return data, daily_vol, annualized_vol


def plot_historical_volatility_diagnostics(data):
    plt.figure(figsize=(9, 5))
    plt.plot(data.index, data["SPY_Close"])
    plt.xlabel("Date")
    plt.ylabel("Adjusted close price")
    plt.title("SPY Historical Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "16_spy_price_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(data.index, data["Log_Return"])
    plt.xlabel("Date")
    plt.ylabel("Daily log return")
    plt.title("SPY Daily Log Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_spy_log_returns.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(data["Log_Return"], bins=60)
    plt.xlabel("Daily log return")
    plt.ylabel("Frequency")
    plt.title("Distribution of SPY Daily Log Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "18_spy_return_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(data.index, data["Rolling_Annualized_Vol"])
    plt.xlabel("Date")
    plt.ylabel("Rolling annualized volatility")
    plt.title("SPY 21-Day Rolling Annualized Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "19_spy_rolling_volatility.png", dpi=150)
    plt.close()


# ============================================================
# Step 8: Implied volatility
# ============================================================

def implied_volatility_bisection(market_price, S, K, T, r, option_type="call",
                                low=1e-6, high=5.0, tol=1e-8, max_iter=200):
    """
    Solves for implied volatility using bisection.
    """
    if option_type == "call":
        price_func = black_scholes_call
    elif option_type == "put":
        price_func = black_scholes_put
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    low_price = price_func(S, K, T, r, low)
    high_price = price_func(S, K, T, r, high)

    if market_price < low_price or market_price > high_price:
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = price_func(S, K, T, r, mid)

        if abs(mid_price - market_price) < tol:
            return mid

        if mid_price < market_price:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def synthetic_volatility_smile_demo():
    """
    Creates a synthetic implied volatility smile example.
    This does not require market option-chain data.
    """
    strikes = np.linspace(350, 650, 31)

    # Synthetic market IV pattern: higher IV away from ATM.
    synthetic_market_iv = 0.18 + 0.000004 * (strikes - S0) ** 2

    market_prices = np.array([
        black_scholes_call(S0, strike, T, r, iv)
        for strike, iv in zip(strikes, synthetic_market_iv)
    ])

    recovered_iv = np.array([
        implied_volatility_bisection(price, S0, strike, T, r, option_type="call")
        for price, strike in zip(market_prices, strikes)
    ])

    plt.figure(figsize=(8, 5))
    plt.plot(strikes, recovered_iv, marker="o")
    plt.axvline(S0, linestyle="--", linewidth=1, label="Current SPY level")
    plt.xlabel("Strike price")
    plt.ylabel("Implied volatility")
    plt.title("Synthetic Implied Volatility Smile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "20_synthetic_implied_volatility_smile.png", dpi=150)
    plt.close()

    iv_table = pd.DataFrame({
        "Strike": strikes,
        "Synthetic_Market_IV": synthetic_market_iv,
        "Market_Call_Price": market_prices,
        "Recovered_Implied_IV": recovered_iv
    })
    iv_table.to_csv(OUTPUT_DIR / "synthetic_implied_volatility_table.csv", index=False)


# ============================================================
# Step 9: Model risk diagnostics summary
# ============================================================

def create_summary_report(results):
    report_path = OUTPUT_DIR / "project_summary.txt"

    lines = []
    lines.append("SPY Stochastic Option Pricing Project Summary")
    lines.append("=" * 55)
    lines.append("")
    lines.append("Base inputs:")
    lines.append(f"S0     : {S0}")
    lines.append(f"K      : {K}")
    lines.append(f"T      : {T}")
    lines.append(f"r      : {r}")
    lines.append(f"sigma  : {sigma}")
    lines.append("")

    bs_call = black_scholes_call(S0, K, T, r, sigma)
    bs_put = black_scholes_put(S0, K, T, r, sigma)

    lines.append("Black-Scholes analytical prices:")
    lines.append(f"Call price : {bs_call:.6f}")
    lines.append(f"Put price  : {bs_put:.6f}")
    lines.append("")

    lines.append("Monte Carlo validation:")
    lines.append(f"MC call price        : {results['mc_price']:.6f}")
    lines.append(f"MC standard error    : {results['mc_standard_error']:.6f}")
    lines.append(f"MC 95% CI            : [{results['mc_ci_lower']:.6f}, {results['mc_ci_upper']:.6f}]")
    lines.append("")

    lines.append("Greeks:")
    lines.append(f"Call Delta           : {call_delta(S0, K, T, r, sigma):.6f}")
    lines.append(f"Put Delta            : {put_delta(S0, K, T, r, sigma):.6f}")
    lines.append(f"Gamma                : {gamma(S0, K, T, r, sigma):.8f}")
    lines.append(f"Vega                 : {vega(S0, K, T, r, sigma):.6f}")
    lines.append(f"Call Theta per day   : {call_theta(S0, K, T, r, sigma) / 252:.6f}")
    lines.append(f"Call Rho per 1% rate : {call_rho(S0, K, T, r, sigma) / 100:.6f}")
    lines.append("")

    if "historical_vol" in results:
        lines.append("Historical volatility calibration:")
        lines.append(f"Daily volatility      : {results['daily_vol']:.6f}")
        lines.append(f"Annualized volatility : {results['historical_vol']:.6f}")
        lines.append(f"Calibrated BS call    : {results['calibrated_call']:.6f}")
        lines.append(f"Calibrated BS put     : {results['calibrated_put']:.6f}")
        lines.append("")

    lines.append("Model risk observations:")
    lines.append("- Black-Scholes assumes constant volatility, but rolling volatility changes over time.")
    lines.append("- Black-Scholes assumes lognormal prices and normally distributed log returns; real returns may show fat tails.")
    lines.append("- Black-Scholes assumes continuous trading and no transaction costs; real hedging is discrete and costly.")
    lines.append("- Implied volatility smile/skew indicates that the market does not use one constant volatility for all strikes.")
    lines.append("")

    lines.append("CV bullet:")
    lines.append("Built a stochastic option pricing framework for SPY options using Geometric Brownian Motion, Itô's Lemma,")
    lines.append("and the Black-Scholes PDE; implemented analytical pricing, Monte Carlo validation, Greeks estimation,")
    lines.append("historical volatility calibration, implied volatility extraction, and model risk diagnostics.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main runner
# ============================================================

def main():
    print("Running SPY Stochastic Option Pricing Project...")

    plot_call_payoff()
    plot_brownian_and_gbm()
    plot_black_scholes_sensitivities()

    mc_results = plot_monte_carlo_outputs()
    plot_greeks()
    synthetic_volatility_smile_demo()

    results = dict(mc_results)

    # Historical data section requires internet access.
    try:
        spy_data = download_spy_data(start="2020-01-01")
        spy_data, daily_vol, annualized_vol = estimate_historical_volatility(spy_data)
        plot_historical_volatility_diagnostics(spy_data)
        spy_data.to_csv(OUTPUT_DIR / "spy_processed_returns.csv")

        calibrated_call = black_scholes_call(S0, K, T, r, annualized_vol)
        calibrated_put = black_scholes_put(S0, K, T, r, annualized_vol)

        results.update({
            "daily_vol": daily_vol,
            "historical_vol": annualized_vol,
            "calibrated_call": calibrated_call,
            "calibrated_put": calibrated_put
        })

        print(f"Historical annualized volatility: {annualized_vol:.4f}")
        print(f"Calibrated call price          : {calibrated_call:.4f}")
        print(f"Calibrated put price           : {calibrated_put:.4f}")

    except Exception as e:
        print("Historical data section skipped.")
        print("Reason:", str(e))
        print("Run locally with internet access and yfinance installed to enable it.")

    create_summary_report(results)

    print("Project completed. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
