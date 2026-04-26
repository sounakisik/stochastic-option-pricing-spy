# Stochastic Option Pricing Project: Black-Scholes, Brownian Motion, Monte Carlo, and Model Risk

## Project Objective

This project builds an end-to-end stochastic option pricing framework for European options written on **SPY**, the SPDR S&P 500 ETF Trust.

The objective is to connect stochastic calculus with practical derivative pricing by moving from asset price modeling to analytical pricing, Monte Carlo validation, Greeks-based risk sensitivity, volatility calibration, implied volatility, and model risk diagnostics.

## What the Project Achieves

The project demonstrates the full workflow of a quantitative option pricing model:

1. Defines a European call/put option pricing problem on SPY.
2. Models the underlying asset using Brownian Motion and Geometric Brownian Motion.
3. Implements the Black-Scholes analytical pricing formula.
4. Validates analytical prices using Monte Carlo simulation.
5. Computes Greeks: Delta, Gamma, Vega, Theta, and Rho.
6. Estimates historical volatility from SPY log returns.
7. Uses calibrated volatility inside the Black-Scholes formula.
8. Extracts implied volatility through numerical root-finding.
9. Produces model risk diagnostics around volatility, return distribution, and model assumptions.

## Mathematical Foundation

The Black-Scholes model assumes that the underlying asset follows Geometric Brownian Motion:

```math
dS_t = \mu S_t dt + \sigma S_t dW_t
```

Under the risk-neutral measure:

```math
dS_t = rS_tdt + \sigma S_tdW_t^{\mathbb{Q}}
```

The European call price is:

```math
C_0 = S_0N(d_1) - Ke^{-rT}N(d_2)
```

where:

```math
d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
```

```math
d_2 = d_1 - \sigma\sqrt{T}
```

Monte Carlo pricing estimates:

```math
C_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\max(S_T-K,0)]
```

## Project Structure

```text
spy_stochastic_option_pricing_project/
│
├── spy_option_pricing_project.py
├── requirements.txt
├── README.md
└── outputs/
    ├── generated plots
    ├── processed SPY data
    ├── implied volatility table
    └── project_summary.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python spy_option_pricing_project.py
```

The script will generate all plots and outputs inside the `outputs/` folder.

## Important Note on Data

The historical volatility calibration section uses `yfinance` to download SPY data. This requires internet access.

If internet access is unavailable, the script will still run the simulation, Black-Scholes pricing, Monte Carlo validation, Greeks, and synthetic implied volatility sections.

## Key Outputs

The project generates plots for:

- European call payoff
- Brownian Motion paths
- Geometric Brownian Motion paths
- Black-Scholes price sensitivity
- Monte Carlo terminal price distribution
- Monte Carlo convergence
- Delta, Gamma, Vega, Theta, Rho
- SPY historical prices
- SPY log returns
- rolling annualized volatility
- synthetic implied volatility smile

## Model Risk Discussion

The project highlights important limitations of Black-Scholes:

- volatility is not constant in real markets
- returns may show fat tails
- prices may jump
- trading is not continuous
- hedging has transaction costs
- implied volatility varies by strike and maturity

This makes the project relevant not only for pricing but also for model validation and model risk management.

## CV Bullet

Built a stochastic option pricing framework for SPY options using Geometric Brownian Motion, Itô's Lemma, and the Black-Scholes PDE; implemented analytical pricing, Monte Carlo validation, Greeks estimation, historical volatility calibration, implied volatility extraction, and model risk diagnostics.

## Interview Explanation

I developed an end-to-end option pricing framework for SPY European options. I started by modeling the underlying asset using Geometric Brownian Motion, derived the Black-Scholes pricing setup, implemented analytical pricing and Monte Carlo validation, computed Greeks for risk sensitivity, and then moved toward model validation by estimating historical volatility, extracting implied volatility, and analyzing where Black-Scholes assumptions break down.
