#!/usr/bin/env python3
# Sharpe-optimal portfolio backtest vs user portfolio and ATX
# Features: Bayesian shrinkage option for covariance; long-only Sharpe maximization

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta
from itertools import combinations

# ===========================
# Configuration
# ===========================
tickers = [
    "RBI.VI", "ATS.VI", "OMV.VI", "VER.VI", "TKA.VI",
    "FQT.VI", "LNZ.VI", "WIE.VI", "PAL.VI", "STR.VI"
]

weights_user = np.array([
    0.35, 0.197122, 0.128821, 0.00, 0.095171,
    0.202878, 0.00, 0.00, 0.00, 0.026009
])

index_ticker = "^ATX"
risk_free_rate = 0.03  # annual
start_date = "2025-10-01"  # backtest period
end_date = None
show_plot = True
out_file = "portfolio_comparison.png"
TRADING_DAYS = 252
holding_days = 63  # 3-month optimization horizon

OPTIMAL_SHRINKAGE = True        # Tune shrinkage via cross-validation

# ===========================
# Helper Functions
# ===========================
def _extract_price_panel(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = {c[0] for c in df.columns}
        if "Adj Close" in lvl0:
            panel = df["Adj Close"]
        elif "Close" in lvl0:
            panel = df["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found.")
    else:
        if "Adj Close" in df.columns:
            panel = df["Adj Close"]
        elif "Close" in df.columns:
            panel = df["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found.")
    return panel if isinstance(panel, pd.DataFrame) else panel.to_frame()

def download_prices(tickers_list, start=None, end=None) -> pd.DataFrame:
    df1 = yf.download(tickers_list, start=start, end=end, progress=False, auto_adjust=False, threads=True)
    try:
        return _extract_price_panel(df1)
    except Exception:
        df2 = yf.download(tickers_list, start=start, end=end, progress=False, auto_adjust=True, threads=True)
        return _extract_price_panel(df2)

def jensen_alpha_beta_daily(port_ret: pd.Series, mkt_ret: pd.Series, rf_daily: float):
    df = pd.DataFrame({"rp": port_ret, "rm": mkt_ret}).dropna()
    if df.empty:
        return np.nan, np.nan
    ex_rp = df["rp"] - rf_daily
    ex_rm = df["rm"] - rf_daily
    var_rm = np.var(ex_rm, ddof=1)
    if var_rm <= 0:
        return np.nan, np.nan
    cov = np.cov(ex_rm, ex_rp, ddof=1)[0, 1]
    beta = cov / var_rm
    alpha_daily = ex_rp.mean() - beta * ex_rm.mean()
    return alpha_daily, beta

def max_drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = dd.min()
    end_idx = dd.idxmin()
    start_idx = equity.loc[:end_idx].idxmax()
    return mdd, start_idx, end_idx

def downside_deviation(returns: pd.Series, mar: float = 0.0):
    downside = returns[returns < mar]
    if downside.size == 0:
        return 0.0
    return np.sqrt((downside**2).mean())

def normalize_equities_same_start(*equities):
    """Align and normalize multiple equity curves to start at 1."""
    common_start = max([eq.index[0] for eq in equities])
    aligned = [eq.loc[common_start:] for eq in equities]
    common_idx = aligned[0].index
    for eq in aligned[1:]:
        common_idx = common_idx.intersection(eq.index)
    normalized = [(eq.loc[common_idx] / eq.loc[common_idx].iloc[0]) for eq in aligned]
    return normalized

# ===========================
# Optimization Functions
# ===========================
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return vol, ret

def compute_cvar(weights, returns_matrix, alpha=0.95):
    """Compute CVaR (Conditional Value at Risk) for portfolio"""
    portfolio_returns = returns_matrix @ weights
    var_threshold = np.percentile(portfolio_returns, (1 - alpha) * 100)
    cvar = -portfolio_returns[portfolio_returns <= var_threshold].mean()
    return cvar

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
    """Negative Sharpe (annualized, long-only) for minimization."""
    vol, ret = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - risk_free) / vol if vol > 0 else -np.inf
    return -sharpe

def optimize_shrinkage_intensity(asset_returns, index_returns, mean_returns, 
                                  betas, holding_days, n_cv_folds=3):
    """
    Cross-validation to find optimal shrinkage parameter
    Tests lambdas from 0 (pure factor) to 1 (pure sample)
    """
    print("\n  Optimizing shrinkage intensity via cross-validation...")
    
    # Prepare factor and sample covariances
    factor_cov = np.atleast_2d(np.cov(index_returns))
    specific_var = asset_returns.var() - betas ** 2 * index_returns.var()
    specific_var[specific_var < 0] = 0
    B = betas.reshape(-1, 1)
    F = factor_cov
    D = np.diag(specific_var)
    cov_factor = B @ F @ B.T + D
    cov_sample = asset_returns.cov().values * holding_days
    
    # Test different shrinkage intensities
    lambdas = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100%
    sharpe_scores = []
    
    n_samples = len(asset_returns)
    fold_size = n_samples // n_cv_folds
    
    for lam in lambdas:
        fold_sharpes = []
        cov_blend = lam * cov_sample + (1 - lam) * cov_factor
        
        for fold in range(n_cv_folds):
            # Simple train/test split
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_cv_folds - 1 else n_samples
            
            # Compute weights on training data
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1.0) for _ in range(len(mean_returns)))
            
            result = sco.minimize(
                lambda w: -(np.dot(w, mean_returns) - 0.03/4) / np.sqrt(np.dot(w.T, np.dot(cov_blend, w))),
                len(mean_returns) * [1./len(mean_returns)],
                method='SLSQP', bounds=bounds, constraints=constraints
            )
            
            # Evaluate on test fold (simulate performance)
            test_returns = asset_returns.iloc[test_start:test_end]
            test_port_ret = (test_returns.values @ result.x).mean() * holding_days
            test_port_vol = (test_returns.values @ result.x).std() * np.sqrt(holding_days)
            fold_sharpe = test_port_ret / test_port_vol if test_port_vol > 0 else 0
            fold_sharpes.append(fold_sharpe)
        
        sharpe_scores.append(np.mean(fold_sharpes))
    
    optimal_lambda = lambdas[np.argmax(sharpe_scores)]
    print(f"  Optimal shrinkage intensity: {optimal_lambda:.2f} (lambda)")
    print(f"  CV Sharpe scores: {dict(zip([f'{l:.1f}' for l in lambdas], [f'{s:.3f}' for s in sharpe_scores]))}")
    
    return optimal_lambda

# ===========================
# Beta & Variance Analysis
# ===========================
def analyze_beta_drivers(betas, tickers, weights_user, weights_optimal):
    """Analyze which assets drive portfolio beta."""
    print("\n" + "="*70)
    print(" BETA DRIVER ANALYSIS")
    print("="*70)
    
    beta_df = pd.DataFrame({
        'Ticker': tickers,
        'Beta': betas,
        'User Weight': weights_user,
        'Optimal Weight': weights_optimal,
        'User Beta Contribution': betas * weights_user,
        'Optimal Beta Contribution': betas * weights_optimal
    })
    beta_df = beta_df.sort_values('Beta', ascending=False)
    
    print("\nIndividual Asset Betas (sorted by beta):")
    print(beta_df.to_string(index=False))
    
    user_portfolio_beta = np.sum(betas * weights_user)
    optimal_portfolio_beta = np.sum(betas * weights_optimal)
    
    print(f"\nUser Portfolio Beta: {user_portfolio_beta:.4f}")
    print(f"Optimal Portfolio Beta: {optimal_portfolio_beta:.4f}")
    
    print("\nTop 3 Beta Contributors (User Portfolio):")
    top_user = beta_df.nlargest(3, 'User Beta Contribution')[['Ticker', 'Beta', 'User Weight', 'User Beta Contribution']]
    print(top_user.to_string(index=False))
    
    print("\nTop 3 Beta Contributors (Optimal Portfolio):")
    top_opt = beta_df.nlargest(3, 'Optimal Beta Contribution')[['Ticker', 'Beta', 'Optimal Weight', 'Optimal Beta Contribution']]
    print(top_opt.to_string(index=False))

def analyze_top_variance_pairs(cov_matrix, tickers):
    """Find top 3 asset pairs with highest covariance."""
    print("\n" + "="*70)
    print(" TOP 3 VARIANCE/COVARIANCE PAIRS")
    print("="*70)
    
    pairs = []
    for i, j in combinations(range(len(tickers)), 2):
        pairs.append({
            'Asset 1': tickers[i],
            'Asset 2': tickers[j],
            'Covariance': cov_matrix[i, j],
            'Correlation': cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j]),
            'Variance 1': cov_matrix[i, i],
            'Variance 2': cov_matrix[j, j]
        })
    
    pairs_df = pd.DataFrame(pairs)
    
    print("\nTop 3 Pairs by Covariance (absolute value):")
    top_cov = pairs_df.loc[pairs_df['Covariance'].abs().nlargest(3).index]
    for idx, row in top_cov.iterrows():
        print(f"\n{row['Asset 1']} ↔ {row['Asset 2']}")
        print(f"  Covariance: {row['Covariance']:.6f}")
        print(f"  Correlation: {row['Correlation']:.4f}")
        print(f"  Variance {row['Asset 1']}: {row['Variance 1']:.6f}")
        print(f"  Variance {row['Asset 2']}: {row['Variance 2']:.6f}")
    
    print("\n\nTop 3 Assets by Individual Variance:")
    variances = [(tickers[i], cov_matrix[i, i]) for i in range(len(tickers))]
    variances.sort(key=lambda x: x[1], reverse=True)
    for ticker, var in variances[:3]:
        print(f"  {ticker}: {var:.6f}")

# ===========================
# Main
# ===========================
def main():
    print("\n" + "="*70)
    print(" SHARPE-OPTIMAL PORTFOLIO BACKTEST (NO CONSTRAINT PENALTIES)")
    print("="*70)
    if OPTIMAL_SHRINKAGE:
        print(f"\nCovariance Shrinkage: Optimized via CV")
    else:
        print(f"\nCovariance Shrinkage: Fixed 50/50 blend")

    # Download optimization data
    print("\n[1/7] Downloading 1-year data for portfolio optimization...")
    opt_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    data_opt = download_prices(tickers + [index_ticker], start=opt_start, end=None)
    returns_opt = data_opt.pct_change().dropna()
    asset_returns_opt = returns_opt[tickers]
    index_returns_opt = returns_opt[index_ticker]

    # Estimate betas
    print("[2/7] Computing factor model (beta to ATX) and covariance estimates...")
    betas = []
    for ticker in tickers:
        cov = np.cov(asset_returns_opt[ticker], index_returns_opt)[0, 1]
        var = index_returns_opt.var()
        betas.append(cov / var)
    betas = np.array(betas)

    # Factor covariance
    factor_cov = np.atleast_2d(np.cov(index_returns_opt))
    specific_var = asset_returns_opt.var() - betas ** 2 * index_returns_opt.var()
    specific_var[specific_var < 0] = 0
    B = betas.reshape(-1, 1)
    F = factor_cov
    D = np.diag(specific_var)
    cov_factor = B @ F @ B.T + D

    # Sample covariance
    cov_sample = asset_returns_opt.cov().values * holding_days

    # Ledoit-Wolf
    lw = LedoitWolf().fit(asset_returns_opt)
    cov_lw = lw.covariance_ * holding_days

    # Optimal shrinkage
    if OPTIMAL_SHRINKAGE:
        shrinkage_param = optimize_shrinkage_intensity(
            asset_returns_opt, index_returns_opt, 
            asset_returns_opt.mean() * holding_days, 
            betas, holding_days
        )
    else:
        shrinkage_param = 0.5
    
    cov_bayes = shrinkage_param * cov_sample + (1 - shrinkage_param) * cov_factor
    print(f"\n  Using shrinkage parameter: {shrinkage_param:.2f}")

    # Mean returns
    mean_returns = asset_returns_opt.mean() * holding_days

    # Optimize Sharpe (long-only, sum to 1)
    print("\n[3/7] Optimizing Sharpe ratio (long-only, sum to 1)...")
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1.0) for _ in range(len(tickers)))
    result_sharpe = sco.minimize(
        neg_sharpe, len(tickers)*[1./len(tickers)],
        args=(mean_returns, cov_bayes, risk_free_rate),
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    
    weights_optimal = result_sharpe.x

    print("\n[4/7] Optimal Sharpe Portfolio Weights:")
    for t, w in zip(tickers, weights_optimal):
        print(f"  {t}: {w:.4f}")

    # Beta and Variance Analysis
    print("\n[5/7] Analyzing beta drivers and variance structure...")
    analyze_beta_drivers(betas, tickers, weights_user, weights_optimal)
    analyze_top_variance_pairs(cov_bayes, tickers)

    # Download backtest data
    print(f"\n[6/7] Downloading backtest data from {start_date}...")
    prices_bt = download_prices(tickers, start=start_date, end=end_date)
    bench_bt = download_prices([index_ticker], start=start_date, end=end_date).iloc[:, 0]

    prices_bt = prices_bt.dropna(how="any")
    bench_bt = bench_bt.dropna()
    if prices_bt.empty or bench_bt.empty:
        sys.exit("Insufficient backtest data.")

    # Returns
    rets_bt = prices_bt.pct_change().dropna()
    mkt_bt = bench_bt.pct_change().dropna()
    idx = rets_bt.index.intersection(mkt_bt.index)
    rets_bt = rets_bt.loc[idx]
    mkt_bt = mkt_bt.loc[idx]
    if rets_bt.empty or mkt_bt.empty:
        sys.exit("No overlapping days.")

    missing = [t for t in tickers if t not in rets_bt.columns]
    if missing:
        sys.exit(f"Missing tickers: {missing}")
    rets_bt = rets_bt[tickers]

    # Portfolio returns
    port_user_daily = (rets_bt * weights_user).sum(axis=1)
    port_opt_daily = (rets_bt * weights_optimal).sum(axis=1)

    # Equity curves
    eq_user_raw = (1 + port_user_daily).cumprod()
    eq_opt_raw = (1 + port_opt_daily).cumprod()
    eq_mkt_raw = (1 + mkt_bt).cumprod()

    eq_user, eq_opt, eq_mkt = normalize_equities_same_start(eq_user_raw, eq_opt_raw, eq_mkt_raw)
    common_idx = eq_user.index

    port_user_daily = port_user_daily.loc[common_idx]
    port_opt_daily = port_opt_daily.loc[common_idx]
    mkt_bt = mkt_bt.loc[common_idx]

    n_days = len(port_user_daily)
    rf_daily = risk_free_rate / TRADING_DAYS

    # Metrics
    def compute_metrics(port_daily, eq, label):
        print(f"\n{'='*70}")
        print(f" {label}")
        print(f"{'='*70}")
        period_return = eq.iloc[-1] - 1.0
        avg_daily = port_daily.mean()
        std_daily = port_daily.std(ddof=1)
        sharpe_daily = ((port_daily - rf_daily).mean() / std_daily) if std_daily > 0 else np.nan
        active_daily = port_daily - mkt_bt
        te_daily = active_daily.std(ddof=1)
        ir_daily = (active_daily.mean() / te_daily) if te_daily > 0 else np.nan
        corr = np.corrcoef(port_daily, mkt_bt)[0, 1] if std_daily > 0 and mkt_bt.std(ddof=1) > 0 else np.nan
        ddv = downside_deviation(port_daily, mar=0.0)
        sortino_daily = (port_daily.mean() / ddv) if ddv > 0 else np.nan
        alpha_daily, beta = jensen_alpha_beta_daily(port_daily, mkt_bt, rf_daily)
        alpha_period = alpha_daily * n_days if pd.notna(alpha_daily) else np.nan
        mdd, mdd_start, mdd_end = max_drawdown(eq)

        print(f"Period Return: {period_return:.2%}")
        print(f"Avg Daily Return: {avg_daily:.6f}")
        print(f"Volatility (daily): {std_daily:.6f}")
        print(f"Sharpe (daily): {sharpe_daily:.4f}" if pd.notna(sharpe_daily) else "Sharpe: n/a")
        print(f"Tracking Error (daily): {te_daily:.6f}")
        print(f"Information Ratio (daily): {ir_daily:.4f}" if pd.notna(ir_daily) else "IR: n/a")
        print(f"Correlation (vs Benchmark): {corr:.4f}" if pd.notna(corr) else "Corr: n/a")
        print(f"Sortino (daily): {sortino_daily:.4f}" if pd.notna(sortino_daily) else "Sortino: n/a")
        print(f"Beta: {beta:.3f}" if pd.notna(beta) else "Beta: n/a")
        print(f"Alpha (daily): {alpha_daily:.6f}" if pd.notna(alpha_daily) else "Alpha (daily): n/a")
        print(f"Alpha (period): {alpha_period:.4%}" if pd.notna(alpha_period) else "Alpha (period): n/a")
        print(f"Max Drawdown: {mdd:.2%} ({mdd_start.date()} to {mdd_end.date()})")

    print(f"\n[7/7] Computing metrics for backtest period: {common_idx[0].date()} → {common_idx[-1].date()} ({n_days} days)")
    compute_metrics(port_user_daily, eq_user, "USER PORTFOLIO")
    compute_metrics(port_opt_daily, eq_opt, "OPTIMAL PORTFOLIO (Sharpe)")

    # Benchmark metrics
    print(f"\n{'='*70}")
    print(f" BENCHMARK ({index_ticker})")
    print(f"{'='*70}")
    period_return_mkt = eq_mkt.iloc[-1] - 1.0
    print(f"Period Return: {period_return_mkt:.2%}")
    print(f"Avg Daily Return: {mkt_bt.mean():.6f}")
    print(f"Volatility (daily): {mkt_bt.std(ddof=1):.6f}")
    mdd_mkt, mdd_start_mkt, mdd_end_mkt = max_drawdown(eq_mkt)
    print(f"Max Drawdown: {mdd_mkt:.2%} ({mdd_start_mkt.date()} to {mdd_end_mkt.date()})")

    # Plot
    plt.figure(figsize=(12, 7))
    eq_user.plot(label="User Portfolio", linewidth=2.5, color='blue')
    eq_opt.plot(label="Optimal Portfolio (Sharpe)", linewidth=2.5, color='green')
    eq_mkt.plot(label=f"Benchmark ({index_ticker})", linewidth=2, linestyle="--", color='orange')
    plt.title("Equity Curves: Sharpe-Optimal vs User vs ATX", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Indexed Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"\n📈 Chart saved: {out_file}\n")
    if show_plot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    main()
