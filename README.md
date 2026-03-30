# Qube Quant Trading

Info
- Beta neut
- Long/Short
- Daily-Weekly rebalancing
- Risk = $\sqrt{w_t^T \Sigma w_t}$ (current notational portfolio, std dev based on daily last years stock returns)
- SR = sqrt(252) x E(PnL) / sigma(PnL)


Strategies
- Earnings suprises
- News sentiment
- High/low past return futures
- Boring stocks
- Firms about to be or recently added/removed from index
- Firms about to be or recently sold/acquired
- Most patent filing companies
- See if $E(R) = \gama_0 + \gama_0 \beta + \gama_1 \beta^2$ -> Arb


Process
- Normalise data to z-scores
- Backtesting
- Determine statistical significance of the signal out of sample
- Determine economic significance of the signal out of sample
- Avoid biases: Look ahead, index constituents, closing prices, unwanted exposures, overfiting, multiple testing


Papers
- The supraview of return predictive signals, Review of accounting studies(330 signals)
- Assaying anomalies (backtesting enginge)
- Boyd et al. 2024

Notes
- Signals: significance, cross-sectional/panel regressions
- Use normalised, residualised returns: single factor/capm, multi factor
- combining signals
- top down signals/factor investing
- Diversify by: signal, stratey, beta/beta netural, industry
- Test portfolio optimisation for the most predictable stocks
- Empirical evidence (Chow, Hsu, Kalesnik, little, 2011)
  - Huristic based, optimisation based
- Cov matrix: EWAM, DCC, RS
- Look for regime changes in covariances/variances over time, how long they presist

Constraints
- Max ADV = 2%