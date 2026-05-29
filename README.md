# Titan Markets Sector Rotation Model v1.0

This workspace contains a production-ready generator for two deliverables:

- `Titan_Markets_Sector_Rotation_Model_v1.xlsx`: black/white/silver Excel dashboard with sector scores, event study data, Fed events, and tactical allocation sheets.
- `Titan_Markets_Fed_Pivot_Playbook.pdf`: 8-page client-facing Fed Pivot Playbook with methodology, backtest tables, recommended allocation, risk guardrails, deployment calendar, and disclosures.

## Run

```bash
python3 titan_sector_rotation_model_v1.py
```

Outputs are written to:

```text
deliverables/
```

If network access is unavailable, run the demonstration mode:

```bash
python3 titan_sector_rotation_model_v1.py --offline
```

## Data Sources

- Fed policy event detection: FRED effective federal funds rate (`DFF`)
- Equity sector prices: Yahoo Finance public chart endpoint for SPY and SPDR sector ETFs
- Sector universe: `XLC`, `XLY`, `XLP`, `XLE`, `XLF`, `XLV`, `XLI`, `XLB`, `XLRE`, `XLK`, `XLU`

## Model Logic

The model detects large daily changes in the effective federal funds rate and treats them as Fed policy-change events. For each event, it measures sector ETF alpha versus SPY during:

- 45 trading days before the policy change
- 45 trading days after the policy change

Each sector is scored by:

- Average alpha
- Median alpha
- Hit rate
- Alpha volatility
- Consistency ratio
- Average max drawdown

The recommended tactical sleeve uses the top six post-cut sectors by composite score, with a 25% single-sector cap.

## Client Use Notes

This is research infrastructure for Titan Entry and institutional consulting workflows. Before using results with clients, refresh the model, review the raw event list, confirm live Fed probabilities, and apply mandate-specific restrictions.

Backtests do not include transaction costs, taxes, slippage, advisory fees, or total-return reinvestment.
