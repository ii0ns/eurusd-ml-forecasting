# EUR/USD Next-Day Forecasting with Random Forest

A supervised learning pipeline that evaluates whether the next-day return of EUR/USD can be predicted from price history and Gold/Oil returns. It uses Yahoo Finance data from 2015 to April 2025, and all results are compared against a naive persistence baseline (the previous close used as the forecast).

> Full writeup: [ii0ns.github.io](https://ii0ns.github.io)

---

## Results

Test period: 2023-04-17 to 2025-04-28 (528 trading days, chronological 80/20 split).

| Metric | Random Forest | Persistence baseline |
|--------|---------------|----------------------|
| R² (price) | 0.9529 | 0.9526 |
| RMSE | 0.004781 | 0.004799 |
| MAE | 0.003523 | 0.003529 |
| MAPE | 0.3258% | 0.3263% |

| Metric | Value | Reference |
|--------|-------|-----------|
| Directional accuracy | 51.14% | majority-class rate: 53.03% |
| R² on returns | 0.006 | 0 = no explanatory power |

**Interpretation.** Price levels are highly autocorrelated, so any forecast close to the previous close achieves a high R². The model does not improve meaningfully on the persistence baseline, and its directional accuracy is not better than chance. On this feature set, next-day EUR/USD returns are not predictable.

---

## Pipeline

```
yfinance → EnhancedDataEngineer → OptimizedModelDeveloper → ModelEvaluator
```

| Step | Class | Description |
|------|-------|-------------|
| Data | `EnhancedDataEngineer` | Downloads EUR/USD, Gold and Oil from Yahoo Finance and builds 27 features |
| Model | `OptimizedModelDeveloper` | Trains a Random Forest on the next-day return with a chronological split, and exports results |
| Evaluation | `ModelEvaluator` | Generates prediction plots, error distribution and feature importances |

**Target:** next-day return, `close(t+1) / close(t) - 1`. Predicted returns are converted back to prices to compute the price metrics.

**Data range:** 2015-01-01 to 2025-04-30 (fixed for reproducibility).
Train: 2015-03-11 to 2023-04-14 (2,110 days). Test: 2023-04-17 to 2025-04-28 (528 days).

---

## Features (27 total)

All features use only information available at day *t*.

- Returns: 1, 3 and 7 days
- Volatility: rolling standard deviation over 7 and 14 days
- Price relative to moving averages (7, 21, 50 days) and crossover ratios
- RSI(14) and an overbought/oversold signal
- Lagged returns at 1, 2, 3, 5, 7 and 14 days
- 7-day range and position within the range
- Gold and Oil returns (1 and 7 days), lagged by one day
- Calendar: day of week, month, quarter

---

## Model

```python
RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_split=10,
    min_samples_leaf=4, max_features='sqrt', bootstrap=True,
    n_jobs=-1, random_state=42
)
```

---

## Methodology notes

- **Target leakage.** An earlier version predicted the same-day closing price using features computed from that same close, which produced R² 0.95 and roughly 75% directional accuracy. Changing the target to the next-day return and restricting features to information available at day *t* removed the leakage, and the metrics fell to the values reported above.
- **Timing misalignment in exogenous data.** Using same-day Gold and Oil values gave a directional accuracy of 60.04%. Lagging them by one day reduced it to about 50%, which indicates the earlier result came from a timing misalignment between the series and not from predictive signal. The pipeline now uses the previous day's values. This was observed during development and is not reproducible with the scripts in this repository.
- **Baseline comparison.** All price metrics are reported against a persistence baseline, and directional accuracy against the majority-class rate.

AI tools were used to help review and debug the pipeline.

---

## Limitations

- Next-day EUR/USD returns are not predictable from price history and Gold/Oil returns alone.
- The model performs on par with the persistence baseline and should not be used as a trading signal.
- No macroeconomic features are included (interest rate differentials, inflation releases, central bank decisions).
- Yahoo Finance data may be revised, so a re-run can differ slightly from the figures above.

---

## Project Structure

```
eurusd-ml-forecasting/
├── src/
│   ├── data_engineer.py      # EnhancedDataEngineer
│   ├── model_developer.py    # OptimizedModelDeveloper
│   └── evaluator.py          # ModelEvaluator
├── validation/
│   ├── direction_check.py    # directional accuracy tests
│   └── README.md
├── export/                   # predictions, metrics, feature importance (JSON)
├── main.py                   # entry point
├── evaluation_report.png     # 4-panel evaluation figure
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/ii0ns/eurusd-ml-forecasting
cd eurusd-ml-forecasting
pip install -r requirements.txt

python main.py                            # run the pipeline
python validation/direction_check.py      # requires processed_data.csv from main.py
```

Outputs of `main.py`:
- `evaluation_report.png`: 4-panel evaluation figure
- `export/`: `predictions.json`, `metrics.json`, `feature_importance.json`
- `trained_model.pkl` and `processed_data.csv`: generated locally, not tracked in git

---

## License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is licensed under the MIT License.
