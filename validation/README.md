# Validation

This folder checks one question: **can the model predict the direction (up or down) of EUR/USD the next day?**

## How to run

From the project root:

```bash
pip install scipy
python main.py
python validation/direction_check.py
```

`main.py` must run first because it creates `processed_data.csv`, which the script reads.

## What `direction_check.py` does

It trains models on the first 80% of the data, tests them on the last 20%, and compares their directional accuracy with the majority-class rate ("always up" or "always down", whichever is better on that period). A one-sided binomial test gives a p-value for whether the model beats the majority-class rate by more than chance.

It runs the same question several ways:

1. Random Forest with all features
2. Random Forest without Gold/Oil features
3. Random Forest with Gold/Oil features only
4. Ridge regression on lagged returns only
5. Walk-forward validation (expanding window, 6 folds) with the full Random Forest

## Results

Single holdout (test period 2023-04 to 2025-04):

| Model | Directional accuracy | Majority-class rate |
|---|---|---|
| Random Forest, all features | 51.14% | 53.03% |
| Random Forest, without Gold/Oil | 51.33% | 53.03% |
| Random Forest, Gold/Oil only | 49.43% | 53.03% |
| Ridge, lagged returns only | 51.14% | 53.03% |

Walk-forward (6 folds): accuracy ranged from 44.7% to 51.9%. No fold beat the majority-class rate by a statistically meaningful margin (all p-values above 0.47).

## Conclusion

No model beat the majority-class rate, and every variant scored close to 50%. With these features, the next-day direction of EUR/USD is not predictable. This matches the main results: R² on returns of 0.006, and price accuracy equal to a naive persistence baseline.

## Notes on how the pipeline got here

- **First version (not in this repo):** the target was the same-day closing price and many features were built from that same close. This produced R² 0.95 and a directional accuracy of about 75%, which came from leakage. The target is now the next-day return.
- **Gold/Oil timing:** with same-day Gold and Oil values, directional accuracy was 60.04%. When those values were lagged by one day, it dropped to about 50%. I treated the 60% as a timing artifact between the data series, not a real signal. The pipeline now uses only the previous day's Gold and Oil data. This was observed during development and is not reproducible with the scripts in this folder.
- I used AI assistance to review and debug the pipeline.
