import pandas as pd
import numpy as np
import yfinance as yf


class EnhancedDataEngineer:

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def _download_close(self, ticker):
        df = yf.download(ticker, start=self.start_date, end=self.end_date,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        close = df['Close']
        if isinstance(close, pd.DataFrame):      # newer yfinance returns a column per ticker
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        return close

    def collect_data(self):
        print("Collecting data...")
        # EUR/USD is required: fail loudly rather than fall back to fake data
        eur = self._download_close('EURUSD=X')
        self.data = pd.DataFrame({'EUR_USD': eur})
        print(f"✓ EUR/USD: {len(eur)} rows")

        for name, ticker in [('Gold', 'GC=F'), ('Oil', 'CL=F')]:
            try:
                self.data = self.data.join(self._download_close(ticker).rename(name), how='left')
                print(f"✓ {name} added")
            except Exception as e:
                print(f"  {name} skipped: {e}")

        self.data = self.data.ffill()            # forward-fill only, never backfill
        return self.data

    def create_enhanced_features(self):
        print("Creating features...")
        d = self.data
        close = d['EUR_USD']
        ret = close.pct_change()

        # Returns
        for n in [1, 3, 7]:
            d[f'return_{n}d'] = close.pct_change(n)

        # Volatility
        d['volatility_7'] = ret.rolling(7).std()
        d['volatility_14'] = ret.rolling(14).std()

        # Moving averages as relative (stationary) features
        ma7, ma21, ma50 = (close.rolling(n).mean() for n in (7, 21, 50))
        d['price_vs_MA7'] = close / ma7 - 1
        d['price_vs_MA21'] = close / ma21 - 1
        d['price_vs_MA50'] = close / ma50 - 1
        d['MA_cross_7_21'] = ma7 / ma21
        d['MA_cross_21_50'] = ma21 / ma50

        # RSI (14, simple rolling mean)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-12))
        d['RSI_signal'] = (d['RSI'] > 70).astype(int) - (d['RSI'] < 30).astype(int)

        # Lagged returns
        for lag in [1, 2, 3, 5, 7, 14]:
            d[f'return_lag_{lag}'] = ret.shift(lag)

        # Rolling range, relative
        high7, low7 = close.rolling(7).max(), close.rolling(7).min()
        d['range_7d_pct'] = (high7 - low7) / close
        d['pos_in_range_7d'] = (close - low7) / (high7 - low7 + 1e-12)

        # Exogenous assets: previous day's values only (same-day values leak, see diagnostics)
        for col in ['Gold', 'Oil']:
            if col in d.columns:
                g = d[col].shift(1)
                d[f'{col}_return'] = g.pct_change()
                d[f'{col}_return_7d'] = g.pct_change(7)
                d.drop(columns=col, inplace=True)

        # Calendar
        d['day_of_week'] = d.index.dayofweek
        d['month'] = d.index.month
        d['quarter'] = d.index.quarter

        # TARGET: next day's return (uses information strictly after time t)
        d['target_return'] = ret.shift(-1)

        self.data = d
        n_features = len(d.columns) - 2          # minus EUR_USD and target_return
        print(f"✓ Created {n_features} features")
        return d

    def process_all(self):
        self.collect_data()
        self.create_enhanced_features()
        before = len(self.data)
        self.data = self.data.dropna()
        print(f"✓ Dropped {before - len(self.data)} rows with NaN, final: {len(self.data)} rows")
        self.data.to_csv('processed_data.csv')
        return self.data