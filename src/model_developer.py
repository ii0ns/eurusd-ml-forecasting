import json, os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

TARGET = 'target_return'


class OptimizedModelDeveloper:

    def __init__(self, data):
        self.data = data
        self.model = None

    def prepare_data(self, test_size=0.2):
        print("Preparing data...")
        X = self.data.drop(columns=['EUR_USD', TARGET])
        y = self.data[TARGET]
        close = self.data['EUR_USD']

        split = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X.iloc[:split], X.iloc[split:]
        self.y_train, self.y_test = y.iloc[:split], y.iloc[split:]
        self.close_test = close.iloc[split:]     # close at time t, used to rebuild prices

        print(f"✓ Train: {len(self.X_train)} ({X.index[0].date()} → {X.index[split-1].date()})")
        print(f"✓ Test:  {len(self.X_test)} ({X.index[split].date()} → {X.index[-1].date()})")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        print("\nTraining Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', bootstrap=True,
            n_jobs=-1, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained")
        return self.model

    def evaluate_model(self):
        print("\nEvaluating model...")
        pred_ret = self.model.predict(self.X_test)
        true_ret = self.y_test.values
        c = self.close_test.values

        actual_price = c * (1 + true_ret)        # next-day close
        pred_price = c * (1 + pred_ret)          # model's next-day close
        base_price = c                           # persistence: "tomorrow = today"

        def price_metrics(p):
            return {
                'rmse': float(np.sqrt(mean_squared_error(actual_price, p))),
                'mae': float(mean_absolute_error(actual_price, p)),
                'r2': float(r2_score(actual_price, p)),
                'mape': float(np.mean(np.abs((actual_price - p) / actual_price)) * 100),
            }

        model_m, base_m = price_metrics(pred_price), price_metrics(base_price)
        dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(true_ret)) * 100)
        always_up = float(np.mean(true_ret > 0) * 100)   # naive direction baseline
        r2_returns = float(r2_score(true_ret, pred_ret))

        print("\n" + "=" * 55)
        print(f"{'':24}{'MODEL':>14}{'PERSISTENCE':>14}")
        for k in ['rmse', 'mae', 'r2', 'mape']:
            print(f"{k.upper():24}{model_m[k]:>14.6f}{base_m[k]:>14.6f}")
        print(f"R² on returns:          {r2_returns:.4f}  (0 = no skill)")
        print(f"Directional accuracy:   {dir_acc:.2f}%  (always-up baseline: {always_up:.2f}%)")
        print("=" * 55)

        idx = self.X_test.index
        self.actual_price = pd.Series(actual_price, index=idx)
        self.pred_price = pd.Series(pred_price, index=idx)
        self.results = {'model': model_m, 'baseline': base_m, 'r2_returns': r2_returns,
                        'directional_accuracy': dir_acc, 'always_up_accuracy': always_up}
        return self.results

    def save_model(self):
        joblib.dump(self.model, 'trained_model.pkl')
        print("✓ Model saved as 'trained_model.pkl'")

    def export_results(self, out_dir='export'):
        os.makedirs(out_dir, exist_ok=True)
        r, m = self.results, self.results['model']

        with open(f'{out_dir}/predictions.json', 'w') as f:
            json.dump({'dates': [d.strftime('%Y-%m-%d') for d in self.actual_price.index],
                       'actual': self.actual_price.round(5).tolist(),
                       'predicted': self.pred_price.round(5).tolist()}, f)

        imp = pd.Series(self.model.feature_importances_, index=self.X_test.columns)
        imp = imp.sort_values(ascending=False).head(12)
        with open(f'{out_dir}/feature_importance.json', 'w') as f:
            json.dump({'features': imp.index.tolist(),
                       'importances': imp.round(4).tolist()}, f)

        with open(f'{out_dir}/metrics.json', 'w') as f:
            json.dump({
                'r2': round(m['r2'], 4), 'rmse': round(m['rmse'], 6),
                'mae': round(m['mae'], 6), 'mape': round(m['mape'], 3),
                'directional_accuracy': round(r['directional_accuracy'], 2),
                'always_up_accuracy': round(r['always_up_accuracy'], 2),
                'r2_returns': round(r['r2_returns'], 4),
                'baseline_r2': round(r['baseline']['r2'], 4),
                'baseline_rmse': round(r['baseline']['rmse'], 6),
                'baseline_mae': round(r['baseline']['mae'], 6),
                'train_samples': len(self.X_train), 'test_samples': len(self.X_test),
                'total_features': self.X_train.shape[1],
                'model': 'RandomForestRegressor',
                'n_estimators': self.model.n_estimators, 'max_depth': self.model.max_depth,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }, f, indent=2)
        print(f"✓ JSON exported to '{out_dir}/'")