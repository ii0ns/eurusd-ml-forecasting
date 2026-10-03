from src.data_engineer import EnhancedDataEngineer
from src.model_developer import OptimizedModelDeveloper
from src.evaluator import ModelEvaluator
from datetime import datetime, timedelta

print("="*70)
print("EUR/USD FORECASTING — RANDOM FOREST")
print("="*70)

engineer = EnhancedDataEngineer(start_date="2015-01-01", end_date="2025-04-30")
processed_data = engineer.process_all()

developer = OptimizedModelDeveloper(processed_data)
developer.prepare_data()
developer.train_model()
results = developer.evaluate_model()
developer.save_model()
developer.export_results()

evaluator = ModelEvaluator(developer.model, developer.X_test,
                           developer.actual_price, developer.pred_price)
evaluator.create_plots()

print("\n" + "="*70)
print("DONE!")
print("="*70)