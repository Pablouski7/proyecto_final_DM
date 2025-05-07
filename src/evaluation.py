from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def evaluate_all(models: dict, X_test, y_test):
    results = []
    for name, model in models.items():
        try:
            metrics = evaluate_model(model, X_test, y_test)
            metrics["Model"] = name
            results.append(metrics)
        except Exception as e:
            print(f"⚠️ Error evaluando {name}: {e}")
    return pd.DataFrame(results).set_index("Model")
