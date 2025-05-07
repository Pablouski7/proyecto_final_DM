import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test, y_test):
    # Forzar DataFrame si el modelo espera feature_names
    if not isinstance(X_test, pd.DataFrame) and hasattr(model, 'feature_names_in_'):
        X_test = pd.DataFrame(X_test, columns=model.feature_names_in_)

    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cálculo de MAPE evitando ceros en y_test
    y_test_arr = np.asarray(y_test).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    non_zero = y_test_arr != 0
    mape = np.mean(np.abs((y_test_arr[non_zero] - y_pred_arr[non_zero]) / y_test_arr[non_zero])) * 100

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
