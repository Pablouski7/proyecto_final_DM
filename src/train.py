# src/train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

# Función de carga y preparación

def load_and_prepare_data(path, target, date_column='first_session', cutoff='2018-07-01'):
    df = pd.read_csv(path, parse_dates=[date_column])
    df = df[df[date_column] < cutoff]  # Filtro temporal

    # Filtrar registros válidos para el target
    df = df[~df[target].isna()].copy()

    # Separar X e y
    y = df[target]
    X = df.drop(columns=[target, 'uid', 'first_session', 'last_session', 'first_order', 'last_order',
                         'LTV_180', 'CAC_source_30', 'ltv_cohort_avg', 'cac_cohort_avg', 'conversion_rate_cohort'],
                errors='ignore')

    return X, y

# Preprocesamiento automático

def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    # Imputadores
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinación
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

# Entrenamiento de modelos individuales
def train_models(X_train, y_train, preprocessor):
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'sgd': SGDRegressor(max_iter=1000, tol=1e-3),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'cat': cb.CatBoostRegressor(verbose=0, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

    return trained_models

# Ensamblador (Stacking)
def train_stacking_model(X_train, y_train, preprocessor, base_models):
    estimators = [(name, model.named_steps['regressor']) for name, model in base_models.items()]
    final_estimator = Ridge()

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=True
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacking', stacking)
    ])
    pipe.fit(X_train, y_train)
    return pipe

# Guardado de modelos

def save_models(trained_models: dict, target_name: str, save_path: str = 'models/'):
    os.makedirs(save_path, exist_ok=True)
    for name, model in trained_models.items():
        filename = f"{save_path}{target_name}_{name}.pkl"
        joblib.dump(model, filename)
    print(f"Modelos guardados exitosamente en {save_path}")
