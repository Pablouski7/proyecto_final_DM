import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

def split_data_by_date(df, date_column):
    train_df = df[df[date_column] < '2018-01-01']
    val_df = df[(df[date_column] >= '2018-01-01') & (df[date_column] < '2018-04-01')]
    test_df = df[df[date_column] >= '2018-04-01']
    return train_df, val_df, test_df

# Función de carga y preparación
def load_and_prepare_data(path, target, date_column='first_session'):
    df = pd.read_csv(path, parse_dates=[date_column])

    # Filtrar por fechas para crear los conjuntos de datos
    train_df, val_df, test_df = split_data_by_date(df, date_column)

    # Eliminar filas con valores nulos en el target
    train_df = train_df[~train_df[target].isna()].copy()
    val_df = val_df[~val_df[target].isna()].copy()
    test_df = test_df[~test_df[target].isna()].copy()

    # Separar características (X) y etiquetas (y)
    def split_X_y(df):
        y = df[target]
        X = df.drop(columns=[target, 'uid', 'first_session', 'last_session', 'first_order', 'last_order',
                             'LTV_180', 'CAC_source_30', 'ltv_cohort_avg', 'cac_cohort_avg', 'conversion_rate_cohort'],
                    errors='ignore')
        return X, y

    X_train, y_train = split_X_y(train_df)
    X_val, y_val = split_X_y(val_df)
    X_test, y_test = split_X_y(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Preprocesamiento automático
def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

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

# Ensamblador (Stacking) con preprocesamiento separado
def train_stacking_model(X_train, y_train, preprocessor, base_models):
    # Ajustar el preprocesador en X_train para usarlo luego
    preprocessor.fit(X_train)

    # Creamos transformer que aplica preprocesamiento ya ajustado
    ft = FunctionTransformer(preprocessor.transform, validate=False)

    # Cada estimador base en su propio pipeline
    estimators = []
    for name, pipe in base_models.items():
        estimators.append((
            name,
            Pipeline(steps=[
                ('preprocessor', ft),
                ('regressor', pipe.named_steps['regressor'])
            ])
        ))

    final_estimator = Ridge()
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False
    )

    # Ajuste directo (transformación dentro de pipelines internos)
    stacking.fit(X_train, y_train)
    return stacking

# Guardado de modelos
def save_models(trained_models: dict, target_name: str, save_path: str = 'models/'):
    os.makedirs(save_path, exist_ok=True)
    for name, model in trained_models.items():
        filename = f"{save_path}{target_name}_{name}.pkl"
        joblib.dump(model, filename)
    print(f"Modelos guardados exitosamente en {save_path}")