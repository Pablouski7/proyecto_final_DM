import pandas as pd
import numpy as np
from datetime import timedelta

# ----------------------------
# 1. FEATURES DE COMPORTAMIENTO
# ----------------------------

def get_behavioral_features(visits, orders):
    sessions = visits.groupby('uid').agg(
        n_sessions=('start_ts', 'count'),
        first_session=('start_ts', 'min'),
        last_session=('start_ts', 'max'),
        avg_session_duration=('end_ts', lambda x: ((x - visits.loc[x.index, 'start_ts']).dt.total_seconds().mean()) / 60),
        session_duration_std=('end_ts', lambda x: ((x - visits.loc[x.index, 'start_ts']).dt.total_seconds().std()) / 60)
    ).reset_index()

    order_stats = orders.groupby('uid').agg(
        n_orders=('buy_ts', 'count'),
        revenue_total=('revenue', 'sum'),
        first_order=('buy_ts', 'min'),
        last_order=('buy_ts', 'max'),
        avg_order_value=('revenue', 'mean')
    ).reset_index()

    features = pd.merge(sessions, order_stats, how='left', on='uid')

    features['conversion_delay_days'] = (features['first_order'] - features['first_session']).dt.days
    features['order_span_days'] = (features['last_order'] - features['first_order']).dt.days
    features['orders_per_session'] = features['n_orders'] / features['n_sessions']
    features['avg_days_between_orders'] = (features['order_span_days'] / (features['n_orders'] - 1)).replace([np.inf, -np.inf], np.nan)

    features['days_since_last_session'] = (visits['start_ts'].max() - features['last_session']).dt.days
    features['is_churned'] = features['days_since_last_session'] > 30

    return features

# --------------------------
# 2. FEATURES TEMPORALES
# --------------------------

def get_temporal_features(visits, orders):
    first_session = visits.sort_values('start_ts').drop_duplicates('uid')
    first_session['session_month'] = first_session['start_ts'].dt.month
    first_session['session_quarter'] = first_session['start_ts'].dt.quarter
    first_session['session_day'] = first_session['start_ts'].dt.day
    first_session['session_weekday'] = first_session['start_ts'].dt.weekday
    first_session['session_hour'] = first_session['start_ts'].dt.hour
    first_session['is_weekend_session'] = first_session['session_weekday'] >= 5

    first_order = orders.sort_values('buy_ts').drop_duplicates('uid')
    first_order['conversion_weekday'] = first_order['buy_ts'].dt.weekday

    return first_session[['uid', 'session_month', 'session_day', 'session_quarter', 'session_weekday',
                          'session_hour', 'is_weekend_session']].merge(
           first_order[['uid', 'conversion_weekday']], how='left', on='uid')

# --------------------------
# 3. FEATURES DE MARKETING
# --------------------------

def get_marketing_features(visits, costs, orders):
    last_touch = visits.sort_values('start_ts').drop_duplicates('uid', keep='last')
    basic = last_touch[['uid', 'device', 'source_id']]

    # Tasa de conversión por fuente
    converted_users = orders['uid'].unique()
    total_users_by_source = visits.drop_duplicates('uid').groupby('source_id')['uid'].count()
    converted_by_source = visits[visits['uid'].isin(converted_users)].drop_duplicates('uid').groupby('source_id')['uid'].count()
    source_conversion_rate = (converted_by_source / total_users_by_source).fillna(0).reset_index()
    source_conversion_rate.columns = ['source_id', 'source_conversion_rate']

    # Costo promedio por usuario por fuente
    n_users_by_source = visits.drop_duplicates('uid').groupby('source_id')['uid'].count().reset_index()
    n_users_by_source.columns = ['source_id', 'n_users']
    avg_cost_per_user = costs.groupby('source_id')['costs'].sum().reset_index()
    avg_cost_per_user = avg_cost_per_user.merge(n_users_by_source, on='source_id')
    avg_cost_per_user['avg_cost_per_user'] = avg_cost_per_user['costs'] / avg_cost_per_user['n_users']
    avg_cost_per_user = avg_cost_per_user[['source_id', 'avg_cost_per_user']]

    marketing = basic.merge(source_conversion_rate, on='source_id', how='left')
    marketing = marketing.merge(avg_cost_per_user, on='source_id', how='left')
    return marketing

# --------------------------
# 4. TARGETS CON COHORTES
# --------------------------

def generate_ltv_180(visits, orders):
    first_session = visits.groupby('uid')['start_ts'].min().reset_index()
    first_session.columns = ['uid', 'first_ts']
    orders = orders.merge(first_session, on='uid', how='left')
    orders['days_since_first'] = (orders['buy_ts'] - orders['first_ts']).dt.days
    ltv = orders[orders['days_since_first'] <= 180].groupby('uid')['revenue'].sum().reset_index()
    ltv.columns = ['uid', 'LTV_180']
    return ltv

def generate_cac_source_30(visits, orders, costs):
    first_conversion = orders.groupby('uid')['buy_ts'].min().reset_index()
    first_conversion.columns = ['uid', 'conversion_ts']
    visits_sorted = visits.sort_values('start_ts')
    last_touch = visits_sorted.merge(first_conversion, on='uid', how='inner')
    last_touch = last_touch[last_touch['start_ts'] <= last_touch['conversion_ts']]
    last_source = last_touch.sort_values('start_ts').drop_duplicates('uid', keep='last')[['uid', 'source_id', 'conversion_ts']]

    costs['dt'] = pd.to_datetime(costs['dt'])
    def get_costs(row):
        mask = (costs['dt'] >= row['conversion_ts']) & \
               (costs['dt'] < row['conversion_ts'] + timedelta(days=30)) & \
               (costs['source_id'] == row['source_id'])
        return costs.loc[mask, 'costs'].sum()

    last_source['CAC_source_30'] = last_source.apply(get_costs, axis=1)
    return last_source[['uid', 'CAC_source_30']]

# --------------------------
# 5. FEATURES DE COHORTE
# --------------------------

def get_cohort_features(visits, ltv, cac, orders):
    visits['cohort_month'] = visits['start_ts'].dt.to_period('M')
    cohorts = visits.groupby('uid')['cohort_month'].min().reset_index()

    ltv_cohort = ltv.merge(cohorts, on='uid')
    ltv_avg = ltv_cohort.groupby('cohort_month')['LTV_180'].mean().reset_index()
    ltv_avg.columns = ['cohort_month', 'ltv_cohort_avg']

    cac_cohort = cac.merge(cohorts, on='uid')
    cac_avg = cac_cohort.groupby('cohort_month')['CAC_source_30'].mean().reset_index()
    cac_avg.columns = ['cohort_month', 'cac_cohort_avg']

    n_users = cohorts.groupby('cohort_month')['uid'].count().reset_index()
    n_orders = orders.merge(cohorts, on='uid').groupby('cohort_month')['uid'].nunique().reset_index()
    n_orders.columns = ['cohort_month', 'converted']
    conversion_rate = n_users.merge(n_orders, on='cohort_month', how='left')
    conversion_rate['conversion_rate_cohort'] = conversion_rate['converted'] / conversion_rate['uid']
    conversion_rate = conversion_rate[['cohort_month', 'conversion_rate_cohort']]

    cohort_features = ltv.merge(cohorts, on='uid', how='left')
    cohort_features = cohort_features.merge(ltv_avg, on='cohort_month', how='left')
    cohort_features = cohort_features.merge(cac_avg, on='cohort_month', how='left')
    cohort_features = cohort_features.merge(conversion_rate, on='cohort_month', how='left')
    return cohort_features[['uid', 'ltv_cohort_avg', 'cac_cohort_avg', 'conversion_rate_cohort']]

# --------------------------
# 6. FUNCIÓN PRINCIPAL
# --------------------------

def generate_feature_dataset(visits, orders, costs):
    print("Generando features()...")

    behavioral = get_behavioral_features(visits, orders)
    temporal = get_temporal_features(visits, orders)
    marketing = get_marketing_features(visits, costs, orders)
    ltv = generate_ltv_180(visits, orders)
    cac = generate_cac_source_30(visits, orders, costs)
    cohort_feats = get_cohort_features(visits, ltv, cac, orders)

    df = behavioral.merge(temporal, on='uid', how='left')
    df = df.merge(marketing, on='uid', how='left')
    df = df.merge(ltv, on='uid', how='left')
    df = df.merge(cac, on='uid', how='left')
    df = df.merge(cohort_feats, on='uid', how='left')

    print("Dataset generado. Shape final:", df.shape)
    return df
