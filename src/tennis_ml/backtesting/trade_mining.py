"""
Exploratory filters and clusters for historical bet ledgers.

These reports are diagnostics only. Any profitable pocket found here must be
promoted through nested walk-forward selection before it is deployable.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pnl_backtester import PnLBacktester


FILTER_COLUMNS = [
    'model_type',
    'surface',
    'tournament_importance',
    'round',
    'odds_bucket',
    'probability_bucket',
    'edge_bucket',
    'clv_bucket',
]

CLUSTER_NUMERIC_FEATURES = [
    'model_probability',
    'market_probability',
    'odds',
    'edge',
    'ev',
    'clv',
    'market_overround',
    'friction_adjusted_ev',
    'fractional_kelly_25',
]

CLUSTER_CATEGORICAL_FEATURES = [
    'model_type',
    'surface',
    'round',
    'tournament_importance',
]


def mine_trades(
    bets: pd.DataFrame,
    stake: float = 1.0,
    min_bets: int = 30,
    clusters: int = 8,
) -> dict[str, pd.DataFrame]:
    prepared = prepare_bets_for_mining(bets)
    clustered = cluster_bets(prepared, clusters=clusters) if len(prepared) >= clusters else prepared.assign(cluster=pd.NA)
    return {
        'filter_profit_report': filter_profit_report(prepared, stake=stake, min_bets=min_bets),
        'profitable_filter_rules': profitable_filter_rules(prepared, stake=stake, min_bets=min_bets),
        'cluster_report': cluster_report(clustered, stake=stake),
        'clustered_bets': clustered,
    }


def prepare_bets_for_mining(bets: pd.DataFrame) -> pd.DataFrame:
    frame = bets.copy()
    frame['profit_metric'] = frame['friction_adjusted_profit'] if 'friction_adjusted_profit' in frame.columns else frame['profit']
    frame['odds_bucket'] = pd.cut(
        frame['odds'],
        bins=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0],
        labels=['1-1.5', '1.5-2', '2-3', '3-5', '5-10', '10+'],
        include_lowest=True,
    ).astype(str)
    frame['probability_bucket'] = pd.cut(
        frame['model_probability'],
        bins=[0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
        labels=['0-25%', '25-40%', '40-50%', '50-60%', '60-75%', '75-100%'],
        include_lowest=True,
    ).astype(str)
    frame['edge_bucket'] = pd.cut(
        frame['edge'],
        bins=[-1, 0.03, 0.05, 0.075, 0.10, 1],
        labels=['<=3%', '3-5%', '5-7.5%', '7.5-10%', '10%+'],
        include_lowest=True,
    ).astype(str)
    frame['clv_bucket'] = pd.cut(
        frame['clv'].fillna(0),
        bins=[-10, -0.05, 0, 0.025, 0.05, 10],
        labels=['<-5%', '-5-0%', '0-2.5%', '2.5-5%', '5%+'],
        include_lowest=True,
    ).astype(str)
    return frame


def filter_profit_report(bets: pd.DataFrame, stake: float, min_bets: int) -> pd.DataFrame:
    rows = []
    for column in FILTER_COLUMNS:
        for value, group in bets.groupby(column, dropna=False):
            if len(group) < min_bets:
                continue
            summary = PnLBacktester(stake=stake).summarize(group)
            summary['filter'] = column
            summary['value'] = value
            rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        by=['friction_adjusted_roi', 'bets'],
        ascending=[False, False],
    ) if rows else pd.DataFrame()


def profitable_filter_rules(bets: pd.DataFrame, stake: float, min_bets: int) -> pd.DataFrame:
    rows = []
    for size in [2, 3]:
        for columns in combinations(FILTER_COLUMNS, size):
            for values, group in bets.groupby(list(columns), dropna=False):
                if len(group) < min_bets:
                    continue
                summary = PnLBacktester(stake=stake).summarize(group)
                if summary['friction_adjusted_roi'] <= 0:
                    continue
                summary['rule'] = ' & '.join(f'{column}={value}' for column, value in zip(columns, values))
                summary['rule_size'] = size
                rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        by=['friction_adjusted_roi', 'bets'],
        ascending=[False, False],
    ) if rows else pd.DataFrame()


def cluster_bets(bets: pd.DataFrame, clusters: int = 8) -> pd.DataFrame:
    frame = bets.copy()
    numeric = [column for column in CLUSTER_NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in CLUSTER_CATEGORICAL_FEATURES if column in frame.columns]
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    matrix = preprocessor.fit_transform(frame[numeric + categorical])
    model = KMeans(n_clusters=min(clusters, len(frame)), random_state=42, n_init=20)
    frame['cluster'] = model.fit_predict(matrix)
    return frame


def cluster_report(bets: pd.DataFrame, stake: float) -> pd.DataFrame:
    if bets.empty or 'cluster' not in bets.columns:
        return pd.DataFrame()
    rows = []
    for cluster, group in bets.groupby('cluster', dropna=False):
        summary = PnLBacktester(stake=stake).summarize(group)
        summary['cluster'] = cluster
        summary['top_model_type'] = mode_or_blank(group.get('model_type'))
        summary['top_surface'] = mode_or_blank(group.get('surface'))
        summary['top_round'] = mode_or_blank(group.get('round'))
        summary['average_market_probability'] = float(group['market_probability'].mean()) if 'market_probability' in group else 0.0
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        by=['friction_adjusted_roi', 'bets'],
        ascending=[False, False],
    )


def mode_or_blank(series) -> str:
    if series is None or series.empty:
        return ''
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ''
