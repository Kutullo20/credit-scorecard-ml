import pandas as pd
import numpy as np


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 5):
    """Calculate Weight of Evidence (WOE) and Information Value (IV)for a single feature against the target variable."""
    df_temp = df[[feature, target]].copy()

    # Bin continuous variables
    if df_temp[feature].nunique() > 10:
        df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')
    else:
        df_temp['bin'] = df_temp[feature]

    # Count events (defaulted=1) and non-events (defaulted=0) per bin
    grouped = df_temp.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    # Totals
    total_events     = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    # Distribution
    grouped['dist_events']     = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events

    # Avoid log(0)
    grouped['dist_events']     = grouped['dist_events'].replace(0, 0.0001)
    grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 0.0001)

    # WOE and IV
    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv']  = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']

    iv_total = grouped['iv'].sum()
    return grouped[['events', 'non_events', 'woe', 'iv']], iv_total


def get_iv_summary(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """ Calculate IV for all features and rank by predictive power."""
    results = []
    for feature in features:
        try:
            _, iv = calculate_woe_iv(df, feature, target)
            results.append({'feature': feature, 'iv': round(iv, 4)})
        except Exception:
            pass

    iv_df = pd.DataFrame(results).sort_values('iv', ascending=False)

    # IV interpretation used in banking
    def interpret(iv):
        if iv < 0.02:   return 'Useless'
        elif iv < 0.1:  return 'Weak'
        elif iv < 0.3:  return 'Medium'
        elif iv < 0.5:  return 'Strong'
        else:           return 'Very Strong'

    iv_df['strength'] = iv_df['iv'].apply(interpret)
    return iv_df.reset_index(drop=True)


if __name__ == '__main__':
    from preprocess import load_and_prepare, get_features_and_target

    df = load_and_prepare('data/loan_applicants.csv')
    X, y = get_features_and_target(df)

    features = X.columns.tolist()
    iv_summary = get_iv_summary(df, features, 'defaulted')

    print(iv_summary.to_string(index=False))