import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_prepare, get_features_and_target


# Features to DROP based on WOE/IV analysis (useless predictors)
DROP_FEATURES = ['gender_encoded', 'marital_encoded']

# Scorecard scaling constants (industry standard)
BASE_SCORE = 600
PDO        = 50   # Points to Double the Odds
BASE_ODDS  = 1    # Odds at base score


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y_train)
    return model, scaler


def build_points_table(model, scaler, features: list) -> pd.DataFrame:
    """ Convert logistic regression coefficients into scorecard points."""
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    coefficients = model.coef_[0]
    intercept    = model.intercept_[0]

    points = []
    for i, feature in enumerate(features):
        coef        = coefficients[i]
        scale       = scaler.scale_[i]
        mean        = scaler.mean_[i]
        pts         = -factor * (coef / scale)
        points.append({
            'feature'     : feature,
            'coefficient' : round(coef, 4),
            'points'      : round(pts, 1),
        })

    # Intercept contribution
    intercept_pts = factor * (intercept + sum(
        coefficients[i] * scaler.mean_[i] / scaler.scale_[i]
        for i in range(len(features))
    )) + offset

    df_points = pd.DataFrame(points).sort_values('points')
    return df_points, round(intercept_pts, 1)


def score_applicant(row: pd.Series, model, scaler, features: list,
                    df_points: pd.DataFrame, base_pts: float) -> int:
    """ Score a single applicant and return their credit score."""
    X = row[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    log_odds  = model.decision_function(X_scaled)[0]
    factor    = PDO / np.log(2)
    score     = int(BASE_SCORE - factor * log_odds)
    return max(300, min(850, score))


def get_decision(score: int) -> str:
    if score >= 650:   return 'APPROVED'
    elif score >= 500: return 'MANUAL REVIEW'
    else:              return 'DECLINED'


if __name__ == '__main__':
    # Load and prepare
    df = load_and_prepare('data/loan_applicants.csv')
    X, y = get_features_and_target(df)

    # Drop useless features
    X = X.drop(columns=DROP_FEATURES)
    features = X.columns.tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model, scaler = train_model(X_train, y_train)
    print(f'Training accuracy : {model.score(scaler.transform(X_train), y_train)*100:.1f}%')
    print(f'Test accuracy     : {model.score(scaler.transform(X_test), y_test)*100:.1f}%')

    # Points table
    df_points, base_pts = build_points_table(model, scaler, features)
    print(f'\n== SCORECARD POINTS TABLE ==\n')
    print(df_points.to_string(index=False))
    print(f'\nBase score contribution: {base_pts}')

    # Save points table
    os.makedirs('outputs', exist_ok=True)
    df_points.to_csv('outputs/scorecard_points.csv', index=False)
    print(f'\nSaved → outputs/scorecard_points.csv')

    # Score first 5 applicants as example
    print(f'\n== SAMPLE APPLICANT SCORES ==\n')
    for i in range(5):
        row   = X.iloc[i]
        score = score_applicant(row, model, scaler, features, df_points, base_pts)
        decision = get_decision(score)
        actual   = '(actually defaulted)' if y.iloc[i] == 1 else '(actually paid back)'
        print(f'Applicant {i+1}: Score={score}  {decision}  {actual}')