import pandas as pd
import numpy as np

def load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Encode categorical columns
    df['gender_encoded'] = (df['gender'] == 'Male').astype(int)
    df['marital_encoded'] = (df['marital_status'] == 'Married').astype(int)

    emp_map = {'Employed': 2, 'Self-Employed': 1, 'Unemployed': 0}
    df['employment_encoded'] = df['employment_status'].map(emp_map)

    purpose_map = {'Home Finance': 3, 'Car Finance': 2, 'Personal': 1, 'Credit Card': 0}
    df['purpose_encoded'] = df['loan_purpose'].map(purpose_map)

    # Feature engineering
    df['debt_to_income'] = df['existing_debt_zar'] / (df['monthly_income_zar'] + 1)
    df['loan_to_income'] = df['loan_amount_zar'] / (df['monthly_income_zar'] + 1)
    df['expense_ratio'] = df['monthly_expenses_zar'] / (df['monthly_income_zar'] + 1)

    return df


def get_features_and_target(df: pd.DataFrame):
    features = [
        'age',
        'gender_encoded',
        'marital_encoded',
        'dependants',
        'employment_encoded',
        'years_employed',
        'monthly_income_zar',
        'existing_debt_zar',
        'num_existing_loans',
        'loan_amount_zar',
        'loan_term_months',
        'previous_default',
        'debt_to_income',
        'loan_to_income',
        'expense_ratio',
    ]
    X = df[features]
    y = df['defaulted']
    return X, y


if __name__ == '__main__':
    df = load_and_prepare('data/loan_applicants.csv')
    X, y = get_features_and_target(df)
    print(f'Features shape : {X.shape}')
    print(f'Target shape   : {y.shape}')
    print(f'Default rate   : {y.mean()*100:.1f}%')
    print(f'\nNew features added:')
    print(f'  debt_to_income  mean = {X["debt_to_income"].mean():.2f}')
    print(f'  loan_to_income  mean = {X["loan_to_income"].mean():.2f}')
    print(f'  expense_ratio   mean = {X["expense_ratio"].mean():.2f}')