import gradio as gr
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import load_and_prepare, get_features_and_target
from scorecard import train_model, build_points_table, score_applicant, get_decision, DROP_FEATURES
from sklearn.model_selection import train_test_split

# Train model once on startup
df       = load_and_prepare('data/loan_applicants.csv')
X, y     = get_features_and_target(df)
X        = X.drop(columns=DROP_FEATURES)
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model, scaler     = train_model(X_train, y_train)
df_points, base_pts = build_points_table(model, scaler, features)


# Scoring function 
def score(age, employment, years_employed, monthly_income,
          monthly_expenses, existing_debt, num_loans,
          loan_amount, loan_purpose, loan_term, previous_default,
          dependants):

    emp_map     = {'Employed': 2, 'Self-Employed': 1, 'Unemployed': 0}
    purpose_map = {'Home Finance': 3, 'Car Finance': 2, 'Personal': 1, 'Credit Card': 0}

    row = pd.Series({
        'age'                 : age,
        'dependants'          : dependants,
        'employment_encoded'  : emp_map[employment],
        'years_employed'      : years_employed,
        'monthly_income_zar'  : monthly_income,
        'existing_debt_zar'   : existing_debt,
        'num_existing_loans'  : num_loans,
        'loan_amount_zar'     : loan_amount,
        'loan_term_months'    : loan_term,
        'previous_default'    : int(previous_default),
        'debt_to_income'      : existing_debt / (monthly_income + 1),
        'loan_to_income'      : loan_amount   / (monthly_income + 1),
        'expense_ratio'       : monthly_expenses / (monthly_income + 1),
    })

    credit_score = score_applicant(row, model, scaler, features, df_points, base_pts)
    decision     = get_decision(credit_score)

    # Risk factors
    flags = []
    if row['debt_to_income']   > 2:    flags.append('High existing debt vs income')
    if row['loan_to_income']   > 10:   flags.append('Loan amount very high vs income')
    if row['expense_ratio']    > 0.7:  flags.append('High monthly expenses')
    if previous_default:               flags.append('Previous default on record')
    if employment == 'Unemployed':     flags.append('No employment income')
    if not flags:                      flags.append('No major risk flags')

    risk_summary = '\n'.join(flags)

    return (
        f"{credit_score} / 850",
        decision,
        risk_summary
    )


# Gradio UI 
with gr.Blocks(title='Credit Scorecard', theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Credit Scorecard — Loan Applicant Scorer
    Enter applicant details below to get an instant credit score and decision.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Personal Details")
            age          = gr.Slider(18, 70, value=35, step=1, label='Age')
            dependants   = gr.Slider(0, 5, value=1, step=1, label='Number of Dependants')
            employment   = gr.Dropdown(
                ['Employed', 'Self-Employed', 'Unemployed'],
                value='Employed', label='Employment Status'
            )
            years_employed = gr.Slider(0, 30, value=5, step=1, label='Years Employed')
            previous_default = gr.Checkbox(label='Previous Default', value=False)

        with gr.Column():
            gr.Markdown("### Financial Details")
            monthly_income   = gr.Number(value=25000, label='Monthly Income (ZAR)')
            monthly_expenses = gr.Number(value=12000, label='Monthly Expenses (ZAR)')
            existing_debt    = gr.Number(value=5000,  label='Existing Debt (ZAR)')
            num_loans        = gr.Slider(0, 5, value=1, step=1, label='Number of Existing Loans')

        with gr.Column():
            gr.Markdown("### Loan Details")
            loan_amount  = gr.Number(value=50000, label='Loan Amount (ZAR)')
            loan_purpose = gr.Dropdown(
                ['Personal', 'Car Finance', 'Home Finance', 'Credit Card'],
                value='Personal', label='Loan Purpose'
            )
            loan_term = gr.Dropdown(
                [12, 24, 36, 48, 60],
                value=36, label='Loan Term (Months)'
            )

    score_btn = gr.Button('Calculate Credit Score', variant='primary')

    with gr.Row():
        credit_score_out = gr.Text(label='Credit Score')
        decision_out     = gr.Text(label='Decision')
        risk_out         = gr.Text(label='Risk Flags', lines=5)

    score_btn.click(
        fn=score,
        inputs=[age, employment, years_employed, monthly_income,
                monthly_expenses, existing_debt, num_loans,
                loan_amount, loan_purpose, loan_term,
                previous_default, dependants],
        outputs=[credit_score_out, decision_out, risk_out]
    )

if __name__ == '__main__':
    demo.launch()