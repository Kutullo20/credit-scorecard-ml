import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_prepare, get_features_and_target
from scorecard import train_model, DROP_FEATURES


def gini_coefficient(y_true, y_prob) -> float:
    """Gini = 2 * AUC - 1. Ranges from -1 (worst) to 1 (best)."""
    auc  = roc_auc_score(y_true, y_prob)
    return round((2 * auc - 1) * 100, 2)


def ks_statistic(y_true, y_prob) -> float:
    """ KS = max separation between cumulative good and bad distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(max(tpr - fpr) * 100, 2)


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')

    # ROC Curve
    ax = axes[0]
    ax.set_facecolor('#0f1117')
    ax.plot(fpr, tpr, color='#4f8ef7', linewidth=2.5,
            label=f'ROC Curve (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color='#374151', linewidth=1,
            linestyle='--', label='Random model')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4f8ef7')
    ax.set_title('ROC Curve', color='#e5e7eb', fontsize=12, pad=12)
    ax.set_xlabel('False Positive Rate', color='#9ca3af')
    ax.set_ylabel('True Positive Rate', color='#9ca3af')
    ax.tick_params(colors='#6b7280')
    ax.legend(facecolor='#1e2130', labelcolor='#e5e7eb', fontsize=9)
    ax.grid(True, color='#1e2130', linewidth=0.6)

    # Confusion Matrix
    ax = axes[1]
    ax.set_facecolor('#0f1117')
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap='Blues')
    labels = ['Paid Back', 'Defaulted']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color='#9ca3af')
    ax.set_yticklabels(labels, color='#9ca3af')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='#e5e7eb', fontsize=16, fontweight='bold')
    ax.set_title('Confusion Matrix', color='#e5e7eb', fontsize=12, pad=12)
    ax.set_xlabel('Predicted', color='#9ca3af')
    ax.set_ylabel('Actual', color='#9ca3af')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/evaluation_charts.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.show()


if __name__ == '__main__':
    df = load_and_prepare('data/loan_applicants.csv')
    X, y = get_features_and_target(df)
    X = X.drop(columns=DROP_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, scaler = train_model(X_train, y_train)
    y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]

    gini = gini_coefficient(y_test, y_prob)
    ks   = ks_statistic(y_test, y_prob)
    auc  = round(roc_auc_score(y_test, y_prob), 4)

    print('MODEL EVALUATION \n')
    print(f'  ROC-AUC Score  : {auc}')
    print(f'  Gini           : {gini}%')
    print(f'  KS Statistic   : {ks}%')
    print()

    def interpret_gini(g):
        if g >= 60: return 'Excellent'
        elif g >= 40: return 'Good'
        elif g >= 20: return 'Acceptable'
        else: return 'Poor'

    print(f'  Model strength : {interpret_gini(gini)}')
    print()
    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred,
          target_names=['Paid Back', 'Defaulted']))
    plot_roc_curve(y_test, y_prob)
    print('Charts saved in outputs/evaluation_charts.png')