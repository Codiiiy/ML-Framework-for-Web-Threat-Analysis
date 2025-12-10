# baseline_comparison.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import your existing classes exactly as train.py does
from train import Config, PhishingDetector, FeatureExtractor

# ----------------------------------------------------------------------
# 1. Define lexical-only columns (must match FeatureExtractor exactly)
# ----------------------------------------------------------------------
LEXICAL_FEATURES = [
    'url_len', 'host_len', 'path_len', 'query_len',
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_digits', 'num_at_signs', 'num_question_marks', 'num_equals',
    'num_ampersands', 'has_ip_host', 'entropy_host', 'entropy_path',
    'entropy_url', 'has_login_kw', 'has_pay_kw', 'has_suspicious_tld',
    'protocol_https', 'subdomain_count', 'digit_letter_ratio',
    'has_punycode', 'has_misspelled_brand', 'keyword_pressure',
    'rare_tld', 'path_depth', 'num_params'
]

# ----------------------------------------------------------------------
# 2. Simple rule-based classifier (tuned from feature_importance.csv)
# ----------------------------------------------------------------------
def rule_based_predict(row):
    """Return 1 (phishing) if any strong heuristic is triggered"""
    if row.get('has_ip_host', 0) == 1:
        return 1
    if row.get('host_len', 0) > 45:
        return 1
    if row.get('num_underscores', 0) > 2:
        return 1
    if row.get('has_login_kw', 0) == 1 and row.get('num_forms', 0) > 1:
        return 1
    if row.get('has_password_field', 0) == 1 and row.get('protocol_https', 0) == 0:
        return 1
    if row.get('num_obfuscated_js', 0) > 0:
        return 1
    return 0

# ----------------------------------------------------------------------
# 3. Main comparison function
# ----------------------------------------------------------------------
def run_baseline_comparison():
    # Use same config as your original training
    config = Config(
        n_estimators=100,
        max_depth=12,
        learning_rate=0.05,
        n_splits=10,
        enable_feature_scaling=True,
        debug_mode=False,
        run_local=True,
        base_dir=".",
        dataset_dir="dataset"
    )

    print("Loading and processing dataset (same as train.py)...")
    detector = PhishingDetector(config)
    X_full, y = detector.prepare_data()   # This is the EXACT same X and y as your main model!

    print(f"Dataset loaded: {X_full.shape[0]} samples, {X_full.shape[1]} features")
    print(f"Class distribution: benign={sum(y==0)}, malicious={sum(y==1)}")

    scaler = StandardScaler() if config.enable_feature_scaling else None

    # ------------------------------------------------------------------
    # 1. Full XGBoost (your main model)
    # ------------------------------------------------------------------
    print("\nEvaluating Full XGBoost (lexical + HTML features)...")
    from train import xgb  # reuse the same model params
    class_counts = np.bincount(y)
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb_metrics = evaluate_model(xgb_model, X_full.copy(), y, config, scaler)

    # ------------------------------------------------------------------
    # 2. Logistic Regression on lexical features only
    # ------------------------------------------------------------------
    print("Evaluating Logistic Regression (lexical features only)...")
    X_lexical = X_full[LEXICAL_FEATURES].copy()
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_metrics = evaluate_model(lr_model, X_lexical, y, config, scaler)

    # ------------------------------------------------------------------
    # 3. Rule-based heuristic classifier
    # ------------------------------------------------------------------
    print("Evaluating Rule-Based heuristic classifier...")
    rule_metrics = evaluate_rule_based(X_full.copy(), y, config)

    # ------------------------------------------------------------------
    # Compile and display results
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        'Full XGBoost (Lexical + HTML)': xgb_metrics,
        'Logistic Regression (Lexical Only)': lr_metrics,
        'Rule-Based Heuristics': rule_metrics
    }).T

    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(results.round(4))
    print("="*80)

    # Save results
    os.makedirs("policies", exist_ok=True)
    output_path = "policies/baseline_comparison.csv"
    results.to_csv(output_path)
    print(f"\nResults saved to {output_path}")

    return results

# ----------------------------------------------------------------------
# Helper evaluation functions
# ----------------------------------------------------------------------
def evaluate_model(model, X, y, config, scaler):
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    metrics = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if scaler:
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics.append({
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_prob)
        })

    return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}

def evaluate_rule_based(X, y, config):
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    metrics = []

    for train_idx, val_idx in skf.split(X, y):
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        y_pred = X_val.apply(rule_based_predict, axis=1)

        metrics.append({
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_pred)
        })

    return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_baseline_comparison()