# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:06:33 2025

@author: iamwa
"""

"""
coffee_rust_full_pipeline.py

Full analysis pipeline:
- EDA & plotting
- Preprocessing (encoding, scaling, imputation)
- Baseline ML: LogisticRegression, RandomForest, XGBoost
- AutoML: FLAML
- Explainable AI: SHAP, PDP
- Bayesian: PyMC hierarchical logistic (incidence), Beta regression (severity)
- Hybrid: ML outputs as covariates in Bayesian model
- Spatio-temporal Group CV (by County)
- Save plots to ./plots/ and models to ./models/

Author: Generated for user
"""

# -----------------------
# Imports & setup
# -----------------------
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# AutoML
from flaml import AutoML

# Explainable AI
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Bayesian
import pymc as pm
import arviz as az

# Utility
import joblib
import json
from datetime import datetime

# -----------------------
# Paths & create folders
# -----------------------
PLOTS_DIR = "plots"
MODELS_DIR = "models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# Load data
# -----------------------
DATAFILE = "C:/Users/iamwa/Desktop/Coffee_rust.xlsx"  # replace with your file path
df = pd.read_excel(DATAFILE)

# quick check
print("Rows:", len(df))
print(df.columns.tolist())

# -----------------------
# Variable lists
# -----------------------
TARGET_INC = "Incidence"
TARGET_SEV = "Severity_pct"

# features list (the 15 features)
features = [
    "Daily_Relative_Humidity_pct", "Daily_Temperature_C", "Precipitation_mm_day",
    "Leaf_Wetness_hours_day", "Elevation_m", "Coffee_variety", "Plant_age_years",
    "Shade_pct", "Fungicide_use", "Fungicide_freq_per_season",
    "Past_outbreak_history", "Lagged_incidence_prev_week", "NDVI",
    "Distance_to_nearest_infected_farm_m"
]

# Keep County for grouping/spatial CV & plotting
GROUP_COL = "County"

# -----------------------
# 0) Basic EDA & plots
# -----------------------
def save_fig(fig, name, tight=True):
    path = os.path.join(PLOTS_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# 0.1: target balance
fig = plt.figure(figsize=(6,4))
sns.countplot(x=TARGET_INC, data=df)
plt.title("Incidence (class balance)")
save_fig(fig, "incidence_balance.png")

# 0.2: severity distribution (only where incidence==1)
fig = plt.figure(figsize=(8,4))
sns.histplot(df[TARGET_SEV], bins=40, kde=True)
plt.title("Severity (%) distribution")
save_fig(fig, "severity_distribution.png")

# 0.3: pairwise correlation heatmap for continuous features
cont_feats = [
    "Daily_Relative_Humidity_pct","Daily_Temperature_C","Precipitation_mm_day",
    "Leaf_Wetness_hours_day","Elevation_m","Plant_age_years","Shade_pct","NDVI","Distance_to_nearest_infected_farm_m"
]
corr = df[cont_feats].corr()
fig = plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix (continuous features)")
save_fig(fig, "correlation_matrix.png")

# 0.4 County counts
fig = plt.figure(figsize=(8,4))
sns.countplot(y=GROUP_COL, data=df, order=df[GROUP_COL].value_counts().index)
plt.title("Observations per County")
save_fig(fig, "county_counts.png")

# 0.5 Boxplots for top drivers by incidence
fig, axs = plt.subplots(2,3, figsize=(14,8))
top_plot_vars = ["Daily_Relative_Humidity_pct","Leaf_Wetness_hours_day","Precipitation_mm_day",
                 "NDVI","Shade_pct","Distance_to_nearest_infected_farm_m"]
for ax, v in zip(axs.flatten(), top_plot_vars):
    sns.boxplot(x=TARGET_INC, y=v, data=df, ax=ax)
    ax.set_title(v)
save_fig(fig, "boxplots_by_incidence.png")

# 0.6 Save a basic summary table
df[features + [TARGET_INC, TARGET_SEV, GROUP_COL]].describe().to_csv(os.path.join(PLOTS_DIR, "basic_describe.csv"))

# -----------------------
# 1) Preprocessing pipeline
# -----------------------
# Identify feature types
numeric_features = [
    "Daily_Relative_Humidity_pct","Daily_Temperature_C","Precipitation_mm_day",
    "Leaf_Wetness_hours_day","Elevation_m","Plant_age_years","Shade_pct","NDVI","Distance_to_nearest_infected_farm_m"
]
categorical_features = ["Coffee_variety","Fungicide_use","Past_outbreak_history","Lagged_incidence_prev_week","Fungicide_freq_per_season"]
# Note: Freq_per_season is numeric but often small ints; treat as numeric instead:
categorical_features = ["Coffee_variety","Fungicide_use","Past_outbreak_history","Lagged_incidence_prev_week"]
numeric_features_extended = numeric_features + ["Fungicide_freq_per_season"]

# Imputer/scaler for numeric
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical encoder
import sklearn
from sklearn.preprocessing import OneHotEncoder

# Check sklearn version
sklearn_version = sklearn.__version__

if sklearn_version >= "1.2":
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot)
])


preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features_extended),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# Fit preprocessor on full data for later use
X_all = df[numeric_features_extended + categorical_features]
preprocessor.fit(X_all)

# Transform full feature matrix (for ML)
X = preprocessor.transform(X_all)
# Get transformed column names (for SHAP / PDP)
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
num_names = numeric_features_extended
feature_names = list(num_names) + list(cat_names)

print("Feature vector length:", len(feature_names))

# Targets
y_inc = df[TARGET_INC].values
y_sev = df[TARGET_SEV].values / 100.0  # scale to 0-1 for Beta regression

# Grouping (County) and optional index
groups = df[GROUP_COL].values
indices = df.index.values

# -----------------------
# 2) Train-test split (stratified by incidence)
# -----------------------
X_train, X_test, y_train, y_test, g_train, g_test, idx_train, idx_test = train_test_split(
    X, y_inc, groups, indices, test_size=0.2, stratify=y_inc, random_state=42
)

# For severity we will align later by index if modeling conditional on incidence

# Save split samples counts
with open(os.path.join(MODELS_DIR, "split_info.json"), "w") as f:
    json.dump({
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "incidence_rate_train": float(y_train.mean()),
        "incidence_rate_test": float(y_test.mean())
    }, f, indent=2)

# -----------------------
# 3) Baseline supervised models
# -----------------------
# Helper function: evaluate classifier & plot ROC/PR/Calibration
def evaluate_classifier(model, X_t, y_t, name_prefix):
    y_prob = model.predict_proba(X_t)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc_roc = roc_auc_score(y_t, y_prob)
    precision, recall, _ = precision_recall_curve(y_t, y_prob)
    auc_pr = auc(recall, precision)
    brier = brier_score_loss(y_t, y_prob)
    acc = accuracy_score(y_t, y_pred)
    print(f"[{name_prefix}] AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f} | Brier: {brier:.4f} | Acc: {acc:.4f}")
    # ROC
    fpr, tpr, _ = roc_curve(y_t, y_prob)
    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name_prefix}")
    plt.legend()
    save_fig(fig, f"{name_prefix}_roc.png")
    # PR
    fig = plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {name_prefix}")
    save_fig(fig, f"{name_prefix}_pr.png")
    return {"auc_roc":auc_roc, "auc_pr":auc_pr, "brier":brier, "acc":acc, "y_prob":y_prob}

# 3.1 Logistic regression (regularized)
clf_log = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
clf_log.fit(X_train, y_train)
eval_log = evaluate_classifier(clf_log, X_test, y_test, "Logistic")

# Save logistic model
joblib.dump(clf_log, os.path.join(MODELS_DIR, "logistic.joblib"))

# 3.2 Random Forest
clf_rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced")
clf_rf.fit(X_train, y_train)
eval_rf = evaluate_classifier(clf_rf, X_test, y_test, "RandomForest")
joblib.dump(clf_rf, os.path.join(MODELS_DIR, "rf.joblib"))

# 3.3 XGBoost
clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=300, random_state=42)
clf_xgb.fit(X_train, y_train)
eval_xgb = evaluate_classifier(clf_xgb, X_test, y_test, "XGBoost")
joblib.dump(clf_xgb, os.path.join(MODELS_DIR, "xgb.joblib"))
#-------------------------------------

# -----------------------
# 3.4 Additional Models: LightGBM, CatBoost, ANN, SVM, Naive Bayes
# -----------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

# Helper plotting functions
def plot_calibration_curves(models_dict, X_test, y_test, filename):
    fig = plt.figure(figsize=(7,6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]
        else:  # e.g., SVM decision_function
            y_prob = model.decision_function(X_test)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration curves")
    plt.legend()
    save_fig(fig, filename)

def plot_learning_curves(models_dict, X, y, filename_prefix):
    for name, model in models_dict.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5,
            train_sizes=np.linspace(0.1,1.0,5),
            scoring="roc_auc", n_jobs=-1
        )
        fig = plt.figure(figsize=(7,5))
        plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
        plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV")
        plt.xlabel("Training examples")
        plt.ylabel("AUC")
        plt.title(f"Learning curve - {name}")
        plt.legend()
        save_fig(fig, f"{filename_prefix}_{name}_learning_curve.png")

# ---- Train additional models ----
# LightGBM
clf_lgb = lgb.LGBMClassifier(n_estimators=300, random_state=42, class_weight="balanced")
clf_lgb.fit(X_train, y_train)
eval_lgb = evaluate_classifier(clf_lgb, X_test, y_test, "LightGBM")
joblib.dump(clf_lgb, os.path.join(MODELS_DIR, "lgb.joblib"))

# CatBoost
clf_cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=42,
                             verbose=0, class_weights=[1, (y_train==0).sum()/(y_train==1).sum()])
clf_cat.fit(X_train, y_train)
eval_cat = evaluate_classifier(clf_cat, X_test, y_test, "CatBoost")
joblib.dump(clf_cat, os.path.join(MODELS_DIR, "cat.joblib"))

# ANN (MLPClassifier)
clf_ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                        max_iter=300, random_state=42)
clf_ann.fit(X_train, y_train)
eval_ann = evaluate_classifier(clf_ann, X_test, y_test, "ANN")
joblib.dump(clf_ann, os.path.join(MODELS_DIR, "ann.joblib"))

# SVM
clf_svm = SVC(kernel="rbf", probability=True, random_state=42)
clf_svm.fit(X_train, y_train)
eval_svm = evaluate_classifier(clf_svm, X_test, y_test, "SVM")
joblib.dump(clf_svm, os.path.join(MODELS_DIR, "svm.joblib"))

# Naive Bayes
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
eval_nb = evaluate_classifier(clf_nb, X_test, y_test, "NaiveBayes")
joblib.dump(clf_nb, os.path.join(MODELS_DIR, "nb.joblib"))

# -----------------------
# 3.5 Combined Plots
# -----------------------
models = {
    "Logistic": clf_log,
    "RandomForest": clf_rf,
    "XGBoost": clf_xgb,
    "LightGBM": clf_lgb,
    "CatBoost": clf_cat,
    "ANN": clf_ann,
    "SVM": clf_svm,
    "NaiveBayes": clf_nb
}

results = {}
all_roc, all_pr = [], []

for name, model in models.items():
    metrics = evaluate_classifier(model, X_test, y_test, name)
    results[name] = metrics
    y_prob = metrics["y_prob"]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    all_roc.append((name, fpr, tpr, metrics["auc_roc"]))
    all_pr.append((name, recall, precision, metrics["auc_pr"]))

# Combined ROC
fig = plt.figure(figsize=(7,6))
for name, fpr, tpr, auc_val in all_roc:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves - All models")
plt.legend()
save_fig(fig, "all_models_roc.png")

# Combined PR
fig = plt.figure(figsize=(7,6))
for name, recall, precision, auc_pr in all_pr:
    plt.plot(recall, precision, label=f"{name} (AUC-PR={auc_pr:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves - All models")
plt.legend()
save_fig(fig, "all_models_pr.png")

# Calibration Curves
plot_calibration_curves(models, X_test, y_test, "all_models_calibration.png")

# Learning Curves
plot_learning_curves(models, X_train, y_train, "all_models")

# -------------------------------
# Scalability Metrics for All Models
# -------------------------------
# --------------------------------------
# Scalability Metrics for All Models
# --------------------------------------
import time
import joblib
import os
import psutil
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

# Try CatBoost if installed
try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False


# ==============================
# Define scalability function
# ==============================
def evaluate_scalability(model, X_train, y_train, X_test, n_inference=500, model_name="model"):
    metrics = {}

    # ---- Training Time ----
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    metrics["train_time_sec"] = round(end_time - start_time, 4)

    # ---- Model Size ----
    tmp_file = f"{model_name}.pkl"
    joblib.dump(model, tmp_file)
    metrics["model_size_MB"] = round(os.path.getsize(tmp_file) / (1024 * 1024), 4)
    os.remove(tmp_file)

    # ---- Inference Latency ----
    start_time = time.time()
    for _ in range(n_inference):
        idx = np.random.randint(0, len(X_test))
        _ = model.predict(X_test[idx:idx+1])
    end_time = time.time()
    metrics["inference_latency_ms"] = round(((end_time - start_time) / n_inference) * 1000, 4)

    # ---- RAM / Memory Usage ----
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    metrics["ram_usage_MB"] = round(mem_info.rss / (1024 * 1024), 4)

    return metrics, model


# ==============================
# Define models
# ==============================
models_dict = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "ANN": MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", max_iter=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "NaiveBayes": GaussianNB()
}
if catboost_available:
    models_dict["CatBoost"] = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)


# ==============================
# Run scalability evaluation
# ==============================
results = []

for name, model in models_dict.items():
    print(f"Evaluating {name} ...")
    metrics, trained_model = evaluate_scalability(model, X_train, y_train, X_test, model_name=name)
    metrics["model"] = name
    results.append(metrics)

# Convert to DataFrame
scalability_df = pd.DataFrame(results)
scalability_df = scalability_df[["model", "train_time_sec", "model_size_MB", "inference_latency_ms", "ram_usage_MB"]]

# Print results nicely
print("\n=== Scalability Metrics (All Models) ===")
print(scalability_df.to_string(index=False))

# Save results
scalability_df.to_csv("scalability_metrics_all_models.csv", index=False)
print("✅ Scalability metrics saved as scalability_metrics_all_models.csv")
