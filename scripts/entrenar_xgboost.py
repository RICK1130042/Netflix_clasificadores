"""Baseline XGBoost con train/test estratificado (80/20) y transformaciones sin fuga.

Uso Colab (después de clonar repositorio):
  !pip install -r requirements.txt
  !python scripts/entrenar_xgboost.py --input data/prepared/netflix_limpio.csv

Este baseline no sustituye las decisiones del notebook original: al comparar
clasificadores, usar exactamente la misma partición y columnas.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Baseline provisional basado en la selección conversada en el proyecto.
# Revisar contra la lista final del equipo y justificar exclusiones/inclusiones.
FEATURES_BASE = ["age", "subscription_type", "watch_hours", "last_login_days",
                 "region", "device", "monthly_fee", "payment_method", "number_of_profiles"]


def entrenar(ruta: Path, salida: Path):
    df = pd.read_csv(ruta)
    faltantes = set(FEATURES_BASE + ["churned"]) - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {sorted(faltantes)}")
    if not df["churned"].dropna().isin([0, 1]).all() or df["churned"].isna().any():
        raise ValueError("Target inválido")
    X, y = df[FEATURES_BASE], df["churned"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    nums = X.select_dtypes(include="number").columns.tolist()
    cats = [c for c in FEATURES_BASE if c not in nums]
    pre = ColumnTransformer([
        ("numericas", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums),
        ("categoricas", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    modelo = Pipeline([("preprocesador", pre), ("xgboost", XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=2))])
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    salida.mkdir(parents=True, exist_ok=True)
    metricas = {
        "modelo": "XGBoost baseline", "random_state": 42,
        "train_n": int(len(y_train)), "test_n": int(len(y_test)),
        "features": FEATURES_BASE,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_churn": float(precision_score(y_test, pred, zero_division=0)),
        "recall_churn": float(recall_score(y_test, pred, zero_division=0)),
        "f1_churn": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "matriz_confusion": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }
    (salida / "metricas_xgboost.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"y_real": y_test.to_numpy(), "y_pred": np.asarray(pred),
                  "prob_churn": proba}, index=y_test.index).sort_index().to_csv(
        salida / "predicciones_xgboost_test.csv", index_label="index_original")
    joblib.dump(modelo, salida / "xgboost_pipeline.joblib")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/prepared/netflix_limpio.csv"))
    p.add_argument("--output", type=Path, default=Path("outputs/xgboost"))
    a = p.parse_args()
    entrenar(a.input, a.output)
