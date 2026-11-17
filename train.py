#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - Standalone Model Training & Serialization Script
Course: DataTalksClub Machine Learning Zoomcamp (Midterm Project)
Author: Mayur Chawla

This script:
1. Loads the Bank Marketing dataset (bank-additional-full.csv).
2. Cleans and standardizes column names.
3. Excludes target leakage variable ('duration') for realistic prospective scoring.
4. Splits the data into full train (80%) and test (20%) with stratification.
5. Encodes categorical variables and numerical features using DictVectorizer.
6. Trains the tuned winning model (XGBoost with eta=0.05, max_depth=5).
7. Evaluates the final model on the unseen test set.
8. Serializes the DictVectorizer and trained model into 'model.bin'.
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb


def load_and_preprocess_data(data_path: str):
    """Load dataset and prepare features and target."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[1/5] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path, sep=';')

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace('.', '_')
    print(f"      Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Domain rule: Exclude duration to avoid target leakage in telemarketing lead scoring
    features = [c for c in df.columns if c not in ['y', 'duration']]
    print(f"      Selected {len(features)} operational features (excluded 'duration' to prevent leakage).")

    X = df[features]
    y = (df['y'] == 'yes').astype(int)

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Vectorize features and train tuned XGBoost model."""
    print("[2/5] Vectorizing features using DictVectorizer...")
    dv = DictVectorizer(sparse=False)
    X_train_vec = dv.fit_transform(X_train.to_dict(orient='records'))

    # Sanitize feature names for XGBoost DMatrix
    feature_names = [
        f.replace('[', '_').replace(']', '_').replace('<', '_').replace('=', '_')
        for f in dv.get_feature_names_out().tolist()
    ]

    print("[3/5] Training tuned XGBoost model on full training set...")
    dfulltrain = xgb.DMatrix(X_train_vec, label=y_train, feature_names=feature_names)

    params = {
        'eta': 0.05,
        'max_depth': 5,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42
    }

    model = xgb.train(params, dfulltrain, num_boost_round=100)
    print("      Model training complete.")

    return dv, model, feature_names


def evaluate_model(dv, model, feature_names, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate trained model on unseen test set."""
    print("[4/5] Evaluating model on unseen test set (20%)...")
    X_test_vec = dv.transform(X_test.to_dict(orient='records'))
    dtest = xgb.DMatrix(X_test_vec, label=y_test, feature_names=feature_names)

    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print("=" * 50)
    print("           FINAL TEST EVALUATION METRICS           ")
    print("=" * 50)
    print(f"  Test ROC-AUC   : {auc:.5f}")
    print(f"  Test Accuracy  : {acc:.5f} ({acc*100:.2f}%)")
    print(f"  Test Precision : {prec:.5f}")
    print(f"  Test Recall    : {rec:.5f}")
    print(f"  Test F1-Score  : {f1:.5f}")
    print("=" * 50)


def save_model(dv, model, output_file: str):
    """Serialize vectorizer and trained model."""
    print(f"[5/5] Saving model and vectorizer to '{output_file}'...")
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"      Successfully exported to '{output_file}'! Ready for deployment.")


def main():
    parser = argparse.ArgumentParser(description="Train and save Bank Marketing Term Deposit prediction model.")
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/bank-additional-full.csv',
        help="Path to CSV dataset (default: data/bank-additional-full.csv)"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='model.bin',
        help="Path to output serialized model file (default: model.bin)"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    output_file = os.path.join(base_dir, args.output_file) if not os.path.isabs(args.output_file) else args.output_file

    X, y = load_and_preprocess_data(data_path)

    # 80% Full Train, 20% Test split
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dv, model, feature_names = train_model(X_full_train, y_full_train)
    evaluate_model(dv, model, feature_names, X_test, y_test)
    save_model(dv, model, output_file)


if __name__ == '__main__':
    main()
