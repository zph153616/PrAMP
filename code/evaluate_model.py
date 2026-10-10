#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation of the hybrid PrAMP classifier.

Two protocols are run on the supplied training CSV:
  1. an 80/20 stratified hold-out split (seed 42) evaluated with the saved
     checkpoint, written to test_metrics.csv;
  2. a 10-fold cross-validation (30 epochs, lr 0.001, batch 32),
     written to cv_fold_metrics.csv and cv_metrics_summary.csv.

Reported metrics: TP/FP/TN/FN, Accuracy, Sensitivity, Specificity, Precision,
F1, MCC, AUROC, AUPRC and Balanced Accuracy.
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, roc_auc_score)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset

from train_hybrid_amp import HybridAMPDataset, HybridPrAMPModel, PHYSCHEM_FEATURES, SEED


def full_metrics(y_true, y_prob, threshold=0.5):
    y_pred = [1 if p > threshold else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
        auprc = float("nan")
    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        Accuracy=float(accuracy_score(y_true, y_pred)),
        Sensitivity=float(sens), Specificity=float(spec),
        Precision=float(precision_score(y_true, y_pred, zero_division=0)),
        F1=float(f1_score(y_true, y_pred, zero_division=0)),
        MCC=float(matthews_corrcoef(y_true, y_pred)),
        AUROC=float(auc), AUPRC=float(auprc),
        BalancedAccuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def evaluate_loader(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for seqs, feats, ys in loader:
            preds.extend(model(seqs, feats).tolist())
            labels.extend(ys.tolist())
    return labels, preds


def build_fold_model(dim):
    model = HybridPrAMPModel(manual_feat_dim=dim)
    model.apply(
        lambda layer: layer.reset_parameters()
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear, torch.nn.Embedding))
        else None
    )
    return model


def main(csv_file, model_file, k_folds, epochs, batch_size, lr):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = HybridAMPDataset(csv_file)

    # 1) 80/20 hold-out evaluation with the saved checkpoint
    _, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=SEED, stratify=dataset.labels.numpy()
    )
    try:
        ckpt = torch.load(model_file, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(model_file, map_location="cpu", weights_only=False)
    model = HybridPrAMPModel(manual_feat_dim=ckpt["manual_feat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    y_true, y_prob = evaluate_loader(model, DataLoader(Subset(dataset, test_idx), batch_size=batch_size))
    test_m = full_metrics(y_true, y_prob)
    print("=== Hold-out test metrics (80/20, seed 42, best checkpoint) ===")
    for k, v in test_m.items():
        print(f"  {k}: {v}")

    # Re-seed so the CV matches a standalone cross-validation run.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 2) 10-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    dim = len(PHYSCHEM_FEATURES)
    rows = []
    for fold, (tr, te) in enumerate(kfold.split(dataset), 1):
        tr_loader = DataLoader(Subset(dataset, tr), batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(Subset(dataset, te), batch_size=batch_size)
        m = build_fold_model(dim)
        crit = torch.nn.BCELoss()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            m.train()
            for seqs, feats, ys in tr_loader:
                opt.zero_grad()
                loss = crit(m(seqs, feats), ys)
                loss.backward()
                opt.step()
        yt, yp = evaluate_loader(m, te_loader)
        fm = full_metrics(yt, yp)
        fm["Fold"] = fold
        rows.append(fm)
        print(f"Fold {fold:02d} | Acc {fm['Accuracy']:.4f} | SENS {fm['Sensitivity']:.4f} | "
              f"SPEC {fm['Specificity']:.4f} | F1 {fm['F1']:.4f} | MCC {fm['MCC']:.4f} | "
              f"AUROC {fm['AUROC']:.4f} | AUPRC {fm['AUPRC']:.4f}")

    df = pd.DataFrame(rows)
    summ = df.drop(columns=["Fold"]).agg(["mean", "std"]).T
    summ.columns = ["mean", "std"]
    print(f"=== {k_folds}-fold CV summary (mean +/- std) ===")
    for idx, r in summ.iterrows():
        print(f"  {idx}: {r['mean']:.4f} +/- {r['std']:.4f}")
    df.to_csv("cv_fold_metrics.csv", index=False)
    summ.to_csv("cv_metrics_summary.csv")
    pd.DataFrame([test_m]).to_csv("test_metrics.csv", index=False)
    print("Wrote: test_metrics.csv / cv_fold_metrics.csv / cv_metrics_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the hybrid PrAMP classifier (hold-out + 10-fold CV).")
    parser.add_argument("-i", "--input", default="final_train_dataset.csv", help="Training CSV (columns sequence,label)")
    parser.add_argument("-m", "--model", default="prAMP_hybrid_model.pth", help="Checkpoint produced by train_hybrid_amp.py")
    parser.add_argument("--k-folds", type=int, default=10, help="Number of CV folds (default: 10)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per fold (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    main(args.input, args.model, args.k_folds, args.epochs, args.batch_size, args.lr)
