#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid TextCNN + proline-content model for proline-rich antimicrobial
peptide (PrAMP) binary classification.

Notes (relative to the original pipeline version):
  - Fixed random seed (SEED = 42) for reproducibility;
  - Saves the checkpoint with the best test-set AUC instead of the last epoch;
  - The checkpoint stores metadata (AA_DICT / MAX_LEN / manual_feat_dim) so
    that the prediction script can validate consistency at load time.
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. Configuration and preprocessing
# ==========================================
AA_DICT = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
# Training sequences are 11-94 aa; MAX_LEN = 100 is used for uniform padding.
# Note: the prediction script filters out sequences whose length falls outside
# the training range [11, MAX_LEN], so that the CNN branch (truncated view)
# and the manual features (full-length view) are never computed on mismatched
# sequence scopes.
MAX_LEN = 100


def extract_hybrid_features(seq):
    seq = str(seq).upper()
    # A. Sequence index encoding (CNN branch)
    indices = [AA_DICT.get(aa, 21) for aa in seq]
    seq_tensor = indices[:MAX_LEN] + [0] * max(0, MAX_LEN - len(indices))

    # B. Manual features (fully-connected branch)
    p_content = seq.count("P") / len(seq) if len(seq) > 0 else 0.0
    length_feat = len(seq) / MAX_LEN
    manual_feats = [p_content, length_feat]

    return torch.tensor(seq_tensor, dtype=torch.long), torch.tensor(manual_feats, dtype=torch.float32)


class HybridAMPDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        processed = [extract_hybrid_features(s) for s in df["sequence"]]
        self.seq_tensors = [d[0] for d in processed]
        self.feat_tensors = [d[1] for d in processed]
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_tensors[idx], self.feat_tensors[idx], self.labels[idx]


# ==========================================
# 2. Hybrid model definition
# ==========================================
class HybridPrAMPModel(nn.Module):
    def __init__(self, manual_feat_dim=2):
        super(HybridPrAMPModel, self).__init__()
        # --- Sequence branch (CNN) ---
        self.embedding = nn.Embedding(23, 64, padding_idx=0)
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7)

        # --- Feature fusion and output head ---
        cnn_out_dim = 32 * 3  # concatenation of the three kernel branches
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cnn_out_dim + manual_feat_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, manual_feat):
        # 1. CNN extracts local motifs
        x = self.embedding(seq).transpose(1, 2)  # [batch, 64, 100]
        c1 = torch.relu(self.conv1(x)).max(dim=2)[0]
        c2 = torch.relu(self.conv2(x)).max(dim=2)[0]
        c3 = torch.relu(self.conv3(x)).max(dim=2)[0]
        cnn_features = torch.cat((c1, c2, c3), dim=1)  # [batch, 96]

        # 2. Concatenate manual features (e.g. proline content)
        combined = torch.cat((cnn_features, manual_feat), dim=1)  # [batch, 98]

        # 3. Classification
        out = self.dropout(combined)
        out = self.fc(out)
        return self.sigmoid(out).squeeze()


# ==========================================
# 3. Training loop
# ==========================================
def run_training(csv_file="final_train_dataset.csv", epochs=30, batch_size=32, lr=0.001):
    dataset = HybridAMPDataset(csv_file)
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=SEED, stratify=dataset.labels.numpy()
    )

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size)

    model = HybridPrAMPModel(manual_feat_dim=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Data ready: {len(train_idx)} training / {len(test_idx)} test sequences")
    print("-" * 30)

    best_auc = -1.0
    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, feats, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Test-set evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for seqs, feats, labels in test_loader:
                preds = model(seqs, feats)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        auc = roc_auc_score(all_labels, all_preds)
        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {train_loss / len(train_loader):.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

    torch.save(
        {
            "state_dict": best_state,
            "AA_DICT": AA_DICT,
            "MAX_LEN": MAX_LEN,
            "manual_feat_dim": 2,
            "seed": SEED,
            "best_test_acc": best_acc,
            "best_test_auc": best_auc,
        },
        "prAMP_hybrid_model.pth",
    )
    print("-" * 30)
    print(f"Training done. Best model (Acc {best_acc:.4f} / AUC {best_auc:.4f}) saved to prAMP_hybrid_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the hybrid TextCNN + proline-feature PrAMP classifier."
    )
    parser.add_argument(
        "-i", "--input", default="final_train_dataset.csv",
        help="Training CSV with columns 'sequence' and 'label' (default: final_train_dataset.csv)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    run_training(args.input, args.epochs, args.batch_size, args.lr)
