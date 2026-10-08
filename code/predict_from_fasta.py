#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch prediction of proline-rich antimicrobial peptides (PrAMPs) from a
FASTA file using the trained hybrid model.

Notes (relative to the original pipeline version):
  - Reads metadata (AA_DICT / MAX_LEN) from the model file and validates it;
  - Sequences with length outside the training range [11, 100] are skipped
    and written to skipped_sequences.csv. This avoids the inconsistency of
    "CNN sees a truncated sequence while manual features are computed on the
    full-length sequence";
  - Output CSV columns are unchanged: seq_id, sequence, prediction,
    probability.
"""

import argparse
import os

import torch
import pandas as pd
from Bio import SeqIO

from train_hybrid_amp import HybridPrAMPModel, extract_hybrid_features, AA_DICT, MAX_LEN

# The model checkpoint lives at the repository root (one level above code/),
# so the path is resolved relative to this script and is independent of the
# current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(REPO_ROOT, "prAMP_hybrid_model.pth")
MIN_LEN = 11  # shortest sequence length in the training set


def load_model():
    # torch >= 2.6 defaults to weights_only=True. Checkpoints produced by older
    # versions may carry numpy scalars in their metadata, which that safe loader
    # rejects; fall back to the legacy behaviour for such files.
    try:
        checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    assert checkpoint["MAX_LEN"] == MAX_LEN, (
        f"MAX_LEN mismatch: model {checkpoint['MAX_LEN']} vs code {MAX_LEN}. "
        "Use the matching version of the training code."
    )
    assert checkpoint["AA_DICT"] == AA_DICT, (
        "AA_DICT mismatch between the model file and the code. "
        "Use the matching version of the training code."
    )
    model = HybridPrAMPModel(manual_feat_dim=checkpoint["manual_feat_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_amp(model, sequence):
    seq_tensor, feat_tensor = extract_hybrid_features(sequence)
    seq_tensor = seq_tensor.unsqueeze(0)  # add batch dimension
    feat_tensor = feat_tensor.unsqueeze(0)
    with torch.no_grad():
        probability = model(seq_tensor, feat_tensor).item()
    label = "Positive (Pr-AMP)" if probability > 0.5 else "Negative"
    return label, probability


def predict_fasta(fasta_file, output_csv="amp_predictions.csv"):
    model = load_model()
    results = []
    skipped = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
            reason = f"length {len(seq)} outside the training range [{MIN_LEN}, {MAX_LEN}]"
            print(f"Skipped ID: {seq_id} ({reason})")
            skipped.append({"seq_id": seq_id, "sequence": seq, "reason": reason})
            continue
        label, prob = predict_amp(model, seq)
        print(f"ID: {seq_id} | {label} | Score: {prob:.4f}")
        results.append({"seq_id": seq_id, "sequence": seq, "prediction": label, "probability": prob})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv("skipped_sequences.csv", index=False)
    print(f"\nPrediction finished: {len(results)} predicted, {len(skipped)} skipped.")
    print(
        f"Results saved to: {output_csv}"
        + (" (filtered sequences written to skipped_sequences.csv)" if skipped else "")
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict PrAMPs from a FASTA file using the trained hybrid model."
    )
    parser.add_argument(
        "-i", "--input", default="data/example_sequences.fasta",
        help="Input FASTA file (default: data/example_sequences.fasta)",
    )
    parser.add_argument(
        "-o", "--output", default="amp_predictions.csv",
        help="Output CSV file (default: amp_predictions.csv)",
    )
    args = parser.parse_args()
    predict_fasta(args.input, args.output)
