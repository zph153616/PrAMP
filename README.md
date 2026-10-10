# Prediction of Proline-rich Antimicrobial Peptides with a Hybrid TextCNN Model

# hybrid sequence + physicochemical-feature deep learning model (PrAMP)

## Dependencies

The following packages are required to run the code:

- python==3.12
- torch>=2.0
- pandas>=2.0
- numpy>=1.24
- biopython>=1.81
- scikit-learn>=1.3
- peptides==0.5.0

To install the dependencies, run:

```
conda env create -f environment.yml
```

or, alternatively:

```
pip install -r requirements.txt
```

## proAMP

```
proAMP/
|-- code
|   |-- train_hybrid_amp.py     # Model training (TextCNN + physicochemical features)
|   |-- predict_from_fasta.py   # Batch prediction on a FASTA file
|   `-- evaluate_model.py       # Independent test-set and 10-fold CV metrics
|-- data
|   `-- example_sequences.fasta # Example input sequences
|-- results
|   |-- test_metrics.csv        # Independent test-set metrics
|   |-- cv_fold_metrics.csv     # Per-fold metrics of the 10-fold CV
|   `-- cv_metrics_summary.csv  # 10-fold CV summary (mean +/- std)
|-- prAMP_hybrid_model.pth      # Trained model checkpoint (best test AUC)
|-- environment.yml             # Conda environment
|-- requirements.txt            # pip requirements
`-- README.md                   # This README file
```

## Trained model

The trained checkpoint `prAMP_hybrid_model.pth` is included in this
repository (approximately 132 KB). It stores the model weights together with the
metadata (`AA_DICT`, `MAX_LEN`, `manual_feat_dim`, `physchem_features`) required
for consistent loading.

### Model architecture

- Sequence branch: amino acid index encoding (21-letter alphabet, zero-padded
  to length 100) -> 64-d embedding -> three parallel 1-D convolution branches
  (kernel sizes 3 / 5 / 7, 32 filters each, global max-pooling).
- Feature branch: eight physicochemical descriptors (normalized sequence length,
  proline content, hydrophobicity, net charge, isoelectric point, aliphatic
  index, Boman index and hydrophobic moment), passed through a linear layer
  (8 -> 16) with ReLU.
- Both branches are concatenated (96 + 16 = 112) and passed through dropout
  (0.5) and a fully-connected layer with a sigmoid output.

### Performance

Independent test set (`results/test_metrics.csv`):

| Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| 0.9738 | 0.9720 | 0.9755 | 0.9754 | 0.9737 | 0.9476 | **0.9965** | 0.9962 |

10-fold cross-validation (`results/cv_metrics_summary.csv`, mean +/- std):

| Accuracy | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|
| 0.9608+/-0.0112 | 0.9610+/-0.0109 | 0.9221+/-0.0221 | **0.9895+/-0.0072** | 0.9856+/-0.0145 |

## Training data

The positive set consists of proline-rich peptides (proline content > 15%)
retrieved from three antimicrobial-peptide databases: APD (2026-10 export),
DBAASP (2026-10 export) and DRAMP (2026-09 release). After concatenating the
three databases the sequences are made unique and clustered with `cd-hit`
(100% identity, then 90% identity within the proline-rich subset). The negative
set is drawn from UniProt release 2026_03 (cytoplasmic proteins of 11-94 aa and
non-fragment short proteins of 11-40 aa), filtered against antimicrobial-related
keywords and the PRPRP motif, and sampled to match the positive-set length
distribution. The final balanced data set contains 1,430 positive and 1,430
negative sequences (11-94 aa).

## Usage

Run the commands from the repository root.

### Prediction

1. Put your peptide sequences in a FASTA file (sequences must be 11-100 aa;
   sequences outside this range are skipped and written to
   `skipped_sequences.csv`).
2. Run `code/predict_from_fasta.py`:

```
python code/predict_from_fasta.py -i data/example_sequences.fasta -o amp_predictions.csv
```

3. The output CSV contains the columns `seq_id`, `sequence`, `prediction`
   (`Positive (Pr-AMP)` / `Negative`, threshold 0.5) and `probability`.

### Training

1. Prepare a training CSV with two columns: `sequence` (peptide string) and
   `label` (1 = proline-rich AMP, 0 = negative).
2. Run `code/train_hybrid_amp.py`:

```
python code/train_hybrid_amp.py -i final_train_dataset.csv --epochs 30 --batch-size 32 --lr 0.001
```

3. An 80/20 stratified train/test split is applied (seed 42), and the
   checkpoint with the best test AUC is saved to `prAMP_hybrid_model.pth`.

### Evaluation

Run `code/evaluate_model.py` to recompute the independent test-set metrics and
the 10-fold cross-validation (writes `test_metrics.csv`, `cv_fold_metrics.csv`
and `cv_metrics_summary.csv`):

```
python code/evaluate_model.py -i final_train_dataset.csv -m prAMP_hybrid_model.pth
```
