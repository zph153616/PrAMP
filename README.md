# Prediction of Proline-rich Antimicrobial Peptides with a Hybrid TextCNN Model

# hybrid sequence + proline-feature deep learning model (PrAMP)

## Dependencies

The following packages are required to run the code:

- python==3.12
- torch>=2.0
- pandas>=2.0
- numpy>=1.24
- biopython>=1.81
- scikit-learn>=1.3

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
|   |-- train_hybrid_amp.py     # Model training (TextCNN + proline features)
|   `-- predict_from_fasta.py   # Batch prediction on a FASTA file
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
repository (131 KB). It stores the model weights together with the metadata
(`AA_DICT`, `MAX_LEN`, `manual_feat_dim`) required for consistent loading.

### Model architecture

- Sequence branch: amino acid index encoding (21-letter alphabet, zero-padded
  to length 100) -> 64-d embedding -> three parallel 1-D convolution branches
  (kernel sizes 3 / 5 / 7, 32 filters each, global max-pooling).
- Feature branch: two manual features - proline content and normalized
  sequence length.
- Both branches are concatenated and passed through dropout (0.5) and a
  fully-connected layer with a sigmoid output.

### Performance

Independent test set (`results/test_metrics.csv`):

| Accuracy | Sensitivity | Specificity | Precision | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| 0.9648 | 0.9668 | 0.9628 | 0.9628 | 0.9648 | 0.9296 | **0.9891** | 0.9851 |

10-fold cross-validation (`results/cv_metrics_summary.csv`, mean +/- std):

| Accuracy | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|
| 0.9668+/-0.0131 | 0.9673+/-0.0127 | 0.9340+/-0.0262 | **0.9924+/-0.0066** | 0.9891+/-0.0137 |

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

