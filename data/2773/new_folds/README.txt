2773 revised split layout
=========================

warm/
  Fresh random warm-start five-fold test partition, split seed 42.
  Each run is approximately 70:10:20 train/validation/test.
  Files are train_runN.csv, val_runN.csv, and test_runN.csv.
  The legacy unified_folds.pkl is not used by the revised split script.

drug-cold/
  Current drug cold-start benchmark.
  Drugs are grouped by RDKit canonical isomeric SMILES.
  Five folds, approximately 70:10:20 train/validation/test, split seed 42.
  Files are train_runN.csv, val_runN.csv, and test_runN.csv.

target-cold/
  Current target cold-start benchmark.
  Proteins are clustered at 30% sequence identity and 80% coverage with
  MMseqs2 cov-mode 0. The largest cluster is pinned to train and the remaining
  clusters are assigned to one fixed train/validation/test split using the
  KinetX protein-cold proportions.
  Files are train.csv, val.csv, and test.csv.
  Until make_2773_target_cold_fixed_split.py is run with MMseqs2 (or the
  matching clusters.tsv), any train_runN.csv/test_runN.csv files in this
  directory are the legacy exact-FASTA folds and must not be used as the
  revised similarity-controlled split.
