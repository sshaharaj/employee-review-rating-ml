## Data

This project uses employee review data derived from Glassdoor reviews, including
free-text review fields (e.g., pros, cons, headlines) and structured metadata
(e.g., firm, job title), along with an overall employee rating on a 1–5 scale.

The primary dataset used for training contains approximately 500,000 observations
and is provided as a CSV file (e.g., `424_F2025_Final_PC_large_train_v1.csv`).
An additional smaller training set and a held-out test set were also used in
development.

Due to data usage and redistribution restrictions, the raw datasets are not
included in this repository. To reproduce the results, users should obtain the
Glassdoor review datasets from the original source and place the required CSV
files in this directory, updating file paths in `src/train_model.py` as needed.
