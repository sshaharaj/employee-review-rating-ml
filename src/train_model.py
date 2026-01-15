# -*- coding: utf-8 -*-
"""Employee Review Rating Prediction (Glassdoor)

Supervised machine learning pipeline to predict employee review ratings (1-5)
from Glassdoor review text and available structured features.

Outputs:
- Test-set predictions (CSV)
- Diagnostic plots (word clouds, feature importance, distribution shift checks,
  confusion matrix, and error analysis)
- A short "actionable insights" analysis with an explicit correlation vs causation caveat.

Notes:
- The raw dataset is intentionally NOT included in this repository.
- Set the file paths in the CONFIG section before running.
- Model logic is intentionally unchanged; edits are limited to documentation,
  portability (local/non-notebook), and non-essential metadata/prints.

"""

# ============================================================================
# IMPORTS & GPU CHECK
# ============================================================================

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if VERBOSE:
    print(f"> Device in use: {device}")

if device.type != "cuda" and VERBOSE:
    print("Warning: GPU not detected; training may be slow on CPU.")
    raise RuntimeError("This script expects a GPU.")

if device.type == "cuda" and VERBOSE:
    print(f"> GPU name: {torch.cuda.get_device_name(0)}")
    print(f"> GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import os
import time
import csv
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm.auto import tqdm

if VERBOSE:
    print("\n[Setup finished]\n")

# ============================================================================
# USER CONFIGURATION (EDIT THESE)
# ============================================================================

# ---- Paths: fill these in manually before running (dataset not included) ----
LARGE_TRAIN_PATH = ""  # TODO: set path to large training CSV
SMALL_TRAIN_PATH = ""  # TODO: set path to small training CSV
TEST_PATH        = ""  # TODO: set path to test CSV

# Output directory (folder for CSV + probabilities)
OUTPUT_DIR = "reports/outputs"  # output folder for artifacts

# Directory for plots (PNG files)
OUTPUT_PLOT_DIR = "reports/figures"  # output folder for plots

# ---- Runtime flag ----
VERBOSE = True  # set False to reduce console output

def _require_path(p: str, name: str) -> None:
    if not isinstance(p, str) or not p.strip():
        raise ValueError(f"Missing required path: {name}. Set it in CONFIG before running.")

_require_path(LARGE_TRAIN_PATH, "LARGE_TRAIN_PATH")
_require_path(SMALL_TRAIN_PATH, "SMALL_TRAIN_PATH")
_require_path(TEST_PATH, "TEST_PATH")


# Output file paths (files inside OUTPUT_DIR)
PREDICTION_CSV_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
PROBA_NPY_PATH      = os.path.join(OUTPUT_DIR, "distilbert_test_probabilities.npy")

# ---- Metadata (optional) ----
MODEL_LABEL = "DistilBERT-base-uncased"

# ---- Model & training hyperparameters ----
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH  = 256
BATCH_SIZE      = 64
NUM_EPOCHS      = 4
LR              = 2e-5
WARMUP_FRAC     = 0.10
RANDOM_SEED     = 424
TRAIN_SAMPLE    = None  # e.g., 300000 for debugging, or None for full data

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

overall_start = time.time()

# Make sure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 70)
print("1) Loading datasets")
print("=" * 70)

full_large = pd.read_csv(LARGE_TRAIN_PATH)
full_small = pd.read_csv(SMALL_TRAIN_PATH)
test_frame = pd.read_csv(TEST_PATH)

# Use columns common to both train files
common_columns = full_large.columns.intersection(full_small.columns)
merged_train = pd.concat(
    [full_large[common_columns], full_small[common_columns]],
    ignore_index=True
)

print(f"- Combined training size: {len(merged_train):,} rows")
print(f"- Test set size:         {len(test_frame):,} rows")

# ============================================================================
# TEXT CONSTRUCTION
# ============================================================================

print("\n" + "=" * 70)
print("2) Building text input for the model")
print("=" * 70)


def build_combined_text(frame: pd.DataFrame) -> pd.Series:
    """
    Creates a single text field from headline, pros, cons, job_title, and firm.
    """
    headline = frame["headline"].fillna("").astype(str)
    pros     = frame["pros"].fillna("").astype(str)
    cons     = frame["cons"].fillna("").astype(str)
    job      = frame["job_title"].fillna("").astype(str)
    firm     = frame["firm"].fillna("").astype(str)

    combined = (
        "Employer: " + firm + " | Role: " + job + " | " + headline +
        " [SEP] Pros: " + pros +
        " [SEP] Cons: " + cons
    )
    return combined


merged_train["text_input"] = build_combined_text(merged_train)
test_frame["text_input"]   = build_combined_text(test_frame)

# ratings are from 1 to 5; convert to 0..4 for PyTorch
merged_train["label"] = merged_train["rating"].astype(int) - 1

print("Example processed review snippet:")
print(merged_train["text_input"].iloc[0][:220] + "...")
print()

# Optional subsample for speed
if TRAIN_SAMPLE is not None and TRAIN_SAMPLE < len(merged_train):
    print(f"Subsampling training data down to {TRAIN_SAMPLE:,} rows...")
    merged_train = merged_train.sample(
        n=TRAIN_SAMPLE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)
    print(f"New training size: {len(merged_train):,}")

# Train-validation split
train_df, val_df = train_test_split(
    merged_train,
    test_size=0.10,
    stratify=merged_train["label"],
    random_state=RANDOM_SEED
)

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")


# ============================================================================
# DATASET DEFINITION
# ============================================================================
class GlassdoorReviewDataset(Dataset):
    def __init__(self, text_series, label_series=None, tokenizer=None, max_len=256):
        self.texts = text_series.reset_index(drop=True)
        self.labels = None
        if label_series is not None:
            self.labels = label_series.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_sample = str(self.texts.iloc[index])

        encodings = self.tokenizer(
            text_sample,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            label_val = int(self.labels.iloc[index])
            item["labels"] = torch.tensor(label_val, dtype=torch.long)

        return item


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3) Initializing DistilBERT classifier")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=5,
    problem_type="single_label_classification",
)

model.to(device)

print(f"- Base model: {BASE_MODEL_NAME}")
total_params = sum(p.numel() for p in model.parameters())
print(f"- Total parameters: {total_params:,}")

# Datasets and loaders
train_dataset = GlassdoorReviewDataset(
    text_series=train_df["text_input"],
    label_series=train_df["label"],
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LENGTH,
)
val_dataset = GlassdoorReviewDataset(
    text_series=val_df["text_input"],
    label_series=val_df["label"],
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LENGTH,
)
test_dataset = GlassdoorReviewDataset(
    text_series=test_frame["text_input"],
    label_series=None,
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LENGTH,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================

print("\n" + "=" * 70)
print("4) Preparing optimizer & scheduler")
print("=" * 70)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_train_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_train_steps * WARMUP_FRAC)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_train_steps,
)

print(f"- Total steps:  {total_train_steps:,}")
print(f"- Warmup steps: {warmup_steps:,}")
print(f"- Epochs:       {NUM_EPOCHS}")

# ============================================================================
# TRAINING & EVALUATION UTILITIES
# ============================================================================


def run_training_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        optimizer.zero_grad()

        batch_input_ids = batch["input_ids"].to(device)
        batch_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_mask,
            labels=batch_labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)

        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    avg_loss = running_loss / len(loader)
    epoch_acc = correct / max(total, 1)
    return avg_loss, epoch_acc


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch_input_ids = batch["input_ids"].to(device)
            batch_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
                labels=batch_labels,
            )

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(labels_all, preds_all)
    return avg_loss, accuracy, preds_all, labels_all


# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "=" * 70)
print("5) Starting fine-tuning loop")
print("=" * 70)

best_val_accuracy = 0.0
best_state_dict = None
training_log = []

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} --------------------------------------")
    epoch_start_time = time.time()

    train_loss, train_acc = run_training_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    val_loss, val_acc, val_preds, val_labels = evaluate_model(
        model, val_loader, device
    )

    epoch_duration = (time.time() - epoch_start_time) / 60.0

    print(f"Training loss:   {train_loss:.4f}, Training acc:   {train_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}")
    print(f"Epoch duration:  {epoch_duration:.1f} minutes")

    training_log.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
    )

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(">>> Validation accuracy improved; saving this model snapshot.")

    # Very simple early stopping heuristic
    if epoch >= 2 and val_acc < training_log[-2]["val_acc"] - 0.01:
        print("Validation accuracy dropped noticeably; stopping training early.")
        break

print("\n" + "=" * 70)
print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
print("=" * 70)

# Load best checkpoint
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
model.to(device)

# ============================================================================
# COMPUTE TRAINING ACCURACY ON FULL TRAIN DATA
# ============================================================================

print("\n" + "=" * 70)
print("6) Evaluating accuracy on full training data")
print("=" * 70)

full_train_dataset = GlassdoorReviewDataset(
    text_series=merged_train["text_input"],
    label_series=merged_train["label"],
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LENGTH,
)

full_train_loader = DataLoader(
    full_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

full_loss, full_acc, full_preds, full_labels = evaluate_model(model, full_train_loader, device)
train_accuracy_percent = full_acc * 100.0

print(f"Full training set accuracy: {train_accuracy_percent:.2f}%")

# ============================================================================
# PREDICT ON TEST SET
# ============================================================================

print("\n" + "=" * 70)
print("7) Generating predictions for competition test set")
print("=" * 70)

model.eval()
test_predictions = []
test_probabilities = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        batch_input_ids = batch["input_ids"].to(device)
        batch_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_mask,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        test_predictions.extend(preds.cpu().numpy())
        test_probabilities.extend(probs.cpu().numpy())

test_predictions = np.array(test_predictions) + 1  # back to 1..5
test_probabilities = np.array(test_probabilities)

print("\nDistribution of predicted ratings (1-5):")
print(pd.Series(test_predictions).value_counts().sort_index())
print()

# ============================================================================
# SAVE OUTPUTS (PREDICTIONS CSV + PROBABILITIES)
# ============================================================================

if VERBOSE:
    print("\n" + "=" * 70)
    print("8) Saving outputs")
    print("=" * 70)


prediction_rows = [[int(val)] for val in test_predictions]

with open(PREDICTION_CSV_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["predicted_rating"])
    writer.writerows(prediction_rows)

if VERBOSE:
    print(f"- Predictions CSV written to: {PREDICTION_CSV_PATH}")

np.save(PROBA_NPY_PATH, test_probabilities)
print(f"- Test set class probabilities saved to: {PROBA_NPY_PATH}")

# ============================================================================
# DIAGNOSTICS AND INSIGHTS (PLOTS SAVED AS PNG)
# ============================================================================
print("\n" + "=" * 70)
print("9) Diagnostics (word clouds, feature importance, errors, themes)")
print("=" * 70)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sns.set(style="whitegrid")

# Simple numeric features for train vs test plots
merged_train["text_len"] = merged_train["text_input"].str.len()
test_frame["text_len"]  = test_frame["text_input"].str.len()

for df in [merged_train, test_frame]:
    df["len_headline"] = df["headline"].fillna("").str.len()
    df["len_pros"]     = df["pros"].fillna("").str.len()
    df["len_cons"]     = df["cons"].fillna("").str.len()

# ------------------------------------------------------------------
# Word clouds for 5★ vs 1★ reviews
# ------------------------------------------------------------------
print("\n[Q1b] Word clouds (5★ vs 1★)")

text_5 = " ".join(merged_train.loc[merged_train["rating"] == 5, "text_input"])
text_1 = " ".join(merged_train.loc[merged_train["rating"] == 1, "text_input"])

wc5 = WordCloud(width=800, height=400, background_color="white").generate(text_5)
wc1 = WordCloud(width=800, height=400, background_color="white").generate(text_1)

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.imshow(wc5, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud – 5-Star Reviews")

plt.subplot(1, 2, 2)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud – 1-Star Reviews")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_wordclouds_5vs1.png"), dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# Feature importance (interpretable surrogate on text)
# ------------------------------------------------------------------
print("\n[Q1b] Feature importance (TF-IDF + Logistic Regression surrogate)")

X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(
    merged_train["text_input"],
    merged_train["rating"],
    test_size=0.10,
    random_state=RANDOM_SEED,
    stratify=merged_train["rating"]
)

tfidf = TfidfVectorizer(max_features=30000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf   = tfidf.transform(X_val_text)

logit = LogisticRegression(max_iter=300, n_jobs=-1, multi_class="multinomial")
logit.fit(X_train_tfidf, y_train_text)

val_preds_surrogate = logit.predict(X_val_tfidf)
val_acc_surrogate = accuracy_score(y_val_text, val_preds_surrogate)
print(f"Surrogate validation accuracy: {val_acc_surrogate:.4f}")

coef_matrix = logit.coef_
importance = np.abs(coef_matrix).sum(axis=0)
feature_names = np.array(tfidf.get_feature_names_out())

top_k = 20
top_idx = np.argsort(importance)[-top_k:]
top_terms = feature_names[top_idx]
top_vals  = importance[top_idx]

order = np.argsort(top_vals)
top_terms = top_terms[order]
top_vals  = top_vals[order]

plt.figure(figsize=(8, 6))
plt.barh(top_terms, top_vals)
plt.xlabel("Aggregate abs(coefficient)")
plt.ylabel("Token")
plt.title("Top text features (TF-IDF + Logistic Regression)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# Feature distributions – train vs test
# ------------------------------------------------------------------
print("\n[Q1b] Feature distributions – train vs test")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.kdeplot(data=merged_train, x="year_review", label="Train", fill=True)
sns.kdeplot(data=test_frame, x="year_review", label="Test", fill=True, alpha=0.4)
plt.title("Year of Review – Train vs Test")
plt.legend()

plt.subplot(2, 2, 2)
sns.kdeplot(data=merged_train, x="text_len", label="Train", fill=True)
sns.kdeplot(data=test_frame, x="text_len", label="Test", fill=True, alpha=0.4)
plt.title("Text Length – Train vs Test")
plt.legend()

plt.subplot(2, 2, 3)
sns.kdeplot(data=merged_train, x="len_pros", label="Train", fill=True)
sns.kdeplot(data=test_frame, x="len_pros", label="Test", fill=True, alpha=0.4)
plt.title("Pros Length – Train vs Test")
plt.legend()

plt.subplot(2, 2, 4)
sns.kdeplot(data=merged_train, x="len_cons", label="Train", fill=True)
sns.kdeplot(data=test_frame, x="len_cons", label="Test", fill=True, alpha=0.4)
plt.title("Cons Length – Train vs Test")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_feature_distributions.png"), dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# Confusion matrix (full training data, DistilBERT)
# ------------------------------------------------------------------
print("\n[Q1b] Confusion matrix – DistilBERT on full training data")

true_train = np.array(full_labels) + 1
pred_train = np.array(full_preds) + 1

cm = confusion_matrix(true_train, pred_train, labels=[1, 2, 3, 4, 5])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – DistilBERT (Train)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_confusion_matrix_train.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Confusion matrix (rows = true rating, cols = predicted rating):")
print(cm)

# ------------------------------------------------------------------
# Error behavior across ratings (DistilBERT, train data)
# ------------------------------------------------------------------
print("\n[Q1b] Error distribution and bias by true rating – DistilBERT")

errors = true_train - pred_train

plt.figure(figsize=(6, 4))
plt.hist(errors, bins=np.arange(-4.5, 5.5, 1), edgecolor="black")
plt.xlabel("Prediction error (true rating – predicted rating)")
plt.ylabel("Count")
plt.title("Prediction Error Distribution (Train Data)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_error_distribution_train.png"), dpi=300, bbox_inches="tight")
plt.show()

bias_df = pd.DataFrame({"true": true_train, "pred": pred_train})
group_stats = bias_df.groupby("true")["pred"].agg(["mean", "count"]).reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=group_stats, x="true", y="mean")
plt.xlabel("True rating")
plt.ylabel("Average predicted rating")
plt.title("Average predicted rating by true rating (Train Data)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1b_rating_bias_train.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Average predicted rating by true rating:")
print(group_stats)

# ------------------------------------------------------------------
# Theme-based insights and text length vs rating
# ------------------------------------------------------------------
print("\nTheme-based rating differences + text length vs rating")

theme_words = ["toxic", "management", "manager", "pay", "benefits", "overtime", "growth", "work-life"]

for w in theme_words:
    merged_train[f"theme_{w}"] = merged_train["text_input"].str.lower().str.contains(w).astype(int)

theme_stats = []
for w in theme_words:
    mask = merged_train[f"theme_{w}"] == 1
    n_with = mask.sum()
    if n_with < 100:
        continue
    mean_with = merged_train.loc[mask, "rating"].mean()
    mean_without = merged_train.loc[~mask, "rating"].mean()
    theme_stats.append({
        "theme": w,
        "n_with": int(n_with),
        "avg_rating_with": mean_with,
        "avg_rating_without": mean_without,
        "diff": mean_with - mean_without
    })

theme_df = pd.DataFrame(theme_stats).sort_values("diff", ascending=False)
print("\nTheme-based rating differences:")
print(theme_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=theme_df, x="theme", y="diff")
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Keyword / theme")
plt.ylabel("Δ avg rating (mentions – does not mention)")
plt.title("Theme association with overall rating")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "q1c_theme_effects.png"), dpi=300, bbox_inches="tight")
plt.show()

# Quintiles of text length
merged_train["len_quintile"] = pd.qcut(
    merged_train["text_len"], q=5, duplicates="drop"
)

# Average rating by quintile
len_stats = (
    merged_train.groupby("len_quintile")["rating"]
    .mean()
    .reset_index()
)

# Convert quintile intervals to strings for plotting
len_stats["len_quintile_str"] = len_stats["len_quintile"].astype(str)

plt.figure(figsize=(7, 4))
sns.lineplot(
    data=len_stats,
    x="len_quintile_str",
    y="rating",
    marker="o"
)
plt.xticks(rotation=45)
plt.xlabel("Review text length quintile")
plt.ylabel("Average rating")
plt.title("Average rating by review text length")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_PLOT_DIR, "q1c_text_length_vs_rating.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ============================================================================
# DONE
# ============================================================================
total_runtime = (time.time() - overall_start) / 60.0
print(f"\nTotal runtime: {total_runtime:.1f} minutes")
print("Prediction & diagnosis outputs generated.")
print(f"All plots saved under: {OUTPUT_PLOT_DIR}")
