# Employee Review Rating Prediction (Glassdoor)

## Overview
This project builds a supervised machine learning pipeline to predict
employee review ratings (1–5) using Glassdoor review text and structured
metadata.

The focus is on feature engineering, model evaluation, diagnostic analysis,
and interpretation under realistic computational constraints.

## Problem
- Task: Multi-class classification (ratings 1–5)
- Inputs: Review text and structured features
- Output: Predicted employee rating

## Approach
- Cleaned and preprocessed raw review data
- Engineered text-based and numerical features
- Trained a supervised classification model
- Evaluated performance using accuracy and diagnostic plots

## Evaluation & Diagnostics
The analysis includes:
- Word clouds comparing 1-star vs 5-star reviews
- Feature importance analysis
- Train vs test feature distribution checks
- Confusion matrix
- Prediction error distribution analysis

## Insights
Model outputs are used to identify themes and features associated with
low and high employee ratings. Results are predictive and not interpreted
as causal.

## Reproducibility
- Model architecture and hyperparameters are fixed
- Raw data is not included due to usage restrictions
- Code is organized for clarity and reproducibility

## Tools
Python, pandas, NumPy, scikit-learn, PyTorch, Transformers, NLP libraries
