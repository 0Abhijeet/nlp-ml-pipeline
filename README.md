# SMS Spam Classifier

A text classification pipeline that detects spam SMS messages using TF-IDF features and Logistic Regression (scikit-learn).

## Overview

The dataset is imbalanced (far more legitimate "ham" messages than spam), so the project uses a stratified train/test split, class-weighted training, and precision/recall/F1 rather than accuracy alone.

## Pipeline

1. **Preprocess** (`src/preprocess.py`): lowercase text, strip non-alphabetic characters
2. **Vectorize**: TF-IDF (top 3000 features, English stop words removed)
3. **Split**: 80/20 stratified train/test split (`random_state=42`)
4. **Train** (`src/train.py`): Logistic Regression with `class_weight="balanced"`
5. **Evaluate** (`src/evaluate.py`): classification report and confusion matrix

## Results

Spam class on the held-out test set:

| Metric    | Score |
|-----------|-------|
| Precision | 0.91  |
| Recall    | 0.91  |
| F1-score  | 0.91  |

## Project structure

```
├── data/raw/sms_spam.tsv    # dataset (label \t text)
├── notebooks/exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

## Getting started

```bash
git clone <your-repo-url>
cd nlp-ml-pipeline
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python src/train.py          # trains and saves spam_classifier.pkl
python src/evaluate.py       # prints metrics on the test split
```

Run the scripts from the repository root.

## Dataset

<!-- Add the source (e.g. SMS Spam Collection, UCI) and its license. -->

## Limitations and next steps

- Preprocessing removes digits and symbols (`£`, `!`), which are useful spam signals
- The vectorizer is fit before the split, which can leak test data into training
- No inference script yet for classifying new messages
- Planned: scikit-learn `Pipeline`, cross-validation, Naive Bayes baseline, error analysis
