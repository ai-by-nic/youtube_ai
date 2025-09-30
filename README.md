# Decoding YouTube AI: Which Title Signals Drive Views? (EDA + Baseline Model)

**Author:** Nic (ai-by-nic) — UC Berkeley ML/AI Capstone

## Executive summary

Across 23,002 videos from 30 AI/data channels (2008-02-29 to 2024-06-21), titles that advertise _complete/beginner-friendly value_ ("full course", "for beginners", year tags like "2024") and AI terms ("ChatGPT", "AI") correlate with higher normalized views/day. A regularized linear baseline using TF-IDF title features + simple controls explains ~47% of variance on holdout and reduces error ~27–30% vs a mean baseline. (Details in Results.)

## Rationale

I'm launching an AI-education YouTube channel. Understanding which title patterns and simple timing cues move normalized views/day helps me (and similar creators) craft clearer, more findable titles, then iterate with data instead of vibes.

## Research Question

Which title and lightweight timing/channel signals are associated with higher normalized views/day for AI/ML videos on YouTube?

## Data Sources

- Aggregated CSV of AI/data-focused channels (N=23,002 videos, 30 channels). Fields include video title, channel, publish date, views, and other standard metadata; plus engineered features created in this notebook.
- Time window: 2008-02-29–2024-06-21 (as_of = 2024-06-21).

## Methodology

- Parse publish dates; compute `days_since_publish` and `views_per_day = Views / (days_since_publish+ε)`. Model target is `log1p(views_per_day)`.
- Feature set:
  - Text: TF-IDF on `Title` (unigrams/bigrams).
  - Categorical: channel (one-hot).
  - Simple numerics: title length, punctuation flags, digits, etc.
- Split: train/holdout with leakage-safe handling (date-aware or last-k% by time).
- Baseline: mean predictor on `log1p(views_per_day)`.
- Model: **Ridge regression** on TF-IDF + controls; metrics: RMSE, MAE, R².
- Methods align with the program's EDA (Modules 3–4), linear models (Module 7), regularization (Module 9), and basic NLP via TF-IDF (Module 18).

## Results

**One-sentence answer:** Titles signaling _complete/beginner-friendly learning value_ ("full", "course", "complete", "for beginners", year tags like "2024") and AI terms ("ChatGPT", "AI") associate with **higher normalized views/day**, controlling for channel and timing.

**Scope:** N=23,002 videos, 30 channels; 2008-02-29–2024-06-21; target `log1p(views_per_day)`.

**Baseline:** RMSE=1.685, MAE=1.376, R²≈0.00 (holdout, log-scale mean predictor). **Best model:** Ridge (TF-IDF + channel + numerics). **Holdout:** RMSE=1.231, MAE=0.961, R²=0.467. **Lift vs baseline:** RMSE ↓27.0%, MAE ↓30.2%.

**Top positive title signals:** "full", "course", "2024", "beginners", "learn", "free", "for beginners", "chatgpt", "complete", "ai". **Top negative signals:** "autocad", "live coding", "php", "matlab", "css".

**Figures (see notebook):**

1. Coefficient table for top tokens. 2) Predicted vs. Actual (holdout). 3) Residuals histogram. 4) Segment view by day-of-week × tutorial keyword.

**Error/segment notes:** Under-predicts rare virals; over-predicts some niche "live coding" streams.

**Limitations:** No thumbnail/CTR or recommendation signals; likes/comments excluded to avoid label leakage; time drift and channel confounding possible.

**Implications for my channel:**

1. Title for _learning completeness_ ("Full Course", "For Beginners", year tags).
2. Keep concise, keyword-rich phrasing ("ChatGPT", "AI", "Project").
3. Experiment with publish day/time and repeatable series.

## Next steps

- Time-aware cross-validation; SHAP for explainability; add modest tree models as comparisons.
- Enrich features with duration, series tags, and light NLP beyond TF-IDF.

## Outline of project

- Notebook: **EDA + Baseline Modeling** (`Capstone_A20_Results_Notebook.ipynb`)

## Contact and Further Information

GitHub: ai-by-nic
