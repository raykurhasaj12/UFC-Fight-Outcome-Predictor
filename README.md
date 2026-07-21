UFC Fight Outcome Predictor
A Machine Learning Approach to Combat Sports Prediction

Status: Active — Core model, validation, and interactive dashboard complete. Live card testing ongoing.

Overview

This project builds a machine learning system to predict UFC fight outcomes and identify positive expected value (EV) betting opportunities. It covers the full data science pipeline — from data acquisition and feature engineering, through model development and temporal validation, to a live interactive web dashboard connected to real-time fighter statistics.

The project was motivated by the UFC being one of the least efficiently priced sports markets, due to its relatively small betting volume compared to football or NFL. This creates opportunities for a well-calibrated model to find edges that bookmakers may underestimate.

Project Structure
UFC-Predictor/
│
├── UFC_MODEL.ipynb                    # Full modelling pipeline
├── app.py                             # Flask REST API
├── index.html                         # Interactive dashboard
├── scraper.py                         # Selenium scraper for UFCStats.com
├── model.pkl                          # Saved hybrid model (trained 2010–2018)
├── ufc_2022_2025_preprocessed.csv     # Out-of-sample dataset (2,346 fights)
├── requirements.txt                   # Python dependencies
├── /Plots                             # EDA and model evaluation charts
└── README.md
Pipeline

1. Data Acquisition
Primary dataset: Kaggle UFC Data — 5,902 fights, pre-fight averages only (no data leakage). Out-of-sample: custom Selenium scraper collecting 2,346 fights from UFCStats.com (2022–2026).

2. Feature Engineering
Nine advantage differentials (Red − Blue): reach, height, age, significant strike accuracy, knockdown average, takedown accuracy, submission attempts, win streak, control time.

3. Statistical Validation
Point Biserial Correlation confirmed all nine features statistically significant at the 1% level before inclusion.

Models

Logistic Regression
The baseline model achieved 68.9% in-sample accuracy. However, suspicion of overfitting prompted out-of-sample validation, which revealed accuracy dropping to ~58% — a 10%+ degradation suggesting the model had been fitting to dataset-specific noise rather than genuine patterns.

Random Forest
Slightly lower overall accuracy at 67.4%, but crucially outperformed logistic regression in the 80–90% confidence band — 81.5% win rate vs 74.5%. This revealed the random forest was better calibrated when the model was highly confident, even if marginally weaker on average.

Hybrid Model (LR + RF)
The hybrid was developed to combine the complementary strengths of both approaches. Logistic regression is used as the default, but when it predicts a fighter to have an 80–90% chance of winning, the random forest takes over. This captures the random forest's superior calibration at high confidence while retaining logistic regression's performance across the full range.

Neural Network (sklearn)
A three-layer sklearn neural network (128 → 64 → 32 nodes) was trained on the same 2010–2018 dataset using the same seven features. It outperformed the hybrid model slightly and — importantly — showed no meaningful degradation across the three temporal periods, providing an independent confirmation that the signal being captured is genuine.

Pytorch was also tested but performed marginally worse than sklearn, so the simpler sklearn implementation was retained. This is a useful reminder that architectural complexity does not always translate to better performance, particularly on structured tabular data.

Consensus Ensemble
With two independently built models (Hybrid and Neural Network) both converging on ~62% out-of-sample accuracy, a natural question arose: when both models agree on the same outcome, does this increase confidence? The consensus ensemble flags a prediction only when the Hybrid and Neural Network are within ±5% agreement. This produced the highest accuracy of any single approach but reduced coverage to ~77% of fights — meaning roughly 1 in 4 fights produces no prediction. A ±5% threshold was chosen as the balance between accuracy and coverage.

Temporal Validation

Training was restricted to 2010–2018 to remove pre-modern UFC noise. The model was then evaluated across three distinct time periods it had never seen:

Model	2010–2018 (Training)	2019–2021 (Validation)	2022–2026 (Out-of-Sample)
Hybrid (LR + RF)	61.7%	59.3%	62.5%
Neural Network (sklearn)	63.0%	60.6%	64.1%
Consensus Ensemble (±5%)	63.1%	60.5%	64.0%

The consistency across all three periods — spanning 16 years of UFC data — is the strongest evidence that the models have learned genuine underlying patterns in fighter attributes rather than overfitting to historical noise.

Interactive Dashboard

A local web dashboard was built to apply the model to upcoming fight cards in real time. The user inputs fighter names and bookmaker odds, clicks run, and the tool:

Auto-scrapes live fighter statistics from UFCStats.com via Selenium
Runs the hybrid model to generate win probabilities for each fight
Calculates Expected Value (EV) for each fighter based on model probability vs bookmaker implied probability
Flags value bets where EV exceeds a minimum threshold
Displays factor differentials for every fight (reach, age, strike accuracy etc.)
Runs a 10,000-iteration Monte Carlo simulation across all value bets, showing expected profit, median, best/worst case, and the percentage of simulated cards that are profitable

EV Formula:

EV = (model_probability × (odds − 1) × stake) − ((1 − model_probability) × stake)

A positive EV means the model believes the fighter is underpriced by the bookmaker — the gap between model probability and implied bookmaker probability is the edge being exploited.

bash
pip install -r requirements.txt
python app.py
# Open index.html in your browser
Key Findings
Reach advantage is the strongest predictor of fight outcome (coefficient: 0.261)
Age disadvantage is the strongest negative predictor (coefficient: −0.344)
Restricting training to 2010–2018 improved out-of-sample consistency despite reducing in-sample accuracy from 69.8% to 61.7% — the reduction is not a failure but evidence the model stopped memorising historical noise
Random Forest outperforms Logistic Regression specifically in the 80–90% confidence band (81.5% vs 74.5%), which motivated the hybrid design
sklearn Neural Network slightly outperformed PyTorch on this dataset, confirming that simpler architectures can outperform complex ones on structured tabular data
Two entirely independent models converged on ~62% out-of-sample accuracy, strongly suggesting the signal is real
Tech Stack

Python — scikit-learn, PyTorch, Flask, Selenium, pandas, numpy | Frontend — HTML, CSS, JavaScript, Chart.js

Data Sources
Warrier, R. (2021). UFC Data. Kaggle. https://www.kaggle.com/datasets/rajeevw/ufcdata
UFCStats.com — http://www.ufcstats.com
Fighter URL index: github.com/Greco1899/scrape_ufc_stats

Ray Kurhasaj — github.com/raykurhasaj12
