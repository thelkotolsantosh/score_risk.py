# Risk & Fraud Anomaly Detection

A comprehensive framework for detecting anomalous user behavior that may indicate fraud, abuse, or infrastructure misuse using partially labeled event data.

## Overview

This project demonstrates production-ready approaches to anomaly detection in high-volume transactional systems. Rather than building binary classifiers on incomplete labels, we construct a **risk scoring system** that ranks users by suspicion level and enables business teams to set detection thresholds based on operational tolerance.

### Use Cases

This framework applies to multiple domains:

- **Payment Fraud**: Detecting unauthorized transactions, card testing, and chargebacks
- **Promo/Coupon Abuse**: Identifying systematic exploitation of discounts and referral programs
- **Account Takeover**: Spotting unauthorized access and credential compromise
- **API Abuse**: Finding automated scraping, rate limit exploitation, and resource exhaustion
- **Cloud Cost Leakage**: Detecting infrastructure misuse and unusual compute consumption

## Business Problem

Real-world risk systems face unique challenges:

1. **Incomplete Labels**: Fraud is confirmed only after investigation, investigation happens only for suspected cases, and some fraud slips past entirely
2. **Delayed Feedback**: Labels may arrive weeks or months after events, making time-aware model updates critical
3. **Class Imbalance**: Fraudsters represent <1% of users, yet false positives carry real operational cost
4. **Concept Drift**: Attack patterns evolve; models that worked last month may miss new tactics
5. **Business Trade-offs**: Each false positive triggers manual review; each false negative causes direct loss

## Approach

This project follows a three-phase workflow:

### Phase 1: Problem Framing
- Define the risk ranking task vs. binary classification
- Identify available and delayed labels
- Quantify false positive cost vs. detection benefit

### Phase 2: Feature Engineering
- Extract behavioral patterns from raw event streams
- Build user-level aggregates (count, velocity, entropy, recency)
- Construct derived features capturing sequence and context
- Document feature assumptions and failure modes

### Phase 3: Risk Scoring & Thresholding
- Train unsupervised and semi-supervised models
- Evaluate via precision-at-K and business impact metrics
- Establish decision thresholds based on alert budget
- Monitor score stability and drift over time

## Data

All data is **synthetic and generated programmatically** to simulate realistic patterns:

- **Normal behavior**: Regular purchasing, consistent patterns, expected geographic/temporal distribution
- **Automated abuse**: Bot-like velocity spikes, round-dollar amounts, sequential IDs
- **Account takeover**: Sudden geographic shift, new payment methods, unusual device fingerprints
- **Infrastructure misuse**: API endpoints hammered with identical requests, unusual data volumes

Generate synthetic data:
```bash
python src/simulate_events.py --events 50000 --output data/raw/user_events.csv
```

## Installation

### Requirements
- Python 3.8+
- pip or conda

### Setup

Clone the repository:
```bash
git clone https://github.com/yourusername/risk-fraud-anomaly-detection.git
cd risk-fraud-anomaly-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data
```bash
python src/simulate_events.py \
  --events 100000 \
  --fraud_rate 0.02 \
  --output data/raw/user_events.csv
```

### 2. Build Features
```bash
python src/build_features.py \
  --input data/raw/user_events.csv \
  --output data/processed/features.csv
```

### 3. Train Risk Model
```bash
python src/train_model.py \
  --features data/processed/features.csv \
  --model_path models/risk_scorer.pkl
```

### 4. Score User Risk
```bash
python src/score_risk.py \
  --model_path models/risk_scorer.pkl \
  --features data/processed/features.csv \
  --output results/risk_scores.csv \
  --threshold 0.7
```

## Notebooks

Work through the analysis in order:

1. **01_problem_framing.ipynb**: Define the business problem, label structure, and evaluation metrics
2. **02_eda.ipynb**: Explore synthetic data, identify patterns in normal vs. anomalous behavior
3. **03_feature_engineering.ipynb**: Build behavioral features, analyze feature distributions
4. **04_modeling.ipynb**: Train Isolation Forest and baseline models, compare performance
5. **05_thresholding.ipynb**: Optimize detection threshold, analyze precision/recall trade-offs

## Models

### Isolation Forest
Unsupervised anomaly detection via random partitioning. Effective for:
- High-dimensional feature spaces
- Non-linear decision boundaries
- Fast inference at scale

**Why it works here**: Fraudsters occupy sparse regions of feature space; isolation trees partition into regions efficiently.

### Robust Z-Score Baseline
Simple univariate outlier detection (MAD-based). Serves as:
- Sanity check for complex models
- Production fallback if model performance degrades
- Interpretable alternative for regulated systems

## Evaluation

### Metrics

**Precision at K**: Among top-K flagged users, what % are true positives?
```
Precision@K = True Positives in top K / K
```

**Detection Rate**: Of all confirmed fraudsters, what % did we catch?
```
Detection Rate = True Positives / Total Actual Fraudsters
```

**Alert Volume**: How many users require manual review?
```
Alert Volume = Count of users scoring above threshold
```

**Cost Analysis**: Incorporate business parameters:
- Cost of investigating false positive: $500
- Average fraud loss if missed: $5,000
- Optimal threshold maximizes: (True Positives × 5000) - (False Positives × 500)

### Production Metrics

Beyond offline evaluation:
- **Score Stability**: Standard deviation of model scores on same user over time
- **Label Feedback Lag**: Time from alert to confirmed label; impacts retraining cadence
- **Concept Drift**: Model performance decay when evaluated on recent data vs. training data
- **Alert Fatigue**: Investigator feedback on actionability of high-confidence flags

## Project Structure

```
risk-fraud-anomaly-detection/
├── data/
│   ├── raw/
│   │   └── user_events.csv              # Synthetic event stream
│   └── processed/
│       └── features.csv                 # Aggregated user features
│
├── notebooks/
│   ├── 01_problem_framing.ipynb         # Business problem setup
│   ├── 02_eda.ipynb                     # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb     # Feature construction
│   ├── 04_modeling.ipynb                # Model training & comparison
│   └── 05_thresholding.ipynb            # Threshold optimization
│
├── src/
│   ├── simulate_events.py               # Generate synthetic data
│   ├── build_features.py                # Transform events to features
│   ├── train_model.py                   # Train anomaly detector
│   └── score_risk.py                    # Score new data
│
├── models/
│   └── risk_scorer.pkl                  # Trained Isolation Forest
│
├── reports/
│   ├── business_impact.md               # Cost-benefit analysis
│   └── limitations.md                   # Known issues & assumptions
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── LICENSE                              # MIT License
└── .gitignore                           # Git exclusions
```

## Key Findings

### What Works

- **Velocity features** (events per user per day) are the strongest single predictor of fraud
- **Ensemble of unsupervised models** outperforms any single model
- **Soft scoring** (continuous risk ranks) enables business teams to optimize thresholds dynamically
- **Regular retraining** (weekly) is necessary; model performance decays after 30 days

### What Doesn't

- Binary classification on partially labeled data introduces systematic bias
- High-dimensional feature spaces without domain expertise invite overfitting
- Unsupervised-only approaches miss fraud types not visible in aggregate features
- Static thresholds fail under concept drift

## Trade-offs & Limitations

See [limitations.md](reports/limitations.md) for detailed discussion of:

- False positive cost vs. detection benefit
- Synthetic data realism constraints
- Feature engineering assumptions
- Label bias in evaluation
- Model interpretability vs. performance

## Contributing

Contributions are welcome. Please:

1. Open an issue describing the problem or feature
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/your-idea`)
4. Commit changes with clear messages
5. Push and open a pull request

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this project in research or production, please cite:

```bibtex
@software{risk-fraud-anomaly-2025,
  title = {Risk & Fraud Anomaly Detection},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/risk-fraud-anomaly-detection}
}
```

## Disclaimer

All data presented in this repository is **synthetic and generated for educational purposes only**. No real user, transaction, or fraud data is included. This project is designed as a learning resource for anomaly detection techniques and business applications — not as production fraud detection software. Always validate approaches on your own data and consult domain experts before deployment.

## Additional Resources

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08.pdf) — Liu et al., 2008
- [Anomaly Detection: A Survey](https://arxiv.org/abs/1811.03728) — Goldstein & Uchida, 2016
- [Fraud Detection Methods](https://arxiv.org/abs/1911.08527) — Carcillo et al., 2019
- [Class Imbalance Learning](https://imbalanced-learn.org/) — imbalanced-learn documentation

## Questions?

Open an issue on GitHub or reach out with questions about the framework, models, or implementation.
