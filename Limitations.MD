# Limitations & Known Issues

## Data Limitations

### Synthetic Data Realism

This project uses synthetic data generated to reflect realistic fraud patterns. Key limitations:

**What's realistic:**
- Fraud represents 1-2% of activity (class imbalance)
- Attack patterns cluster in feature space (velocity, geography, payment method)
- Legitimate users have diverse but consistent patterns
- Fraud rates vary by user segment (new accounts at higher risk)

**What's unrealistic:**
- No temporal dynamics (real fraud patterns evolve over minutes, hours, days)
- No spatial correlation (attacks cluster geographically; synthetic data randomizes)
- No relationship to platform features (synthetic data doesn't reflect product-specific abuse)
- No social graph (coordinated fraud rings not captured)
- No payment processor signals (decline rates, chargeback patterns not available)

**Implication**: Models trained here will overfit to synthetic patterns. Always validate on real, held-out data before deployment.

### Label Bias

Synthetic labels are created uniformly at random. Real labels exhibit bias:

1. **Detection bias**: Only fraud we detect gets labeled. Sophisticated attacks slip through, biasing the training set toward "easy" fraud.
2. **Investigation bias**: High-risk users get investigated more, inflating their fraud rate in labels.
3. **Feedback loop bias**: Models trained on past investigations reinforce who gets investigated, creating a feedback loop.
4. **Temporal bias**: Recent fraud is labeled, old fraud isn't. Models may memorize recent patterns and miss resurgences.

**Implication**: Offline metrics (precision, recall on test set) will exceed production performance. Plan for 10-20% performance degradation.

---

## Model Limitations

### Isolation Forest Constraints

**Strengths:**
- Interpretable: High-dimensional outlier detection without "black box" concerns
- Fast: O(n log n) training, O(log n) per-sample inference
- Flexible: Works with mixed feature types, non-linear boundaries
- Robust: Insensitive to feature scaling; handles missing data gracefully

**Weaknesses:**
- Assumes outliers are rare and distinct. In class-overlap regions (new user behaving legitimately but unusually), false positives spike.
- Uniform feature importance. Heavy features (transaction amount) may dominate subtle behavioral signals (timing patterns).
- Static thresholds. A fixed score threshold doesn't adapt as fraud evolves or as population composition shifts.
- Limited context. Scores are per-user; relationships between users (account rings, shared payment methods) are invisible.

**Not suitable for:**
- Coordinated fraud rings (requires graph analysis)
- Contextual outliers (behavior normal within subgroup, unusual globally)
- Real-time streaming with concept drift (model not designed for continuous learning)
- High-precision applications (need explainability beyond "high anomaly score")

### Semi-Supervised Shortcomings

This project uses unsupervised Isolation Forest, not semi-supervised learning. Why?

1. Labeled fraud data is sparse and biased (see Label Bias above).
2. Semi-supervised methods require careful hyperparameter tuning and label quality assumptions.
3. Simpler models are easier to debug, update, and explain to stakeholders.

**If labels become available:**
- Experiment with label propagation or self-training.
- Use labeled data for threshold calibration, not model training (to avoid bias).
- Compare semi-supervised models (LightGBM with class weights, XGBoost with sample weights) against Isolation Forest.

---

## Feature Engineering Limitations

### Aggregation Level

Features are aggregated at the user level over fixed time windows (1 day, 7 days, 30 days). This introduces bias:

1. **Timezone bias**: A user in UTC crossing midnight may have fragmented windows.
2. **Seasonality**: Velocity patterns differ by day of week and time of day; fixed windows smooth these.
3. **Concept shift**: Behavioral changes trigger immediately; aggregation delays detection.

**Better approach:**
- Use sliding windows (overlapping, not fixed daily windows).
- Condition features on user segment (new vs. established, geography, product tier).
- Engineer sequence features (entropy of inter-event times, burstiness metrics).

### Feature Staleness

Features computed daily are only valid for ~24 hours. In high-velocity fraud (dozens of events/hour), a user's risk score is stale within hours.

**Recommendation:**
- Compute incremental features in real-time (append new events to user profile).
- Retrain model weekly; rescore users daily; treat scores as "snapshots in time."
- For ultra-high-velocity systems, move to streaming aggregation (Kafka, Flink).

### Missing Features

This project has access only to event counts, amounts, and timestamps. Real systems often include:

- **Payment method**: Card vs. wallet vs. bank transfer; BIN information
- **Device fingerprint**: Device ID, OS, browser, location; device reputation
- **IP/VPN signals**: Datacenter IPs, VPN detection, proxy presence
- **Behavioral biometrics**: Typing speed, mouse movement, swipe patterns
- **External signals**: Credit bureau data, phone validation, email deliverability
- **Graph features**: Transaction network (who sends to whom), account rings

Models trained with only event-level features will underperform. Integrate external signals before production deployment.

---

## Evaluation Limitations

### Test Set Leakage

This project splits data temporally (train on days 1-30, test on days 31-40). However:

- Fraudsters in test set may overlap with training set (same user, different day).
- Model may memorize user profiles, not learn generalizable fraud patterns.
- Legitimate users behave consistently; anomalies within a user are easier to spot than anomalies across users.

**Better approach:**
- Train-test split by user (never test on user in training set).
- Evaluate on completely new user cohorts.
- Hold out a time window for temporal validation (avoid training on future data).

### No Operational Evaluation

Metrics are computed offline on static test sets. Real operation is messier:

- **Investigator feedback**: Some alerts are "not actionable" for reasons the model can't see (known account, legitimate purchase, recent onboarding).
- **Feedback lag**: It may take weeks to confirm if an alert was correct.
- **Model performance decay**: Detection rate drops after 30 days (concept drift).
- **False positive cost varies**: One false positive to one investigator is a 2-minute check; to another, it's a 30-minute deep dive.

**Recommendation:**
- Deploy with A/B testing: Route 5% of alerts to control (baseline rules) vs. treatment (model).
- Measure investigator agreement, time-to-decision, and false positive actionability.
- Monitor model performance daily; retrain if detection rate drops >10%.

---

## Deployment Considerations

### Scalability

Current implementation uses scikit-learn (single-machine, in-memory). For production:

- **At 1M events/day**: Need streaming feature aggregation and batch scoring.
- **At 10M events/day**: Single-machine bottleneck; move to distributed pipeline (Spark, Beam).
- **At 100M+ events/day**: Real-time scoring requires model serving (TensorFlow Serving, MLflow, SageMaker).

### Explainability

Isolation Forest scores lack interpretability. Stakeholders will ask: "Why is this user flagged?"

Current approach: User sees score (0-1), percentile rank, and feature distribution.

**Better approach:**
- SHAP values: Show which features drove the anomaly score.
- Prototypes: "This user is similar to confirmed fraudsters X and Y."
- Counterfactuals: "If velocity were normal, score would be 0.3 instead of 0.8."
- Actionability: "This flag suggests promo code abuse; recommend reviewing promotional activity."

### Regulatory & Bias

Fraud models may exhibit bias by protected attributes (race, gender, geography). This project does not audit for bias.

**Requirements for regulated use:**
- Bias audit: Model performance parity across demographic groups.
- Transparency: Explain decisions to users (GDPR "right to explanation").
- Fairness: Adjust thresholds to avoid disparate impact.
- Auditability: Log all model inputs, outputs, decisions for regulatory review.

---

## Recommended Next Steps

1. **Validate on real data**: Retrain on your transaction data; compare performance against baseline (rules, vendor solution).
2. **Engineer domain features**: Add payment method, device, IP, and external signals specific to your platform.
3. **Implement feedback loops**: Collect investigator labels daily; retrain weekly.
4. **Add explainability**: Integrate SHAP or similar for stakeholder transparency.
5. **Monitor for drift**: Track detection rate, false positive rate, and score distributions daily.
6. **A/B test in production**: Compare model alerts against baseline; measure investigator efficiency gains.
7. **Expand to other domains**: Apply same framework to promo abuse, account takeover, API abuse.

---

## Known Bugs & TODOs

### Known Issues

- Threshold optimization assumes static fraud rate; doesn't adapt to seasonal changes.
- Feature scaling is hardcoded; may not generalize to different event types.
- No handling of missing data; assumes all events are complete.
- Notebook kernels may be slow on large synthetic datasets (100k+ rows).

### TODO

- [ ] Streaming feature aggregation (Kafka consumer)
- [ ] SHAP explainability integration
- [ ] Bias audit (demographic parity, disparate impact)
- [ ] Production serving (Flask/FastAPI endpoint)
- [ ] Daily performance monitoring dashboard
- [ ] Multi-model ensemble (Isolation Forest + LOF + Robust PCA)
- [ ] Real-time concept drift detection (Adwin, DDM)

---

## Questions?

If you encounter issues or have suggestions, open a GitHub issue or reach out to the maintainers. Be specific about:

- What assumption did you violate? (e.g., "I trained on 100k rows, tested on same users")
- What performance gap did you see? (e.g., "Offline 85% precision, production 65%")
- What's your use case? (e.g., "Payment fraud, 1M transactions/day")

Real-world feedback helps improve the framework for everyone.
