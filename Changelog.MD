# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- [ ] SHAP explainability integration
- [ ] LOF and Robust PCA models
- [ ] REST API server (Flask)
- [ ] Batch scoring pipeline
- [ ] Concept drift detection
- [ ] Multi-model ensemble
- [ ] Production monitoring dashboard
- [ ] Bias audit module

## [0.1.0] - 2025-01-12

### Added
- Initial release of risk & fraud anomaly detection framework
- Event simulator for synthetic fraud pattern generation
- Feature engineering pipeline (event aggregation)
- Isolation Forest anomaly detection model
- Risk scoring and threshold-based flagging
- Comprehensive documentation and getting started guide
- Business impact analysis framework
- Limitations and known issues documentation
- 5 educational notebooks covering full workflow
- MIT License

### Components
- `simulate_events.py`: Generate synthetic user events with fraud patterns
- `build_features.py`: Aggregate events into behavioral features
- `train_model.py`: Train Isolation Forest risk scorer
- `score_risk.py`: Generate risk scores and flags
- `01_problem_framing.ipynb`: Business problem setup
- `02_eda.ipynb`: Exploratory analysis of synthetic data
- `03_feature_engineering.ipynb`: Feature construction deep dive
- `04_modeling.ipynb`: Model training and comparison
- `05_thresholding.ipynb`: Threshold optimization

### Known Limitations
- Synthetic data only (not suitable for direct production use)
- Single model approach (ensemble planned)
- No explainability features (SHAP planned)
- No concept drift detection
- Static thresholds (adaptive thresholding planned)
- Limited to event-level features (missing payment, device, IP signals)

### Documentation
- README.md: Comprehensive project overview
- GETTING_STARTED.md: Step-by-step setup and quickstart
- CONTRIBUTING.md: Contribution guidelines
- reports/business_impact.md: Cost-benefit analysis framework
- reports/limitations.md: Detailed limitations and failure modes
- requirements.txt: Python dependencies

---

## Version Notes

### Naming Convention
- **Major (X.0.0)**: Breaking changes, major feature additions
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, documentation updates

### Release Schedule
- Releases follow user feedback and contribution pace
- Patch releases as-needed for bugs
- Minor releases quarterly (planned)
- Major releases annually (planned)

### Backward Compatibility
- All releases maintain backward compatibility until major version bump
- Deprecation warnings provided 1 version in advance

---

## How to Use This Changelog

- **For users**: Check here before upgrading to see what's new/changed
- **For contributors**: See what's planned and what needs help
- **For maintainers**: Document changes before each release

Please keep this file updated with every meaningful change.

---

## Future Roadmap (2025)

### Q1 2025 (Planned for v0.2.0)
- SHAP integration for explainability
- Additional baseline models (LOF, IF variants)
- Production monitoring module
- Real-world case study

### Q2 2025 (Planned for v0.3.0)
- REST API for scoring
- Batch processing pipeline
- Concept drift detection
- Multi-model ensemble

### Q3 2025 (Planned for v0.4.0)
- Production serving architecture
- Bias audit framework
- Advanced feature engineering
- User documentation portal

### Q4 2025 (Planned for v1.0.0)
- Stable API guarantees
- Comprehensive test coverage
- Production deployment guide
- Community benchmarks

---

## Contributing to Changelog
When submitting a PR:
1. Add an entry under `[Unreleased]` in appropriate section
2. Use past tense ("Added X" not "Add X")
3. Link to relevant issues/PRs if applicable
4. Maintainers will move to version section at release time

Format:
- Brief description [#issue_number](link) - @contributor

Thank you for using and improving this project!
