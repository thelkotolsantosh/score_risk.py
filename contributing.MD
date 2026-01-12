# Contributing
We welcome contributions! This document outlines how to get involved.

## Code of Conduct
Be respectful, inclusive, and constructive. Harassment or discrimination is not tolerated.

## Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push: `git push origin feature/your-idea`
6. Open a pull request with description

## Development Setup
bash
# Clone your fork
git clone https://github.com/yourusername/risk-fraud-anomaly-detection.git
cd risk-fraud-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pylint black jupyter


## Testing

Run tests before submitting PR:

bash
# Unit tests (when added)
pytest tests/

# Linting
pylint src/

# Code style
black src/


## Areas for Contribution

### Code
- New anomaly detection models (LOF, Robust PCA, VAE)
- Production serving (REST API, batch pipeline)
- Explainability (SHAP, LIME integration)
- Performance optimization

### Documentation
- Clarify confusing sections
- Add examples and use cases
- Translate to other languages
- Improve docstrings

### Analysis
- Extend notebooks with new experiments
- Compare against baselines
- Analyze model fairness and bias
- Real-world case studies

### Data
- Contribute synthetic data generators for new fraud types
- Share real-world datasets (anonymized)
- Data augmentation techniques

## Pull Request Process

1. **Update documentation**: If adding features, update README and docstrings
2. **Add tests**: New code should have corresponding tests
3. **Test locally**: Run full pipeline end-to-end
4. **Clear commit messages**: Use imperative mood ("Add feature" not "Added feature")
5. **Reference issues**: Link related GitHub issues
6. **One feature per PR**: Keep PRs focused and reviewable

### PR Template

markdown
## Description
What does this change do?

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Model improvement

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] No breaking changes
- [ ] Tests added/updated


## Reporting Issues

Found a bug? Have a suggestion? Open an issue:

### Bug Report Template

markdown
## Description
What's the problem?

## Steps to Reproduce
1. Run command X
2. Observe behavior Y

## Expected Behavior
What should happen?

## Environment
- Python version
- OS
- Dependencies (from requirements.txt)

## Additional Context
Logs, stack traces, screenshots?


### Feature Request Template

markdown
## Description
What would you like?

## Use Case
Why is this needed?

## Proposed Solution
How might this work?

## Alternatives Considered
Other approaches?


## Style Guide

### Python

- Use PEP 8 (4-space indentation)
- Type hints for function signatures
- Docstrings for all public functions
- Meaningful variable names

python
def score_users(features_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Generate risk scores for users.
    
    Args:
        features_df: User feature matrix
        threshold: Risk score threshold (0-1)
    
    Returns:
        DataFrame with risk scores and flags
    """
    pass


### Commits


Type: Brief summary (50 chars max)

Longer explanation if needed. Wrap at 72 characters.
Reference issue: #123


Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code reorganization (no behavior change)
- `test:` Test additions/updates
- `style:` Formatting (no logic change)

## Review Process

- Maintainers review all PRs within 1 week
- Constructive feedback provided
- May request changes before merging
- Merged PRs appear in next release notes

## Recognition

Contributors are recognized in:
- Commit history
- Release notes
- Contributors section (coming soon)

## Questions?

Open an issue labeled `question` or reach out directly.

Thank you for contributing!
