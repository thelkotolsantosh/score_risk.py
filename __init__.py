"""
Risk & Fraud Anomaly Detection Framework

A production-ready system for detecting fraudulent and anomalous user behavior
using unsupervised and semi-supervised learning on high-volume event data.
"""

__version__ = '0.1.0'
__author__ = 'Risk Detection Team'
__description__ = 'Anomaly detection for fraud, abuse, and infrastructure misuse'

from .src.simulate_events import EventSimulator
from .src.build_features import FeatureBuilder
from .src.train_model import RiskScorer
from .src.score_risk import RiskDetector

__all__ = [
    'EventSimulator',
    'FeatureBuilder',
    'RiskScorer',
    'RiskDetector'
]
