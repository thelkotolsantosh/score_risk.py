"""
Train anomaly detection model.

Trains Isolation Forest for fraud risk scoring.
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RiskScorer:
    """Train and serialize risk scoring model."""

    def __init__(self):
        """Initialize model components."""
        self.model = None
        self.scaler = None
        self.feature_names = None

    def train(self, features_df, contamination=0.05, test_size=0.2, random_state=42):
        """
        Train Isolation Forest model.

        Args:
            features_df (pd.DataFrame): User features with 'is_fraud' label
            contamination (float): Expected proportion of anomalies (0-1)
            test_size (float): Proportion for test set
            random_state (int): Random seed

        Returns:
            dict: Training metrics
        """
        # Separate features and labels
        y = features_df['is_fraud']
        X = features_df.drop(['user_id', 'is_fraud'], axis=1)
        
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Isolation Forest
        print(f"Training Isolation Forest with contamination={contamination}...")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.model.fit(X_train_scaled)

        # Evaluate
        train_scores = self.model.decision_function(X_train_scaled)
        test_scores = self.model.decision_function(X_test_scaled)

        # Convert to risk probabilities (higher = more anomalous)
        train_risk = 1 / (1 + np.exp(-train_scores))
        test_risk = 1 / (1 + np.exp(-test_scores))

        # Metrics
        train_pred = (train_risk > 0.5).astype(int)
        test_pred = (test_risk > 0.5).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'test_precision': precision_score(y_test, test_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'test_recall': recall_score(y_test, test_pred, zero_division=0),
            'train_f1': f1_score(y_train, train_pred, zero_division=0),
            'test_f1': f1_score(y_test, test_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, test_risk),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'num_features': len(self.feature_names)
        }

        return metrics

    def score(self, features_df):
        """
        Generate risk scores for features.

        Args:
            features_df (pd.DataFrame): User features

        Returns:
            np.ndarray: Risk scores (0-1, higher = more anomalous)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        decision_scores = self.model.decision_function(X_scaled)
        
        # Convert to risk probability
        risk_scores = 1 / (1 + np.exp(-decision_scores))
        return risk_scores

    def save(self, model_path):
        """Save model to disk."""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {model_path}")

    def load(self, model_path):
        """Load model from disk."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"Model loaded from {model_path}")


def main():
    """Command-line interface for model training."""
    parser = argparse.ArgumentParser(
        description='Train risk scoring model'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/features.csv',
        help='Input CSV with user features'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/risk_scorer.pkl',
        help='Output model file path'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of anomalies'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for testing'
    )

    args = parser.parse_args()

    # Load features
    print(f"Loading features from {args.features}...")
    features_df = pd.read_csv(args.features)
    print(f"Loaded {len(features_df)} user profiles")

    # Train model
    scorer = RiskScorer()
    metrics = scorer.train(
        features_df,
        contamination=args.contamination,
        test_size=args.test_size
    )

    # Print results
    print("\n=== Training Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Save model
    scorer.save(args.model_path)


if __name__ == '__main__':
    main()
