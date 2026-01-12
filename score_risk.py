"""Score user risk using trained model.
Generates risk scores and applies detection threshold.
"""
import argparse
import pandas as pd
import numpy as np
from train_model import RiskScorer


class RiskDetector:
    """Score users and flag high-risk cases."""

    def __init__(self, model_path):
        """
        Initialize detector with trained model.

        Args:
            model_path (str): Path to saved model pickle file
        """
        self.scorer = RiskScorer()
        self.scorer.load(model_path)

    def score_and_flag(self, features_df, threshold=0.5):
        """
        Generate risk scores and apply threshold.

        Args:
            features_df (pd.DataFrame): User features
            threshold (float): Risk score threshold for flagging (0-1)

        Returns:
            pd.DataFrame: Results with user_id, risk_score, flagged
        """
        # Generate risk scores
        risk_scores = self.scorer.score(features_df)

        # Create results
        results_df = features_df[['user_id']].copy()
        results_df['risk_score'] = risk_scores
        results_df['risk_percentile'] = pd.qcut(risk_scores, q=100, labels=False, duplicates='drop')
        results_df['flagged'] = (risk_scores >= threshold).astype(int)

        return results_df.sort_values('risk_score', ascending=False)

    def print_summary(self, results_df, threshold):
        """Print summary statistics."""
        num_flagged = results_df['flagged'].sum()
        pct_flagged = 100 * num_flagged / len(results_df)
        
        print(f"\n=== Risk Scoring Summary ===")
        print(f"Total users: {len(results_df)}")
        print(f"Threshold: {threshold:.2f}")
        print(f"Flagged: {num_flagged} ({pct_flagged:.1f}%)")
        print(f"\nRisk score distribution:")
        print(results_df['risk_score'].describe())
        
        if num_flagged > 0:
            print(f"\nTop 10 flagged users:")
            print(results_df[results_df['flagged'] == 1].head(10).to_string(index=False))


def main():
    """Command-line interface for risk scoring."""
    parser = argparse.ArgumentParser(
        description='Score user risk and flag anomalies'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/risk_scorer.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/features.csv',
        help='Input CSV with user features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/risk_scores.csv',
        help='Output CSV with scores and flags'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Risk score threshold for flagging (0-1)'
    )

    args = parser.parse_args()

    # Load model and features
    print(f"Loading model from {args.model_path}...")
    detector = RiskDetector(args.model_path)

    print(f"Loading features from {args.features}...")
    features_df = pd.read_csv(args.features)

    # Score
    print(f"Scoring {len(features_df)} users...")
    results_df = detector.score_and_flag(features_df, threshold=args.threshold)

    # Summary
    detector.print_summary(results_df, args.threshold)

    # Save
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
