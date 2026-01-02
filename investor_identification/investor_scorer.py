"""
Investor Scorer - Calculate likelihood that a property owner is an active investor

Combines multiple signals into a 0-100 score:
- Entity type (LLC, Corp, Trust)
- Portfolio size (multiple properties)
- Absentee ownership
- Business address indicators
- Recent purchase activity
- Investor-related names
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from config.settings import (
    INVESTOR_SCORE_WEIGHTS,
    TIER_1_THRESHOLD,
    TIER_2_THRESHOLD,
    TIER_3_THRESHOLD,
    TIER_4_THRESHOLD,
    PORTFOLIO_THRESHOLDS,
)
from config.logging_config import get_logger


class InvestorScorer:
    """
    Calculate investor likelihood scores for property owners.

    Higher scores = more likely to be an active real estate investor
    who would be interested in buying leads.
    """

    def __init__(self, weights: Dict[str, int] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.weights = weights or INVESTOR_SCORE_WEIGHTS

        # Normalize weights to sum to 100
        total_weight = sum(self.weights.values())
        self.normalized_weights = {
            k: (v / total_weight) * 100
            for k, v in self.weights.items()
        }

    def score_investor(self, data: Dict) -> Tuple[int, str, List[str]]:
        """
        Calculate investor score for a single owner/property.

        Args:
            data: Dict with owner/property attributes

        Returns:
            Tuple of (score, tier, reasons)
        """
        score = 0
        reasons = []

        # 1. Entity Type Score (0-30 points)
        entity_score, entity_reasons = self._score_entity_type(data)
        score += entity_score
        reasons.extend(entity_reasons)

        # 2. Portfolio Size Score (0-25 points)
        portfolio_score, portfolio_reasons = self._score_portfolio(data)
        score += portfolio_score
        reasons.extend(portfolio_reasons)

        # 3. Absentee Owner Score (0-15 points)
        absentee_score, absentee_reasons = self._score_absentee(data)
        score += absentee_score
        reasons.extend(absentee_reasons)

        # 4. Business Address Score (0-10 points)
        address_score, address_reasons = self._score_address(data)
        score += address_score
        reasons.extend(address_reasons)

        # 5. Recent Activity Score (0-10 points)
        activity_score, activity_reasons = self._score_activity(data)
        score += activity_score
        reasons.extend(activity_reasons)

        # 6. Investor Name Score (0-10 points)
        name_score, name_reasons = self._score_investor_name(data)
        score += name_score
        reasons.extend(name_reasons)

        # Cap score at 100
        score = min(int(score), 100)

        # Determine tier
        tier = self._get_tier(score)

        return score, tier, reasons

    def _score_entity_type(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on entity type (LLC, Corp, Trust, etc.)"""
        max_points = self.normalized_weights.get('entity_type', 30)
        entity_type = data.get('entity_type', 'individual')
        is_investor = data.get('is_investor_entity', False)

        if entity_type == 'llc':
            return max_points, ["LLC entity (high investor probability)"]
        elif entity_type == 'corporation':
            return max_points * 0.9, ["Corporation (likely investor)"]
        elif entity_type == 'partnership':
            return max_points * 0.85, ["Partnership (likely investor)"]
        elif entity_type == 'trust':
            return max_points * 0.6, ["Trust entity (possible investor)"]
        elif entity_type == 'estate':
            return max_points * 0.3, ["Estate (possible investor)"]
        else:
            return 0, []

    def _score_portfolio(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on number of properties owned"""
        max_points = self.normalized_weights.get('portfolio_size', 25)
        portfolio_size = data.get('portfolio_size', 1)

        if portfolio_size >= PORTFOLIO_THRESHOLDS['institutional']:
            return max_points, [f"Institutional investor ({portfolio_size} properties)"]
        elif portfolio_size >= PORTFOLIO_THRESHOLDS['large']:
            return max_points * 0.9, [f"Large portfolio ({portfolio_size} properties)"]
        elif portfolio_size >= PORTFOLIO_THRESHOLDS['medium']:
            return max_points * 0.7, [f"Medium portfolio ({portfolio_size} properties)"]
        elif portfolio_size >= PORTFOLIO_THRESHOLDS['small']:
            return max_points * 0.5, [f"Small portfolio ({portfolio_size} properties)"]
        else:
            return 0, []

    def _score_absentee(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on absentee ownership"""
        max_points = self.normalized_weights.get('absentee_owner', 15)

        # Check various absentee indicators
        is_absentee = data.get('is_absentee', False)
        absentee_ratio = data.get('absentee_ratio', 0)  # For aggregated data

        if absentee_ratio > 0.8:
            return max_points, ["80%+ properties are absentee-owned"]
        elif absentee_ratio > 0.5:
            return max_points * 0.7, ["50%+ properties are absentee-owned"]
        elif is_absentee:
            return max_points * 0.5, ["Absentee owner"]
        else:
            return 0, []

    def _score_address(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on business address indicators"""
        max_points = self.normalized_weights.get('business_address', 10)
        reasons = []
        score = 0

        if data.get('has_po_box', False):
            score += max_points * 0.6
            reasons.append("PO Box address")

        if data.get('has_business_address', False):
            score += max_points * 0.4
            reasons.append("Business address (suite/floor)")

        return min(score, max_points), reasons

    def _score_activity(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on recent purchase activity"""
        max_points = self.normalized_weights.get('recent_activity', 10)

        recent_buyer = data.get('recent_buyer', False)
        recent_purchases = data.get('recent_purchases', 0)

        if recent_purchases >= 3:
            return max_points, [f"Very active buyer ({recent_purchases} recent purchases)"]
        elif recent_purchases >= 1:
            return max_points * 0.7, [f"Active buyer ({recent_purchases} recent purchase(s))"]
        elif recent_buyer:
            return max_points * 0.5, ["Recent property purchase"]
        else:
            return 0, []

    def _score_investor_name(self, data: Dict) -> Tuple[float, List[str]]:
        """Score based on investor-related keywords in name"""
        max_points = self.normalized_weights.get('investor_name', 10)

        if data.get('has_investor_name', False):
            return max_points, ["Name contains investor keywords (Properties, Holdings, etc.)"]
        else:
            return 0, []

    def _get_tier(self, score: int) -> str:
        """Determine tier based on score"""
        if score >= TIER_1_THRESHOLD:
            return 'tier_1'
        elif score >= TIER_2_THRESHOLD:
            return 'tier_2'
        elif score >= TIER_3_THRESHOLD:
            return 'tier_3'
        elif score >= TIER_4_THRESHOLD:
            return 'tier_4'
        else:
            return 'tier_5'

    def get_tier_description(self, tier: str) -> str:
        """Get human-readable tier description"""
        descriptions = {
            'tier_1': 'High-Confidence Investor - Prime prospect',
            'tier_2': 'Probable Investor - Strong prospect',
            'tier_3': 'Possible Investor - Worth contacting',
            'tier_4': 'Unlikely Investor - Low priority',
            'tier_5': 'Not an Investor - Skip',
        }
        return descriptions.get(tier, 'Unknown')

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all owners in a DataFrame.

        Expects DataFrame with columns from EntityClassifier and PortfolioDetector:
        - entity_type, is_investor_entity, has_investor_name
        - portfolio_size, is_portfolio_investor
        - is_absentee, has_po_box, has_business_address
        - recent_buyer (optional)

        Returns:
            DataFrame with investor_score, investor_tier, score_reasons added
        """
        self.logger.info(f"Scoring {len(df):,} records...")

        df = df.copy()

        # Score each row
        scores = []
        tiers = []
        reasons_list = []

        for _, row in df.iterrows():
            data = row.to_dict()
            score, tier, reasons = self.score_investor(data)
            scores.append(score)
            tiers.append(tier)
            reasons_list.append('; '.join(reasons) if reasons else 'No investor signals')

        df['investor_score'] = scores
        df['investor_tier'] = tiers
        df['score_reasons'] = reasons_list
        df['tier_description'] = df['investor_tier'].apply(self.get_tier_description)

        # Log summary
        self._log_scoring_summary(df)

        return df

    def _log_scoring_summary(self, df: pd.DataFrame):
        """Log summary of scoring results"""
        self.logger.info("Scoring complete:")

        # Score distribution
        avg_score = df['investor_score'].mean()
        self.logger.info(f"  Average score: {avg_score:.1f}")

        # Tier distribution
        tier_counts = df['investor_tier'].value_counts().sort_index()
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            self.logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")

        # High-value prospects
        high_value = len(df[df['investor_score'] >= TIER_1_THRESHOLD])
        self.logger.info(f"  High-value prospects (score >= {TIER_1_THRESHOLD}): {high_value:,}")

    def get_top_prospects(self, df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        """
        Get top N investor prospects by score.
        """
        if 'investor_score' not in df.columns:
            df = self.score_dataframe(df)

        return df.nlargest(top_n, 'investor_score')

    def filter_by_tier(self, df: pd.DataFrame, tiers: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to only include specified tiers.

        Args:
            df: Scored DataFrame
            tiers: List of tiers to include (e.g., ['tier_1', 'tier_2'])
        """
        if 'investor_tier' not in df.columns:
            df = self.score_dataframe(df)

        return df[df['investor_tier'].isin(tiers)]

    def get_scoring_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive scoring summary"""
        if 'investor_score' not in df.columns:
            df = self.score_dataframe(df)

        summary = {
            'total_records': len(df),
            'avg_score': float(df['investor_score'].mean()),
            'median_score': float(df['investor_score'].median()),
            'max_score': int(df['investor_score'].max()),
            'min_score': int(df['investor_score'].min()),
            'tier_distribution': df['investor_tier'].value_counts().to_dict(),
            'high_value_count': int((df['investor_score'] >= TIER_1_THRESHOLD).sum()),
            'probable_investors': int((df['investor_score'] >= TIER_2_THRESHOLD).sum()),
        }

        return summary


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Quick test of investor scorer"""

    # Sample data (as if processed by EntityClassifier and PortfolioDetector)
    sample_data = pd.DataFrame([
        {
            'owner_name': 'ABC PROPERTIES LLC',
            'entity_type': 'llc',
            'is_investor_entity': True,
            'has_investor_name': True,
            'portfolio_size': 15,
            'is_absentee': True,
            'has_po_box': True,
            'has_business_address': False,
            'recent_buyer': True,
        },
        {
            'owner_name': 'JOHN SMITH',
            'entity_type': 'individual',
            'is_investor_entity': False,
            'has_investor_name': False,
            'portfolio_size': 1,
            'is_absentee': False,
            'has_po_box': False,
            'has_business_address': False,
            'recent_buyer': False,
        },
        {
            'owner_name': 'SMITH FAMILY TRUST',
            'entity_type': 'trust',
            'is_investor_entity': True,
            'has_investor_name': False,
            'portfolio_size': 3,
            'is_absentee': True,
            'has_po_box': False,
            'has_business_address': False,
            'recent_buyer': False,
        },
        {
            'owner_name': 'MEGA HOLDINGS INC',
            'entity_type': 'corporation',
            'is_investor_entity': True,
            'has_investor_name': True,
            'portfolio_size': 50,
            'is_absentee': True,
            'has_po_box': True,
            'has_business_address': True,
            'recent_buyer': True,
        },
    ])

    scorer = InvestorScorer()
    scored_df = scorer.score_dataframe(sample_data)

    print("\nInvestor Scoring Results:")
    print("-" * 80)
    for _, row in scored_df.iterrows():
        print(f"\n{row['owner_name']}")
        print(f"  Score: {row['investor_score']}/100")
        print(f"  Tier: {row['investor_tier']} - {row['tier_description']}")
        print(f"  Reasons: {row['score_reasons']}")

    print("\n" + "=" * 80)
    summary = scorer.get_scoring_summary(scored_df)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
