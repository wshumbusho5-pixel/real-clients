"""
Portfolio Detector - Identify investors who own multiple properties

Aggregates property records by owner to find:
- Portfolio size (number of properties)
- Total portfolio value
- Property types owned
- Geographic concentration
- Recent acquisition activity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from config.settings import PORTFOLIO_THRESHOLDS, MIN_PORTFOLIO_SIZE
from config.logging_config import get_logger


class PortfolioDetector:
    """
    Detect portfolio investors by aggregating properties per owner.

    Someone owning 5+ properties is almost certainly an investor.
    This is one of the strongest signals we can use.
    """

    def __init__(self, min_portfolio_size: int = MIN_PORTFOLIO_SIZE):
        self.logger = get_logger(self.__class__.__name__)
        self.min_portfolio_size = min_portfolio_size
        self.thresholds = PORTFOLIO_THRESHOLDS

    def detect_portfolios(self, df: pd.DataFrame,
                         owner_col: str = 'normalized_owner',
                         fallback_col: str = 'owner_name') -> pd.DataFrame:
        """
        Identify owners with multiple properties.

        Args:
            df: DataFrame with property records
            owner_col: Column to group by (normalized preferred)
            fallback_col: Fallback if normalized not available

        Returns:
            DataFrame with portfolio_size added to each property
        """
        self.logger.info(f"Detecting portfolios from {len(df):,} properties...")

        df = df.copy()

        # Use normalized owner if available, else fallback
        group_col = owner_col if owner_col in df.columns else fallback_col

        if group_col not in df.columns:
            self.logger.error(f"Neither {owner_col} nor {fallback_col} found in DataFrame")
            df['portfolio_size'] = 1
            return df

        # Count properties per owner
        owner_counts = df[group_col].value_counts()

        # Map back to dataframe
        df['portfolio_size'] = df[group_col].map(owner_counts).fillna(1).astype(int)

        # Classify portfolio size
        df['portfolio_category'] = df['portfolio_size'].apply(self._categorize_portfolio)

        # Flag portfolio investors
        df['is_portfolio_investor'] = df['portfolio_size'] >= self.min_portfolio_size

        # Log summary
        portfolio_investors = df[df['is_portfolio_investor']][group_col].nunique()
        total_owners = df[group_col].nunique()

        self.logger.info(f"Portfolio detection complete:")
        self.logger.info(f"  Total unique owners: {total_owners:,}")
        self.logger.info(f"  Portfolio investors ({self.min_portfolio_size}+ properties): {portfolio_investors:,}")

        # Log by category
        for category in ['small', 'medium', 'large', 'institutional']:
            count = (df['portfolio_category'] == category).sum()
            if count > 0:
                self.logger.info(f"  {category.title()} portfolios: {count:,} properties")

        return df

    def _categorize_portfolio(self, size: int) -> str:
        """Categorize portfolio by size"""
        if size >= self.thresholds['institutional']:
            return 'institutional'
        elif size >= self.thresholds['large']:
            return 'large'
        elif size >= self.thresholds['medium']:
            return 'medium'
        elif size >= self.thresholds['small']:
            return 'small'
        else:
            return 'single'

    def aggregate_by_owner(self, df: pd.DataFrame,
                          owner_col: str = 'normalized_owner',
                          fallback_col: str = 'owner_name') -> pd.DataFrame:
        """
        Aggregate properties by owner to create investor profiles.

        Returns one row per unique owner with:
        - Portfolio size
        - Total value
        - Property list
        - Address info
        """
        self.logger.info("Aggregating properties by owner...")

        # Use normalized owner if available
        group_col = owner_col if owner_col in df.columns else fallback_col

        if group_col not in df.columns:
            self.logger.error(f"Owner column not found")
            return pd.DataFrame()

        # Define aggregation
        agg_dict = {
            'parcel_id': 'count',  # Portfolio size
        }

        # Add value columns if present
        if 'market_value' in df.columns:
            agg_dict['market_value'] = 'sum'
        if 'assessed_value' in df.columns:
            agg_dict['assessed_value'] = 'sum'

        # Add address (take first)
        if 'owner_address' in df.columns:
            agg_dict['owner_address'] = 'first'
        if 'mailing_address' in df.columns:
            agg_dict['mailing_address'] = 'first'

        # Entity type (take first - should be same for same owner)
        if 'entity_type' in df.columns:
            agg_dict['entity_type'] = 'first'
        if 'is_investor_entity' in df.columns:
            agg_dict['is_investor_entity'] = 'first'
        if 'has_investor_name' in df.columns:
            agg_dict['has_investor_name'] = 'first'
        if 'has_po_box' in df.columns:
            agg_dict['has_po_box'] = 'first'
        if 'has_business_address' in df.columns:
            agg_dict['has_business_address'] = 'first'

        # Aggregate
        owner_df = df.groupby(group_col).agg(agg_dict).reset_index()

        # Rename columns
        owner_df = owner_df.rename(columns={
            group_col: 'owner_name',
            'parcel_id': 'portfolio_size',
            'market_value': 'total_market_value',
            'assessed_value': 'total_assessed_value',
        })

        # Add portfolio category
        owner_df['portfolio_category'] = owner_df['portfolio_size'].apply(self._categorize_portfolio)

        # Flag portfolio investors
        owner_df['is_portfolio_investor'] = owner_df['portfolio_size'] >= self.min_portfolio_size

        # Get sample properties for each owner (up to 5)
        property_samples = df.groupby(group_col).apply(
            lambda x: x['address'].head(5).tolist() if 'address' in x.columns else []
        ).reset_index()
        property_samples.columns = ['owner_name', 'sample_properties']

        owner_df = owner_df.merge(property_samples, on='owner_name', how='left')

        # Sort by portfolio size descending
        owner_df = owner_df.sort_values('portfolio_size', ascending=False)

        self.logger.info(f"Aggregated {len(owner_df):,} unique owners")

        return owner_df

    def get_top_investors(self, df: pd.DataFrame,
                         top_n: int = 100,
                         min_properties: int = 2) -> pd.DataFrame:
        """
        Get the top N investors by portfolio size.

        Args:
            df: DataFrame (either property-level or owner-aggregated)
            top_n: Number of top investors to return
            min_properties: Minimum portfolio size to include

        Returns:
            DataFrame of top investors
        """
        # If this is property-level data, aggregate first
        if 'portfolio_size' not in df.columns or 'owner_name' in df.columns and df['owner_name'].duplicated().any():
            df = self.aggregate_by_owner(df)

        # Filter and sort
        investors = df[df['portfolio_size'] >= min_properties].copy()
        investors = investors.nlargest(top_n, 'portfolio_size')

        return investors

    def detect_recent_buyers(self, df: pd.DataFrame,
                            date_col: str = 'last_sale_date',
                            months: int = 24) -> pd.DataFrame:
        """
        Flag owners who have purchased recently (active buyers).

        Args:
            df: DataFrame with sale date information
            date_col: Column containing last sale date
            months: Consider purchases within this many months as "recent"

        Returns:
            DataFrame with recent_buyer flag added
        """
        df = df.copy()

        if date_col not in df.columns:
            self.logger.warning(f"Sale date column '{date_col}' not found")
            df['recent_buyer'] = False
            df['last_purchase_date'] = None
            return df

        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Calculate cutoff date
        cutoff = datetime.now() - timedelta(days=months * 30)

        # Flag recent purchases
        df['recent_buyer'] = df[date_col] >= cutoff
        df['last_purchase_date'] = df[date_col]

        recent_count = df['recent_buyer'].sum()
        self.logger.info(f"Found {recent_count:,} properties purchased in last {months} months")

        return df

    def get_portfolio_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics about portfolios in the dataset.
        """
        if 'portfolio_size' not in df.columns:
            df = self.detect_portfolios(df)

        # Get owner-level stats
        owner_df = self.aggregate_by_owner(df)

        stats = {
            'total_properties': len(df),
            'total_owners': len(owner_df),
            'portfolio_investors': int(owner_df['is_portfolio_investor'].sum()),
            'single_property_owners': int((owner_df['portfolio_size'] == 1).sum()),
            'avg_portfolio_size': float(owner_df['portfolio_size'].mean()),
            'max_portfolio_size': int(owner_df['portfolio_size'].max()),
            'portfolio_distribution': {
                'single': int((owner_df['portfolio_category'] == 'single').sum()),
                'small': int((owner_df['portfolio_category'] == 'small').sum()),
                'medium': int((owner_df['portfolio_category'] == 'medium').sum()),
                'large': int((owner_df['portfolio_category'] == 'large').sum()),
                'institutional': int((owner_df['portfolio_category'] == 'institutional').sum()),
            }
        }

        if 'total_market_value' in owner_df.columns:
            stats['total_portfolio_value'] = float(owner_df['total_market_value'].sum())
            stats['avg_portfolio_value'] = float(owner_df['total_market_value'].mean())

        return stats


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Quick test of portfolio detector"""

    # Create sample data
    sample_data = pd.DataFrame({
        'parcel_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
        'owner_name': [
            'ABC PROPERTIES LLC',
            'ABC PROPERTIES LLC',
            'ABC PROPERTIES LLC',
            'JOHN SMITH',
            'XYZ HOLDINGS INC',
            'XYZ HOLDINGS INC',
            'JANE DOE',
            'JANE DOE',
        ],
        'normalized_owner': [
            'ABC PROPERTIES LLC',
            'ABC PROPERTIES LLC',
            'ABC PROPERTIES LLC',
            'JOHN SMITH',
            'XYZ HOLDINGS INC',
            'XYZ HOLDINGS INC',
            'JANE DOE',
            'JANE DOE',
        ],
        'address': [
            '123 Main St', '456 Oak Ave', '789 Pine Rd',
            '100 Elm St', '200 Maple Dr', '300 Cedar Ln',
            '400 Birch Way', '500 Walnut St'
        ],
        'market_value': [150000, 200000, 175000, 250000, 500000, 450000, 180000, 220000],
    })

    detector = PortfolioDetector(min_portfolio_size=2)

    # Detect portfolios
    df_with_portfolios = detector.detect_portfolios(sample_data)
    print("\nProperty-level portfolio detection:")
    print(df_with_portfolios[['owner_name', 'portfolio_size', 'portfolio_category', 'is_portfolio_investor']])

    # Aggregate by owner
    owner_profiles = detector.aggregate_by_owner(sample_data)
    print("\nOwner-level aggregation:")
    print(owner_profiles[['owner_name', 'portfolio_size', 'total_market_value', 'portfolio_category']])

    # Get stats
    stats = detector.get_portfolio_stats(sample_data)
    print("\nPortfolio Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
