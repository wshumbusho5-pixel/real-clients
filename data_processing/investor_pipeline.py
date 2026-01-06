"""
Investor Pipeline - Main orchestration for investor identification

Coordinates:
1. Data loading from Franklin County Excel files
2. Entity classification
3. Portfolio detection
4. Investor scoring
5. Export of investor prospects
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, COUNTY_CONFIG, DEFAULT_COUNTY
from config.logging_config import get_logger, log_success, log_failure
from scrapers.franklin_county_excel import FranklinCountyExcelLoader
from investor_identification import EntityClassifier, PortfolioDetector, InvestorScorer


class InvestorPipeline:
    """
    Main pipeline for identifying real estate investors from property records.

    Usage:
        pipeline = InvestorPipeline(county='franklin')  # Columbus
        pipeline = InvestorPipeline(county='hamilton')  # Cincinnati
        investors = pipeline.run()
        pipeline.export_investors(investors)
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR, county: str = DEFAULT_COUNTY):
        self.logger = get_logger(self.__class__.__name__)
        self.data_dir = data_dir
        self.county = county.lower()

        # Validate county
        if self.county not in COUNTY_CONFIG:
            raise ValueError(f"Unknown county: {county}. Available: {list(COUNTY_CONFIG.keys())}")

        self.county_config = COUNTY_CONFIG[self.county]
        self.logger.info(f"Initialized pipeline for {self.county_config['name']} ({self.county_config['city']})")

        # Initialize components
        self.loader = FranklinCountyExcelLoader(data_dir)
        self.classifier = EntityClassifier()
        self.portfolio_detector = PortfolioDetector()
        self.scorer = InvestorScorer()

    def run(self,
            filter_city: bool = True,
            min_portfolio_size: int = 2,
            min_score: int = 40,
            max_results: Optional[int] = None) -> pd.DataFrame:
        """
        Run the full investor identification pipeline.

        Args:
            filter_city: Filter to city zip codes only (based on county config)
            min_portfolio_size: Minimum properties to be considered portfolio investor
            min_score: Minimum investor score to include in results
            max_results: Maximum investors to return (None = all)

        Returns:
            DataFrame of identified investors with scores
        """
        self.logger.info("=" * 60)
        self.logger.info(f"INVESTOR FINDER PIPELINE - {self.county_config['city'].upper()}")
        self.logger.info("=" * 60)

        # Step 1: Load all property data
        self.logger.info("\n[Step 1/5] Loading property data...")
        df = self._load_all_properties(filter_city)

        if df.empty:
            log_failure("No property data loaded")
            return pd.DataFrame()

        self.logger.info(f"Loaded {len(df):,} properties")

        # Step 2: Classify entities
        self.logger.info("\n[Step 2/5] Classifying owner entities...")
        df = self.classifier.classify_dataframe(df)

        # Step 3: Detect portfolios
        self.logger.info("\n[Step 3/5] Detecting portfolio investors...")
        df = self.portfolio_detector.detect_portfolios(df)

        # Step 4: Aggregate by owner
        self.logger.info("\n[Step 4/5] Aggregating by owner...")
        investors_df = self.portfolio_detector.aggregate_by_owner(df)

        # Add absentee ratio for each owner
        investors_df = self._calculate_absentee_ratio(df, investors_df)

        # Step 5: Score investors
        self.logger.info("\n[Step 5/5] Scoring investors...")
        investors_df = self.scorer.score_dataframe(investors_df)

        # Filter by minimum score
        investors_df = investors_df[investors_df['investor_score'] >= min_score]

        # Optionally filter by portfolio size
        if min_portfolio_size > 1:
            investors_df = investors_df[investors_df['portfolio_size'] >= min_portfolio_size]

        # Limit results if specified
        if max_results:
            investors_df = investors_df.head(max_results)

        self.logger.info("\n" + "=" * 60)
        log_success(f"Identified {len(investors_df):,} investor prospects")
        self.logger.info("=" * 60)

        return investors_df

    def _load_all_properties(self, filter_city: bool = True) -> pd.DataFrame:
        """
        Load all properties (not just tax delinquent).

        We want to identify ALL investors, not just those with distressed properties.
        """
        self.logger.info(f"Loading property data for {self.county_config['name']}...")

        try:
            # Load parcel data (all properties) - use county-specific file
            parcel_filename = self.county_config['data_files']['parcel']
            parcel_file = self.data_dir / parcel_filename
            if not parcel_file.exists():
                log_failure(f"{parcel_filename} not found at {parcel_file}")
                return pd.DataFrame()

            df = pd.read_excel(parcel_file, engine='openpyxl')
            self.logger.info(f"Loaded {len(df):,} parcels from {parcel_filename}")

            # Select and rename columns (flexible mapping for different county formats)
            columns_to_keep = {
                'PARCEL ID': 'parcel_id',
                'Parcel ID': 'parcel_id',
                'ParcelID': 'parcel_id',
                'SiteAddress': 'address',
                'Site Address': 'address',
                'PropertyAddress': 'address',
                'OwnerName1': 'owner_name',
                'Owner Name': 'owner_name',
                'OwnerName': 'owner_name',
                'OwnerName2': 'owner_name_2',
                'OwnerAddress1': 'owner_address',
                'Owner Address': 'owner_address',
                'TaxpayerAddress1': 'mailing_address',
                'Mailing Address': 'mailing_address',
                'ZipCode': 'zip_code',
                'Zip': 'zip_code',
                'ZIP': 'zip_code',
                'LUCDesc': 'property_type',
                'Property Type': 'property_type',
                'Neighborhood': 'neighborhood',
            }

            # Keep only columns that exist (first match wins)
            available_cols = {}
            used_targets = set()
            for source, target in columns_to_keep.items():
                if source in df.columns and target not in used_targets:
                    available_cols[source] = target
                    used_targets.add(target)

            df = df[list(available_cols.keys())].copy()
            df.columns = [available_cols[c] for c in df.columns]

            # Clean zip codes
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].fillna(0).astype(int).astype(str)
                df['zip_code'] = df['zip_code'].replace('0', '')

            # Filter to city zip codes if requested
            if filter_city and 'zip_code' in df.columns:
                city_zips = self.county_config['zip_codes']
                df = df[df['zip_code'].isin(city_zips)]
                self.logger.info(f"Filtered to {self.county_config['city']}: {len(df):,} properties")

            # Load value data for market values - use county-specific file
            value_filename = self.county_config['data_files']['value']
            value_file = self.data_dir / value_filename
            if value_file.exists():
                df_value = pd.read_excel(value_file, engine='openpyxl')
                # Flexible column mapping for value file
                value_cols = df_value.columns.tolist()
                parcel_col = next((c for c in value_cols if 'parcel' in c.lower()), None)
                land_col = next((c for c in value_cols if 'land' in c.lower() and 'market' in c.lower()), None)
                impr_col = next((c for c in value_cols if 'impr' in c.lower() and 'market' in c.lower()), None)

                if parcel_col and (land_col or impr_col):
                    land_val = df_value[land_col].fillna(0) if land_col else 0
                    impr_val = df_value[impr_col].fillna(0) if impr_col else 0
                    df_value['market_value'] = land_val + impr_val
                    df_value = df_value[[parcel_col, 'market_value']].copy()
                    df_value.columns = ['parcel_id', 'market_value']
                    df_value = df_value.drop_duplicates(subset=['parcel_id'], keep='last')

                    df = df.merge(df_value, on='parcel_id', how='left')
                    df['market_value'] = df['market_value'].fillna(0)
                    self.logger.info("Added market values")

            # Calculate absentee status
            if 'mailing_address' in df.columns and 'owner_address' in df.columns:
                df['is_absentee'] = df['mailing_address'] != df['owner_address']
                df['is_absentee'] = df['is_absentee'].fillna(False)

            # Add county info to dataframe
            df['county'] = self.county
            df['city'] = self.county_config['city']

            return df

        except Exception as e:
            log_failure(f"Error loading property data: {e}")
            return pd.DataFrame()

    def _calculate_absentee_ratio(self, properties_df: pd.DataFrame,
                                  investors_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate absentee ownership ratio for each investor"""
        if 'is_absentee' not in properties_df.columns:
            investors_df['absentee_ratio'] = 0
            return investors_df

        # Group by normalized owner and calculate absentee ratio
        owner_col = 'normalized_owner' if 'normalized_owner' in properties_df.columns else 'owner_name'

        absentee_stats = properties_df.groupby(owner_col).agg({
            'is_absentee': 'mean'
        }).reset_index()
        absentee_stats.columns = ['owner_name', 'absentee_ratio']

        investors_df = investors_df.merge(absentee_stats, on='owner_name', how='left')
        investors_df['absentee_ratio'] = investors_df['absentee_ratio'].fillna(0)

        return investors_df

    def export_investors(self,
                        df: pd.DataFrame,
                        filename: str = None,
                        by_tier: bool = True) -> List[str]:
        """
        Export investor prospects to CSV.

        Args:
            df: DataFrame of investor prospects
            filename: Base filename for export (default: county-specific name)
            by_tier: Also export separate files by tier

        Returns:
            List of exported file paths
        """
        if df.empty:
            log_failure("No investors to export")
            return []

        # Default filename includes county
        if filename is None:
            filename = f'investor_prospects_{self.county}.csv'

        exported_files = []

        # Ensure output directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Export all investors
        all_path = PROCESSED_DATA_DIR / filename
        df.to_csv(all_path, index=False)
        exported_files.append(str(all_path))
        log_success(f"Exported {len(df):,} {self.county_config['city']} investors to {all_path}")

        # Export by tier
        if by_tier:
            for tier in ['tier_1', 'tier_2', 'tier_3']:
                tier_df = df[df['investor_tier'] == tier]
                if not tier_df.empty:
                    tier_filename = filename.replace('.csv', f'_{tier}.csv')
                    tier_path = PROCESSED_DATA_DIR / tier_filename
                    tier_df.to_csv(tier_path, index=False)
                    exported_files.append(str(tier_path))
                    self.logger.info(f"Exported {len(tier_df):,} {tier} investors to {tier_path}")

        return exported_files

    def get_pipeline_stats(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive pipeline statistics"""
        stats = {
            'run_date': datetime.now().isoformat(),
            'total_investors': len(df),
        }

        if not df.empty:
            stats.update({
                'avg_score': float(df['investor_score'].mean()),
                'avg_portfolio_size': float(df['portfolio_size'].mean()),
                'total_properties_owned': int(df['portfolio_size'].sum()),
                'tier_distribution': df['investor_tier'].value_counts().to_dict(),
                'entity_distribution': df['entity_type'].value_counts().to_dict() if 'entity_type' in df.columns else {},
                'portfolio_distribution': df['portfolio_category'].value_counts().to_dict() if 'portfolio_category' in df.columns else {},
            })

            if 'total_market_value' in df.columns:
                stats['total_portfolio_value'] = float(df['total_market_value'].sum())

        return stats

    def print_summary(self, df: pd.DataFrame):
        """Print a summary of identified investors"""
        print("\n" + "=" * 70)
        print("INVESTOR FINDER - RESULTS SUMMARY")
        print("=" * 70)

        if df.empty:
            print("No investors identified.")
            return

        print(f"\nTotal Investor Prospects: {len(df):,}")

        # Tier breakdown
        print("\nBy Tier:")
        tier_counts = df['investor_tier'].value_counts().sort_index()
        for tier, count in tier_counts.items():
            desc = self.scorer.get_tier_description(tier)
            print(f"  {tier}: {count:,} - {desc}")

        # Portfolio breakdown
        if 'portfolio_category' in df.columns:
            print("\nBy Portfolio Size:")
            portfolio_counts = df['portfolio_category'].value_counts()
            for cat, count in portfolio_counts.items():
                print(f"  {cat.title()}: {count:,}")

        # Entity type breakdown
        if 'entity_type' in df.columns:
            print("\nBy Entity Type:")
            entity_counts = df['entity_type'].value_counts()
            for entity, count in entity_counts.items():
                print(f"  {entity.title()}: {count:,}")

        # Top 10 investors
        print("\nTop 10 Investors by Score:")
        print("-" * 70)
        top_10 = df.nlargest(10, 'investor_score')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            props = row.get('portfolio_size', 'N/A')
            score = row.get('investor_score', 0)
            entity = row.get('entity_type', 'unknown')
            print(f"{i:2}. {row['owner_name'][:45]:45} | {props:3} props | Score: {score:3} | {entity}")

        print("\n" + "=" * 70)


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Run the investor identification pipeline"""

    pipeline = InvestorPipeline()

    # Run pipeline
    investors = pipeline.run(
        columbus_only=True,
        min_portfolio_size=2,
        min_score=40
    )

    # Print summary
    pipeline.print_summary(investors)

    # Export results
    if not investors.empty:
        exported = pipeline.export_investors(investors)
        print(f"\nExported to: {exported}")
