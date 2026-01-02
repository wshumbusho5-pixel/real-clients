"""
Aerial Leads - Data Aggregator

Combines data from multiple sources:
1. Franklin County tax data
2. Code violations
3. Skip tracing
4. Motivation scoring

Produces final enriched lead dataset
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from scrapers.franklin_county import FranklinCountyScraper
from scrapers.code_violations import CodeViolationsScraper, create_sample_violations
from skip_tracing.batch_skip_trace import BatchSkipTracer
from scoring.motivation_scorer import MotivationScorer

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, ENABLE_CODE_VIOLATIONS_SCRAPER
from config.logging_config import log_success, log_failure, get_logger


class LeadAggregator:
    """
    Aggregates data from all sources to create enriched leads
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.tax_scraper = FranklinCountyScraper()
        self.skip_tracer = BatchSkipTracer()
        self.scorer = MotivationScorer()

        # Violations scraper is optional (complex setup)
        if ENABLE_CODE_VIOLATIONS_SCRAPER:
            try:
                self.violations_scraper = CodeViolationsScraper()
            except:
                self.logger.warning("Could not initialize violations scraper - will use mock data")
                self.violations_scraper = None
        else:
            self.violations_scraper = None

    def generate_leads(
        self,
        max_properties: int = 200,
        min_years_delinquent: int = 2,
        min_motivation_score: int = 40,
        zip_codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Full lead generation pipeline

        Steps:
        1. Scrape tax delinquent properties
        2. Scrape code violations
        3. Skip trace for contact info
        4. Calculate motivation scores
        5. Filter by minimum score

        Args:
            max_properties: Maximum properties to process
            min_years_delinquent: Minimum years tax delinquent
            min_motivation_score: Minimum motivation score to include
            zip_codes: List of zip codes (None = all Columbus)

        Returns:
            DataFrame with enriched leads
        """
        self.logger.info("🚀 Starting full lead generation pipeline...")

        # Step 1: Scrape tax data
        self.logger.info("Step 1/4: Scraping tax delinquent properties...")
        properties = self.tax_scraper.scrape(
            delinquent_only=True,
            min_years_delinquent=min_years_delinquent,
            max_results=max_properties,
            zip_codes=zip_codes
        )

        if not properties:
            log_failure("No tax delinquent properties found")
            return pd.DataFrame()

        self.logger.info(f"Found {len(properties)} tax delinquent properties")

        # Step 2: Enrich with code violations
        self.logger.info("Step 2/4: Enriching with code violations...")
        properties = self._add_code_violations(properties)

        # Step 3: Skip trace for contact info
        self.logger.info("Step 3/4: Skip tracing for contact information...")
        properties = self.skip_tracer.skip_trace_batch(properties)

        # Step 4: Calculate motivation scores
        self.logger.info("Step 4/4: Calculating motivation scores...")
        properties = self.scorer.score_batch(properties)

        # Convert to DataFrame
        df = pd.DataFrame(properties)

        # Filter by minimum score
        df = df[df['motivation_score'] >= min_motivation_score]

        # Sort by score (highest first)
        df = df.sort_values('motivation_score', ascending=False)

        log_success(f"Generated {len(df)} premium leads (score >= {min_motivation_score})")

        return df

    def _add_code_violations(self, properties: List[Dict]) -> List[Dict]:
        """
        Add code violation data to properties

        Args:
            properties: List of property dictionaries

        Returns:
            Properties enriched with violation counts
        """
        if self.violations_scraper:
            # Real scraping
            addresses = [p['address'] for p in properties]

            try:
                violations = self.violations_scraper.scrape(addresses)
                aggregated = self.violations_scraper.aggregate_violations(violations)

                # Add violation counts to properties
                for prop in properties:
                    addr = prop['address']
                    if addr in aggregated:
                        prop['code_violations'] = aggregated[addr]['total_violations']
                        prop['critical_violations'] = aggregated[addr]['critical_violations']
                        prop['open_violations'] = aggregated[addr]['open_violations']
                    else:
                        prop['code_violations'] = 0
                        prop['critical_violations'] = 0
                        prop['open_violations'] = 0

            except Exception as e:
                self.logger.error(f"Error scraping violations: {e}")
                self.logger.warning("Using mock violation data")
                self._add_mock_violations(properties)
        else:
            # Mock data
            self._add_mock_violations(properties)

        return properties

    def _add_mock_violations(self, properties: List[Dict]):
        """Add mock violation data for testing"""
        import random

        for prop in properties:
            # Random violation count (weighted toward 0-2)
            num_violations = random.choices(
                [0, 1, 2, 3, 4],
                weights=[40, 30, 15, 10, 5]
            )[0]

            prop['code_violations'] = num_violations
            prop['critical_violations'] = min(num_violations, random.randint(0, 1))
            prop['open_violations'] = random.randint(0, num_violations)

    def load_and_enrich(self, csv_path: str) -> pd.DataFrame:
        """
        Load existing CSV and enrich with missing data

        Useful for enriching data you already scraped

        Args:
            csv_path: Path to CSV file with property data

        Returns:
            Enriched DataFrame
        """
        self.logger.info(f"Loading data from {csv_path}...")

        df = pd.read_csv(csv_path)
        properties = df.to_dict('records')

        # Check what's missing and enrich
        if 'phone' not in df.columns or df['phone'].isna().all():
            self.logger.info("Skip tracing for contact info...")
            properties = self.skip_tracer.skip_trace_batch(properties)

        if 'code_violations' not in df.columns:
            self.logger.info("Adding code violations...")
            properties = self._add_code_violations(properties)

        if 'motivation_score' not in df.columns:
            self.logger.info("Calculating motivation scores...")
            properties = self.scorer.score_batch(properties)

        return pd.DataFrame(properties)

    def export_leads(
        self,
        df: pd.DataFrame,
        filename: str = 'enriched_leads.csv',
        tier_filter: Optional[int] = None
    ) -> str:
        """
        Export leads to CSV

        Args:
            df: DataFrame with leads
            filename: Output filename
            tier_filter: Only export specific tier (1, 2, 3, or None for all)

        Returns:
            Path to exported file
        """
        if df.empty:
            log_failure("No leads to export")
            return ''

        # Filter by tier if specified
        if tier_filter:
            df = df[df['tier'] == tier_filter]
            filename = filename.replace('.csv', f'_tier{tier_filter}.csv')

        # Ensure directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        output_path = PROCESSED_DATA_DIR / filename
        df.to_csv(output_path, index=False)

        log_success(f"Exported {len(df)} leads to {output_path}")
        return str(output_path)

    def export_by_tier(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Export separate files for each tier

        Args:
            df: DataFrame with leads

        Returns:
            Dictionary mapping tier number to file path
        """
        tier_files = {}

        for tier in [1, 2, 3, 4]:
            tier_df = df[df['tier'] == tier]

            if not tier_df.empty:
                filename = f'tier_{tier}_leads.csv'
                path = self.export_leads(tier_df, filename)
                tier_files[tier] = path

                self.logger.info(f"Tier {tier}: {len(tier_df)} leads exported")

        return tier_files

    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the lead dataset

        Args:
            df: DataFrame with leads

        Returns:
            Dictionary with summary stats
        """
        stats = {
            'total_leads': len(df),
            'tier_1_count': len(df[df['tier'] == 1]),
            'tier_2_count': len(df[df['tier'] == 2]),
            'tier_3_count': len(df[df['tier'] == 3]),
            'tier_4_count': len(df[df['tier'] == 4]),
            'avg_motivation_score': df['motivation_score'].mean(),
            'avg_taxes_owed': df['taxes_owed'].mean(),
            'avg_years_delinquent': df['years_delinquent'].mean(),
            'total_potential_value': df['assessed_value'].sum(),
            'total_taxes_owed': df['taxes_owed'].sum(),
            'skip_traced_count': len(df[df.get('skip_traced', False) == True]),
            'phone_available_count': len(df[df['phone'].notna() & (df['phone'] != '')]),
        }

        return stats


# Example usage
if __name__ == '__main__':
    aggregator = LeadAggregator()

    # Generate leads
    leads_df = aggregator.generate_leads(
        max_properties=100,
        min_years_delinquent=2,
        min_motivation_score=50
    )

    # Export
    aggregator.export_leads(leads_df, 'premium_leads.csv')

    # Export by tier
    aggregator.export_by_tier(leads_df)

    # Show stats
    stats = aggregator.generate_summary_stats(leads_df)
    print("\n📊 Lead Generation Summary:")
    print(f"Total leads: {stats['total_leads']}")
    print(f"Tier 1 (80-100): {stats['tier_1_count']}")
    print(f"Tier 2 (60-79): {stats['tier_2_count']}")
    print(f"Tier 3 (40-59): {stats['tier_3_count']}")
    print(f"Average score: {stats['avg_motivation_score']:.1f}")
