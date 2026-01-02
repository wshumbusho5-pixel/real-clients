"""
Aerial Leads - Franklin County Excel Data Loader

Loads tax delinquent properties from Franklin County Auditor's bulk Excel files.
This is MUCH faster and more reliable than web scraping.

Data Source: https://apps.franklincountyauditor.com/Outside_User_Files/
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from config.settings import RAW_DATA_DIR

# Default for tax delinquency filtering (not used in investor finder)
MIN_YEARS_DELINQUENT = 2
from config.logging_config import log_success, log_failure, get_logger


class FranklinCountyExcelLoader:
    """
    Load tax delinquent properties from Franklin County Excel files

    Files needed (download from FTP):
    1. TaxDetail.xlsx - Tax payment details
    2. Parcel.xlsx - Property addresses and owner info
    3. Value.xlsx - Assessed values
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.logger = get_logger(self.__class__.__name__)
        self.data_dir = data_dir

        # File paths
        self.tax_detail_file = data_dir / 'TaxDetail.xlsx'
        self.parcel_file = data_dir / 'Parcel.xlsx'
        self.value_file = data_dir / 'Value.xlsx'

    def load_tax_delinquent_properties(
        self,
        min_amount_owed: float = 1000.0,
        min_years_delinquent: int = MIN_YEARS_DELINQUENT,
        columbus_only: bool = True,
        max_properties: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load tax delinquent properties from Excel files

        Args:
            min_amount_owed: Minimum tax debt (default $1000)
            min_years_delinquent: Minimum years behind on taxes
            columbus_only: Filter to Columbus zip codes only
            max_properties: Maximum properties to return (None = all)

        Returns:
            DataFrame with tax delinquent properties
        """
        self.logger.info("🔄 Loading Franklin County tax data from Excel files...")

        # Step 1: Load tax detail (find delinquent properties)
        self.logger.info("📊 Loading TaxDetail.xlsx...")
        df_tax = self._load_tax_details()

        # Step 2: Load parcel data (addresses, owners)
        self.logger.info("🏠 Loading Parcel.xlsx...")
        df_parcel = self._load_parcels()

        # Step 3: Load values (assessed values)
        self.logger.info("💰 Loading Value.xlsx...")
        df_value = self._load_values()

        # Step 4: Join datasets
        self.logger.info("🔗 Joining datasets...")
        df_merged = self._merge_datasets(df_tax, df_parcel, df_value)

        # Step 5: Filter for delinquent properties
        self.logger.info("🎯 Filtering for tax delinquent properties...")
        df_delinquent = df_merged[df_merged['taxes_owed'] >= min_amount_owed].copy()

        # Step 6: Filter for Columbus if requested
        if columbus_only:
            columbus_zips = self._get_columbus_zip_codes()
            df_delinquent = df_delinquent[
                df_delinquent['zip_code'].isin(columbus_zips)
            ]
            self.logger.info(f"📍 Filtered to Columbus zip codes: {len(df_delinquent)} properties")

        # Step 7: Calculate years delinquent (estimate based on amount owed)
        df_delinquent['years_delinquent'] = self._estimate_years_delinquent(df_delinquent)

        # Step 8: Filter by minimum years
        df_delinquent = df_delinquent[
            df_delinquent['years_delinquent'] >= min_years_delinquent
        ]

        # Step 9: Limit results if requested
        if max_properties:
            df_delinquent = df_delinquent.head(max_properties)

        # Step 10: Clean and format
        df_final = self._format_for_export(df_delinquent)

        log_success(f"✅ Loaded {len(df_final)} tax delinquent properties")

        return df_final

    def _load_tax_details(self) -> pd.DataFrame:
        """Load tax detail file and extract total owed"""
        if not self.tax_detail_file.exists():
            raise FileNotFoundError(f"TaxDetail.xlsx not found at {self.tax_detail_file}")

        # Read Excel file
        df = pd.read_excel(self.tax_detail_file, engine='openpyxl')

        # Key columns: Parcel Id, TotTotal (total amount owed)
        df = df[['Parcel Id', 'TotTotal']].copy()
        df.columns = ['parcel_id', 'taxes_owed']

        # Remove nulls and zeros
        df = df[df['taxes_owed'].notna()]
        df = df[df['taxes_owed'] > 0]

        self.logger.info(f"Found {len(df)} parcels with tax debt")

        return df

    def _load_parcels(self) -> pd.DataFrame:
        """Load parcel file and extract property details"""
        if not self.parcel_file.exists():
            raise FileNotFoundError(f"Parcel.xlsx not found at {self.parcel_file}")

        # Read Excel file
        df = pd.read_excel(self.parcel_file, engine='openpyxl')

        # Key columns
        columns_to_keep = [
            'PARCEL ID',
            'SiteAddress',
            'OwnerName1',
            'OwnerAddress1',
            'TaxpayerAddress1',
            'ZipCode',
            'TaxYear',
            'Neighborhood',
            'LUCDesc',  # Land Use Code Description (property type)
        ]

        df = df[columns_to_keep].copy()
        df.columns = [
            'parcel_id',
            'address',
            'owner_name',
            'owner_address',
            'mailing_address',
            'zip_code',
            'tax_year',
            'neighborhood',
            'property_type'
        ]

        self.logger.info(f"Loaded {len(df)} parcel records")

        return df

    def _load_values(self) -> pd.DataFrame:
        """Load value file and extract assessed values"""
        if not self.value_file.exists():
            raise FileNotFoundError(f"Value.xlsx not found at {self.value_file}")

        # Read Excel file
        df = pd.read_excel(self.value_file, engine='openpyxl')

        # Calculate total market value (land + improvements)
        df['market_value'] = df['MarketLand'].fillna(0) + df['MarketImpr'].fillna(0)
        df['taxable_value'] = df['TaxableLand'].fillna(0) + df['TaxableImpr'].fillna(0)

        # Keep only necessary columns
        df = df[['Parcel Id', 'market_value', 'taxable_value']].copy()
        df.columns = ['parcel_id', 'market_value', 'assessed_value']

        # Remove duplicates (keep most recent)
        df = df.drop_duplicates(subset=['parcel_id'], keep='last')

        self.logger.info(f"Loaded {len(df)} value records")

        return df

    def _merge_datasets(
        self,
        df_tax: pd.DataFrame,
        df_parcel: pd.DataFrame,
        df_value: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge all datasets on parcel_id"""

        # Start with tax data (only properties with debt)
        df = df_tax.copy()

        # Merge parcel data
        df = df.merge(df_parcel, on='parcel_id', how='left')

        # Merge value data
        df = df.merge(df_value, on='parcel_id', how='left')

        # Fill missing values
        df['assessed_value'] = df['assessed_value'].fillna(0)
        df['market_value'] = df['market_value'].fillna(0)

        # Convert zip code to string for comparison
        df['zip_code'] = df['zip_code'].fillna(0).astype(int).astype(str)
        df['zip_code'] = df['zip_code'].replace('0', '')

        self.logger.info(f"Merged datasets: {len(df)} properties")

        return df

    def _estimate_years_delinquent(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate years delinquent based on tax amount owed

        This is a rough estimate. A more accurate method would require
        parsing payment history, but this works for initial scoring.

        Logic:
        - Average annual property tax in Columbus: ~$3,000-$5,000
        - Assume: years_delinquent = taxes_owed / (assessed_value * 0.04)
        - Cap at 10 years max
        """
        # Calculate estimated annual tax (4% of assessed value is typical)
        estimated_annual_tax = df['assessed_value'] * 0.04

        # Avoid division by zero
        estimated_annual_tax = estimated_annual_tax.replace(0, 3000)

        # Calculate years delinquent
        years = (df['taxes_owed'] / estimated_annual_tax).fillna(0)

        # Round and cap between 1 and 10
        years = years.clip(lower=1, upper=10).round().astype(int)

        return years

    def _get_columbus_zip_codes(self) -> List[str]:
        """Get Columbus, Ohio zip codes"""
        return [
            '43201', '43202', '43203', '43204', '43205',
            '43206', '43207', '43209', '43210', '43211',
            '43212', '43213', '43214', '43215', '43219',
            '43220', '43221', '43222', '43223', '43224',
            '43227', '43229', '43230', '43231', '43232',
            '43235', '43240'
        ]

    def _format_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format data to match expected schema"""

        # Calculate tax debt ratio
        df['tax_debt_ratio'] = np.where(
            df['assessed_value'] > 0,
            df['taxes_owed'] / df['assessed_value'],
            0
        )

        # Detect absentee owners (mailing address != property address)
        df['is_absentee'] = df['mailing_address'] != df['owner_address']
        df['is_absentee'] = df['is_absentee'].fillna(False)

        # Add metadata
        df['data_source'] = 'Franklin County Auditor (Excel)'
        df['scraped_date'] = datetime.now().strftime('%Y-%m-%d')

        # Select and order columns
        columns = [
            'parcel_id',
            'address',
            'owner_name',
            'owner_address',
            'mailing_address',
            'zip_code',
            'assessed_value',
            'market_value',
            'taxes_owed',
            'years_delinquent',
            'tax_debt_ratio',
            'is_absentee',
            'property_type',
            'neighborhood',
            'data_source',
            'scraped_date'
        ]

        return df[columns].copy()

    def export_to_csv(self, df: pd.DataFrame, filename: str = 'franklin_county_tax_data.csv') -> str:
        """
        Export properties to CSV

        Args:
            df: DataFrame to export
            filename: Output filename

        Returns:
            Full path to exported file
        """
        if df.empty:
            log_failure("No properties to export")
            return ''

        # Ensure directory exists
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        output_path = RAW_DATA_DIR / filename
        df.to_csv(output_path, index=False)

        log_success(f"Exported {len(df)} properties to {output_path}")
        return str(output_path)


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Quick test of the loader"""
    loader = FranklinCountyExcelLoader()

    # Load first 100 delinquent properties in Columbus
    df = loader.load_tax_delinquent_properties(
        min_amount_owed=2000,
        min_years_delinquent=2,
        columbus_only=True,
        max_properties=100
    )

    # Export to CSV
    output_file = loader.export_to_csv(df, 'test_tax_delinquent.csv')

    # Show summary
    print(f"\n✅ SUCCESS!")
    print(f"Loaded: {len(df)} properties")
    print(f"Average tax debt: ${df['taxes_owed'].mean():,.0f}")
    print(f"Average years delinquent: {df['years_delinquent'].mean():.1f}")
    print(f"\nTop 5 properties by tax debt:")
    print(df.nlargest(5, 'taxes_owed')[['address', 'owner_name', 'taxes_owed', 'years_delinquent']])
