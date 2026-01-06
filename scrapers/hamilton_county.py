"""
Hamilton County Auditor - Property Data Scraper

Scrapes property/parcel data from Hamilton County (Cincinnati) Auditor.
Uses Playwright to automate the web interface and extract bulk data.

Strategy:
1. Search by owner name (A-Z) to get all parcels via CSV export
2. For parcels needing details, visit detail pages to get owner address + market value
3. Combine into pipeline-compatible Excel file

Website: https://wedge.hcauditor.org/
"""

import re
import time
import json
import csv
import io
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from config.logging_config import get_logger
from config.settings import RAW_DATA_DIR

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class HamiltonParcel:
    """Data class for Hamilton County parcel information"""
    parcel_id: str = ""
    owner_name: str = ""
    owner_address: str = ""
    mailing_address: str = ""
    site_address: str = ""
    city: str = "Cincinnati"
    zip_code: str = ""
    market_value: float = 0.0
    land_value: float = 0.0
    building_value: float = 0.0
    property_type: str = ""
    school_district: str = ""
    appraisal_area: str = ""
    transfer_date: str = ""
    sale_price: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'parcel_id': self.parcel_id,
            'owner_name': self.owner_name,
            'owner_address': self.owner_address,
            'mailing_address': self.mailing_address,
            'site_address': self.site_address,
            'city': self.city,
            'zip_code': self.zip_code,
            'market_value': self.market_value,
            'land_value': self.land_value,
            'building_value': self.building_value,
            'property_type': self.property_type,
            'school_district': self.school_district,
            'appraisal_area': self.appraisal_area,
            'transfer_date': self.transfer_date,
            'sale_price': self.sale_price,
        }


class HamiltonCountyScraper:
    """
    Scrape property data from Hamilton County Auditor.

    Strategy:
    1. Search by owner name (A-Z) and export CSV for each letter
    2. Optionally get parcel details for owner address + market value
    3. Combine into pipeline-compatible Excel file
    """

    SEARCH_URL = "https://wedge.hcauditor.org/"
    DETAIL_URL = "https://wedge.hcauditor.org/view/re/{parcel_id}/"

    # Search patterns (must be 2+ chars - API requirement)
    # Generate all 2-letter combinations (AA-ZZ) plus number prefixes
    SEARCH_PATTERNS = (
        [f"{a}{b}" for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for b in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] +
        [f"{n}" for n in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']]
    )

    def __init__(self, headless: bool = False, delay: float = 2.0):
        self.logger = get_logger(self.__class__.__name__)
        self.headless = headless
        self.delay = delay
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        if not PLAYWRIGHT_AVAILABLE:
            self.logger.error("Playwright not installed. Run: pip install playwright && python -m playwright install firefox")

    def _init_browser(self):
        """Initialize browser with download support"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed")

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.firefox.launch(headless=self.headless)
        self.context = self.browser.new_context(accept_downloads=True)
        self.page = self.context.new_page()
        self.logger.info("Firefox browser initialized")

    def _close_browser(self):
        """Close browser"""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except:
            pass
        self.logger.info("Browser closed")

    def _parse_money(self, text: str) -> float:
        """Parse money string to float"""
        try:
            clean = re.sub(r'[$,]', '', str(text))
            return float(clean) if clean else 0.0
        except:
            return 0.0

    def search_by_letter(self, letter: str) -> List[Dict]:
        """
        Search for all parcels where owner name starts with letter.
        Uses CSV export to get all results.

        Args:
            letter: Starting letter (A-Z, 0-9)

        Returns:
            List of dicts with parcel data
        """
        self.logger.info(f"Searching owners starting with: {letter}")

        try:
            self.page.goto(self.SEARCH_URL, wait_until='domcontentloaded', timeout=30000)
            time.sleep(3)

            # Fill owner name search using JavaScript (more reliable)
            self.page.evaluate(f'''() => {{
                document.querySelector('#owner_name').value = '{letter}';
                document.querySelector('#search_by_owner').submit();
            }}''')

            time.sleep(5)

            # Check for results
            body_text = self.page.inner_text('body')
            if 'No results' in body_text or 'no records' in body_text.lower():
                self.logger.info(f"No results for letter: {letter}")
                return []

            # Find result count
            count_match = re.search(r'of ([\d,]+) entries', body_text)
            result_count = int(count_match.group(1).replace(',', '')) if count_match else 0
            self.logger.info(f"Found {result_count} results for '{letter}'")

            if result_count == 0:
                return []

            # Click CSV export and download
            export_btn = self.page.query_selector('a:has-text("Export Results as CSV"), a:has-text("CSV")')

            if not export_btn:
                self.logger.error("CSV export button not found")
                return []

            # Download the CSV
            with self.page.expect_download(timeout=60000) as download_info:
                export_btn.click()

            download = download_info.value
            csv_content = download.path()

            # Read CSV content
            with open(csv_content, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                parcels = []
                for row in reader:
                    parcels.append({
                        'parcel_id': row.get('Parcel Number', '').strip(),
                        'owner_name': row.get('Name', '').strip(),
                        'site_address': row.get('Address', '').strip(),
                        'transfer_date': row.get('Transfer Date', '').strip(),
                        'sale_price': self._parse_money(row.get('Sale Price', '0')),
                    })

            self.logger.info(f"Extracted {len(parcels)} parcels from CSV")
            return parcels

        except Exception as e:
            self.logger.error(f"Error searching letter {letter}: {e}")
            return []

    def get_parcel_details(self, parcel_id: str) -> Optional[HamiltonParcel]:
        """
        Get full details for a parcel from its detail page.

        Args:
            parcel_id: Parcel ID (with or without dashes)

        Returns:
            HamiltonParcel with full details
        """
        try:
            # Remove dashes for URL
            clean_id = parcel_id.replace('-', '')
            url = self.DETAIL_URL.format(parcel_id=clean_id)

            self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            time.sleep(2)

            parcel = HamiltonParcel(parcel_id=parcel_id)
            page_text = self.page.inner_text('body')

            # Extract owner name and address
            owner_section = re.search(
                r'Owner Name and Address\s*\n+(.+?)\n+(.+?)(?:\n|$)',
                page_text, re.IGNORECASE
            )
            if owner_section:
                parcel.owner_name = owner_section.group(1).strip()
                parcel.owner_address = owner_section.group(2).strip()

            # Extract site address
            addr_match = re.search(r'Address\s*\n+([^\n]+)', page_text)
            if addr_match:
                parcel.site_address = addr_match.group(1).strip()

            # Extract mailing address
            mail_section = re.search(
                r'Tax Bill Mail Address\s*\n+(.+?)\n+(.+?)(?:\n|$)',
                page_text, re.IGNORECASE
            )
            if mail_section:
                parcel.mailing_address = f"{mail_section.group(1).strip()} {mail_section.group(2).strip()}"

            # Extract market value
            market_match = re.search(r'Market Total Value\s*([\d,]+)', page_text)
            if market_match:
                parcel.market_value = self._parse_money(market_match.group(1))

            # Extract land value
            land_match = re.search(r'Market Land Value\s*([\d,]+)', page_text)
            if land_match:
                parcel.land_value = self._parse_money(land_match.group(1))

            # Extract building value
            bldg_match = re.search(r'Market Improvement Value\s*([\d,]+)', page_text)
            if bldg_match:
                parcel.building_value = self._parse_money(bldg_match.group(1))

            # Extract property type (land use)
            use_match = re.search(r'Auditor Land Use\s*\n*([^\n]+)', page_text)
            if use_match:
                parcel.property_type = use_match.group(1).strip()

            # Extract school district
            school_match = re.search(r'School District\s+([A-Z][^\t\n]+)', page_text)
            if school_match:
                parcel.school_district = school_match.group(1).strip()

            # Extract appraisal area
            area_match = re.search(r'Appraisal Area\s*\n*([^\n]+)', page_text)
            if area_match:
                parcel.appraisal_area = area_match.group(1).strip()

            # Extract zip from owner address
            zip_match = re.search(r'(\d{5})(?:-\d{4})?', parcel.owner_address)
            if zip_match:
                parcel.zip_code = zip_match.group(1)

            return parcel

        except Exception as e:
            self.logger.error(f"Error getting details for {parcel_id}: {e}")
            return None

    def scrape_all_patterns(self, patterns: List[str] = None, save_progress: bool = True) -> pd.DataFrame:
        """
        Scrape all parcels by searching each 2-letter pattern.

        Args:
            patterns: List of patterns to search (default: AA-ZZ + 00-09)
            save_progress: Save progress after each pattern

        Returns:
            DataFrame with all parcels
        """
        if patterns is None:
            patterns = self.SEARCH_PATTERNS

        all_parcels = []
        progress_file = RAW_DATA_DIR / 'hamilton_scrape_progress.json'
        partial_file = RAW_DATA_DIR / 'hamilton_parcels_partial.csv'

        # Load progress if exists
        completed_patterns = set()
        if save_progress and progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    completed_patterns = set(progress.get('completed_patterns', []))
                    self.logger.info(f"Resuming: {len(completed_patterns)} patterns completed")
            except:
                pass

        # Load existing partial data
        if save_progress and partial_file.exists():
            try:
                existing_df = pd.read_csv(partial_file)
                all_parcels = existing_df.to_dict('records')
                self.logger.info(f"Loaded {len(all_parcels)} existing parcels")
            except:
                pass

        self._init_browser()

        try:
            for i, pattern in enumerate(patterns):
                if pattern in completed_patterns:
                    continue  # Skip silently to avoid log spam

                self.logger.info(f"[{i+1}/{len(patterns)}] Searching: {pattern}")

                parcels = self.search_by_letter(pattern)
                all_parcels.extend(parcels)

                completed_patterns.add(pattern)

                # Save progress
                if save_progress:
                    with open(progress_file, 'w') as f:
                        json.dump({'completed_patterns': list(completed_patterns)}, f)

                    df = pd.DataFrame(all_parcels)
                    df.to_csv(partial_file, index=False)

                    if len(all_parcels) % 5000 == 0:
                        self.logger.info(f"Progress: {len(all_parcels)} total parcels")

                time.sleep(self.delay)

        finally:
            self._close_browser()

        # Create final DataFrame
        df = pd.DataFrame(all_parcels)

        # Remove duplicates by parcel_id
        if 'parcel_id' in df.columns:
            df = df.drop_duplicates(subset=['parcel_id'], keep='first')

        return df

    def enrich_with_details(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Enrich parcel data with details from individual pages.

        Args:
            df: DataFrame with basic parcel data
            sample_size: Number of parcels to enrich (None = all)

        Returns:
            DataFrame with enriched data
        """
        if df.empty:
            return df

        parcels_to_enrich = df if sample_size is None else df.head(sample_size)

        self._init_browser()

        try:
            enriched = []
            total = len(parcels_to_enrich)

            for i, row in parcels_to_enrich.iterrows():
                if i % 100 == 0:
                    self.logger.info(f"Enriching {i}/{total}...")

                parcel = self.get_parcel_details(row['parcel_id'])

                if parcel:
                    enriched.append(parcel.to_dict())
                else:
                    enriched.append(row.to_dict())

                time.sleep(0.5)  # Be gentle

        finally:
            self._close_browser()

        return pd.DataFrame(enriched)

    def enrich_investor_parcels(self, investor_names: List[str], data_file: str = 'Hamilton_Parcel.xlsx') -> pd.DataFrame:
        """
        Enrich parcels only for identified investors (much faster than enriching all).

        Args:
            investor_names: List of investor owner names to enrich
            data_file: Hamilton parcel data file to load

        Returns:
            DataFrame with enriched investor parcels
        """
        data_path = RAW_DATA_DIR / data_file

        if not data_path.exists():
            self.logger.error(f"Data file not found: {data_path}")
            return pd.DataFrame()

        # Load existing data
        self.logger.info(f"Loading Hamilton data from {data_path}...")
        df = pd.read_excel(data_path, engine='openpyxl')
        self.logger.info(f"Loaded {len(df)} total parcels")

        # Find owner column
        owner_col = None
        for col in ['OwnerName1', 'owner_name', 'Owner Name']:
            if col in df.columns:
                owner_col = col
                break

        if not owner_col:
            self.logger.error("Owner column not found in data")
            return pd.DataFrame()

        # Normalize names for matching
        df['_normalized_owner'] = df[owner_col].fillna('').str.upper().str.strip()
        investor_names_normalized = [name.upper().strip() for name in investor_names]

        # Filter to investor parcels only
        investor_parcels = df[df['_normalized_owner'].isin(investor_names_normalized)].copy()
        self.logger.info(f"Found {len(investor_parcels)} parcels belonging to {len(investor_names)} investors")

        if investor_parcels.empty:
            return pd.DataFrame()

        # Get parcel ID column
        parcel_col = None
        for col in ['PARCEL ID', 'parcel_id', 'Parcel ID']:
            if col in investor_parcels.columns:
                parcel_col = col
                break

        if not parcel_col:
            self.logger.error("Parcel ID column not found")
            return pd.DataFrame()

        # Enrich these parcels
        self._init_browser()
        enriched_data = []
        total = len(investor_parcels)
        progress_file = RAW_DATA_DIR / 'hamilton_enrich_progress.json'

        # Load progress
        completed_parcels = set()
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    completed_parcels = set(progress.get('completed', []))
                    enriched_data = progress.get('enriched_data', [])
                    self.logger.info(f"Resuming: {len(completed_parcels)} parcels already enriched")
            except:
                pass

        try:
            for idx, (_, row) in enumerate(investor_parcels.iterrows()):
                parcel_id = str(row[parcel_col])

                if parcel_id in completed_parcels:
                    continue

                if idx % 50 == 0:
                    self.logger.info(f"Enriching investor parcels: {idx}/{total} ({len(enriched_data)} done)")

                parcel = self.get_parcel_details(parcel_id)

                if parcel:
                    parcel_dict = parcel.to_dict()
                    # Preserve original data and add enriched fields
                    parcel_dict['original_owner'] = row[owner_col]
                    enriched_data.append(parcel_dict)
                else:
                    # Keep original row data if enrichment fails
                    enriched_data.append(row.to_dict())

                completed_parcels.add(parcel_id)

                # Save progress every 100 parcels
                if len(completed_parcels) % 100 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'completed': list(completed_parcels),
                            'enriched_data': enriched_data
                        }, f)

                time.sleep(0.5)

        finally:
            self._close_browser()

            # Save final progress
            with open(progress_file, 'w') as f:
                json.dump({
                    'completed': list(completed_parcels),
                    'enriched_data': enriched_data
                }, f)

        enriched_df = pd.DataFrame(enriched_data)
        self.logger.info(f"Enriched {len(enriched_df)} investor parcels")

        # Save enriched investor data
        enriched_path = RAW_DATA_DIR / 'Hamilton_Investor_Parcels_Enriched.xlsx'
        enriched_df.to_excel(enriched_path, index=False, engine='openpyxl')
        self.logger.info(f"Saved enriched data to {enriched_path}")

        return enriched_df

    def scrape_and_save(self, output_file: str = 'Hamilton_Parcel.xlsx', enrich: bool = False) -> str:
        """
        Scrape all Hamilton County data and save to Excel file.

        Args:
            output_file: Output filename (saved to data/raw/)
            enrich: Whether to get full details for each parcel (slow!)

        Returns:
            Path to saved file
        """
        self.logger.info("Starting Hamilton County scrape...")
        self.logger.info(f"Will search {len(self.SEARCH_PATTERNS)} patterns (AA-ZZ + 00-09)")

        # Step 1: Get all parcels via CSV export
        df = self.scrape_all_patterns()

        if df.empty:
            self.logger.error("No data scraped!")
            return ""

        self.logger.info(f"Scraped {len(df)} parcels")

        # Step 2: Optionally enrich with details
        if enrich:
            self.logger.info("Enriching with parcel details (this will take a while)...")
            df = self.enrich_with_details(df)

        # Rename columns to match Franklin County format for pipeline compatibility
        column_mapping = {
            'parcel_id': 'PARCEL ID',
            'site_address': 'SiteAddress',
            'owner_name': 'OwnerName1',
            'owner_address': 'OwnerAddress1',
            'mailing_address': 'TaxpayerAddress1',
            'zip_code': 'ZipCode',
            'property_type': 'LUCDesc',
            'market_value': 'MarketValue',
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Save to Excel
        output_path = RAW_DATA_DIR / output_file
        df.to_excel(output_path, index=False, engine='openpyxl')

        self.logger.info(f"Saved {len(df)} parcels to {output_path}")

        return str(output_path)


# ========================================
# Convenience Functions
# ========================================

def scrape_hamilton_county(headless: bool = False, enrich: bool = False) -> str:
    """
    Scrape Hamilton County property data and save to file.

    Args:
        headless: Run browser in headless mode
        enrich: Get full details for each parcel (slow!)

    Returns:
        Path to saved Excel file
    """
    scraper = HamiltonCountyScraper(headless=headless)
    return scraper.scrape_and_save(enrich=enrich)


def test_scraper(pattern: str = 'SM') -> List[Dict]:
    """Quick test with a 2-char pattern"""
    scraper = HamiltonCountyScraper(headless=False)
    scraper._init_browser()
    try:
        return scraper.search_by_letter(pattern)
    finally:
        scraper._close_browser()


def enrich_hamilton_investors(headless: bool = False) -> str:
    """
    Enrich Hamilton County data for identified investors only.

    This is the smart workflow:
    1. Load Hamilton parcel data (from basic scrape)
    2. Run investor pipeline to identify investors
    3. Enrich only the investor parcels (not all 300K)

    Returns:
        Path to enriched investor parcels file
    """
    from data_processing.investor_pipeline import InvestorPipeline

    # Step 1: Run pipeline to identify investors
    print("Step 1: Identifying Hamilton County investors...")
    pipeline = InvestorPipeline(county='hamilton')

    try:
        investors = pipeline.run(min_portfolio_size=2, min_score=40)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("Make sure you've run the basic scrape first: python3 -m scrapers.hamilton_county --full")
        return ""

    if investors.empty:
        print("No investors identified. Run basic scrape first.")
        return ""

    print(f"Identified {len(investors)} investors")

    # Step 2: Get investor names
    investor_names = investors['owner_name'].tolist()
    print(f"Will enrich parcels for {len(investor_names)} investors")

    # Step 3: Enrich their parcels
    print("\nStep 2: Enriching investor parcels with detailed data...")
    scraper = HamiltonCountyScraper(headless=headless)
    enriched = scraper.enrich_investor_parcels(investor_names)

    if enriched.empty:
        print("No parcels enriched")
        return ""

    output_path = RAW_DATA_DIR / 'Hamilton_Investor_Parcels_Enriched.xlsx'
    print(f"\nDone! Enriched {len(enriched)} parcels saved to: {output_path}")

    return str(output_path)


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    import sys

    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not installed!")
        print("Run: pip install playwright && python -m playwright install firefox")
        sys.exit(1)

    print("\nHamilton County Property Scraper")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full scrape
        print("Starting FULL scrape (AA-ZZ + 00-09)...")
        print("This will search 686 patterns for ~300K+ parcels")
        print("Progress is saved - you can resume if interrupted.\n")
        path = scrape_hamilton_county(headless=False)
        print(f"\nSaved to: {path}")

    elif len(sys.argv) > 1 and sys.argv[1] == '--enrich-investors':
        # Enrich only investor parcels (smart workflow)
        print("SMART ENRICHMENT MODE")
        print("=" * 50)
        print("This will:")
        print("  1. Run pipeline to identify investors")
        print("  2. Enrich ONLY investor parcels (not all 300K)")
        print("  3. Save enriched data with market values, addresses, etc.")
        print()
        path = enrich_hamilton_investors(headless=False)
        if path:
            print(f"\nEnriched data saved to: {path}")

    elif len(sys.argv) > 1 and sys.argv[1] == '--test-detail':
        # Test parcel detail extraction
        parcel_id = sys.argv[2] if len(sys.argv) > 2 else "528-0003-0273-00"
        print(f"Testing parcel detail for: {parcel_id}")

        scraper = HamiltonCountyScraper(headless=False)
        scraper._init_browser()
        try:
            parcel = scraper.get_parcel_details(parcel_id)
            if parcel:
                print(f"\nParcel: {parcel.parcel_id}")
                print(f"Owner: {parcel.owner_name}")
                print(f"Owner Address: {parcel.owner_address}")
                print(f"Site Address: {parcel.site_address}")
                print(f"Market Value: ${parcel.market_value:,.0f}")
                print(f"Property Type: {parcel.property_type}")
                print(f"School District: {parcel.school_district}")
        finally:
            scraper._close_browser()

    else:
        # Test with 2-letter pattern (API requires 2+ chars)
        test_pattern = sys.argv[1] if len(sys.argv) > 1 else "SM"
        if len(test_pattern) < 2:
            test_pattern = test_pattern + "A"  # Pad to 2 chars
        print(f"Testing with pattern: {test_pattern}")
        print("(Use --full for complete scrape)")

        scraper = HamiltonCountyScraper(headless=False)
        scraper._init_browser()

        try:
            parcels = scraper.search_by_letter(test_pattern)

            print(f"\nFound {len(parcels)} parcels")

            for i, p in enumerate(parcels[:5]):
                print(f"\n{i+1}. {p['parcel_id']}")
                print(f"   Owner: {p['owner_name']}")
                print(f"   Address: {p['site_address']}")
        finally:
            scraper._close_browser()
