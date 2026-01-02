"""
Aerial Leads - Columbus Code Violations Scraper

Scrapes code violation data from Columbus, Ohio's enforcement portal.
Sources:
- Columbus Code Enforcement Portal (Accela system - JavaScript-heavy)
- Columbus 311 Open Data (easier alternative)

Due to JavaScript requirements, this uses Selenium/Playwright
"""

import time
import pandas as pd
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from config.settings import COLUMBUS_CODE_ENFORCEMENT_URL, RAW_DATA_DIR
from config.logging_config import log_success, log_failure
from scrapers.base_scraper import BaseScraper


class CodeViolationsScraper(BaseScraper):
    """
    Scrape code violations from Columbus, Ohio

    Uses Selenium for JavaScript-heavy website interaction
    """

    def __init__(self, headless: bool = True):
        super().__init__()
        self.enforcement_url = COLUMBUS_CODE_ENFORCEMENT_URL
        self.headless = headless
        self.driver = None

    def _setup_driver(self):
        """Initialize Selenium WebDriver"""
        options = Options()

        if self.headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument(f'user-agent={self.session.headers["User-Agent"]}')

        try:
            self.driver = webdriver.Chrome(options=options)
            self.logger.info("Chrome WebDriver initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            self.logger.info("Make sure Chrome and chromedriver are installed")
            raise

    def scrape(self, addresses: List[str]) -> Dict[str, List[Dict]]:
        """
        Search for code violations by address

        Args:
            addresses: List of property addresses to search

        Returns:
            Dictionary mapping addresses to their violation records
        """
        self.logger.info(f"🔍 Searching code violations for {len(addresses)} properties...")

        if not self.driver:
            self._setup_driver()

        results = {}

        for i, address in enumerate(addresses, 1):
            self.logger.info(f"Searching {i}/{len(addresses)}: {address}")

            violations = self._search_address(address)
            results[address] = violations

            # Rate limiting
            time.sleep(2)

        log_success(f"Found violations for {len([r for r in results.values() if r])} properties")

        return results

    def _search_address(self, address: str) -> List[Dict]:
        """
        Search for violations at a specific address

        Args:
            address: Property address

        Returns:
            List of violation dictionaries
        """
        violations = []

        try:
            # Navigate to search page
            self.driver.get(self.enforcement_url)
            time.sleep(3)  # Wait for page load

            # Find address search field
            # NOTE: These selectors are PLACEHOLDERS - inspect actual website and update
            try:
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "address-search"))
                )
            except TimeoutException:
                # Try alternative selector
                search_input = self.driver.find_element(By.NAME, "address")

            # Enter address
            search_input.clear()
            search_input.send_keys(address)

            # Click search button
            search_button = self.driver.find_element(By.ID, "search-button")
            search_button.click()

            # Wait for results
            time.sleep(3)

            # Parse results
            violations = self._parse_violation_results()

        except TimeoutException:
            self.logger.warning(f"Timeout searching for address: {address}")
        except NoSuchElementException as e:
            self.logger.warning(f"Element not found for address {address}: {e}")
        except Exception as e:
            self.logger.error(f"Error searching address {address}: {e}")

        return violations

    def _parse_violation_results(self) -> List[Dict]:
        """
        Parse code violation results from current page

        Returns:
            List of violation dictionaries
        """
        violations = []

        try:
            # Get page source
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Find violation rows
            # NOTE: These selectors are PLACEHOLDERS - update based on actual site
            violation_rows = soup.find_all('tr', class_='violation-row')

            for row in violation_rows:
                violation = {
                    'violation_type': self._safe_extract_text(row, 'td', 'type'),
                    'violation_date': self._safe_extract_text(row, 'td', 'date'),
                    'status': self._safe_extract_text(row, 'td', 'status'),
                    'description': self._safe_extract_text(row, 'td', 'description'),
                    'severity': self._determine_severity(self._safe_extract_text(row, 'td', 'type')),
                }

                violations.append(violation)

        except Exception as e:
            self.logger.error(f"Error parsing violations: {e}")

        return violations

    def _safe_extract_text(self, element, tag: str, class_name: str) -> str:
        """Safely extract text from element"""
        try:
            found = element.find(tag, class_=class_name)
            return found.text.strip() if found else ''
        except:
            return ''

    def _determine_severity(self, violation_type: str) -> str:
        """
        Determine violation severity based on type

        Args:
            violation_type: Type of violation

        Returns:
            'critical', 'major', or 'minor'
        """
        violation_type_lower = violation_type.lower()

        critical_keywords = ['housing', 'safety', 'structural', 'health', 'emergency']
        major_keywords = ['property maintenance', 'zoning', 'building', 'electrical']

        if any(keyword in violation_type_lower for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in violation_type_lower for keyword in major_keywords):
            return 'major'
        else:
            return 'minor'

    def aggregate_violations(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Aggregate violation data by address

        Args:
            results: Dictionary mapping addresses to violations

        Returns:
            Dictionary with aggregated violation data per address
        """
        aggregated = {}

        for address, violations in results.items():
            aggregated[address] = {
                'total_violations': len(violations),
                'critical_violations': len([v for v in violations if v.get('severity') == 'critical']),
                'major_violations': len([v for v in violations if v.get('severity') == 'major']),
                'minor_violations': len([v for v in violations if v.get('severity') == 'minor']),
                'open_violations': len([v for v in violations if v.get('status', '').lower() == 'open']),
                'closed_violations': len([v for v in violations if v.get('status', '').lower() == 'closed']),
                'violations': violations
            }

        return aggregated

    def export_to_csv(self, results: Dict[str, List[Dict]], filename: str = 'code_violations.csv'):
        """
        Export violations to CSV

        Args:
            results: Violation results
            filename: Output filename
        """
        if not results:
            log_failure("No violations to export")
            return

        # Flatten data for CSV
        rows = []
        for address, violations in results.items():
            for violation in violations:
                row = {'address': address, **violation}
                rows.append(row)

        if not rows:
            self.logger.warning("No violations found")
            return

        df = pd.DataFrame(rows)
        output_path = RAW_DATA_DIR / filename
        df.to_csv(output_path, index=False)

        log_success(f"Exported {len(rows)} violations to {output_path}")

    def __del__(self):
        """Clean up WebDriver on deletion"""
        if self.driver:
            self.driver.quit()


# ========================================
# Simpler Alternative: Columbus 311 Data
# ========================================

class Columbus311Scraper(BaseScraper):
    """
    Alternative scraper using Columbus 311 open data

    This is MUCH easier than scraping the Accela portal
    Check if Columbus has an open data portal first
    """

    def __init__(self):
        super().__init__()
        # Columbus 311 API endpoint (if available)
        self.api_url = "https://data.columbus.gov/api/311/violations"

    def scrape(self, addresses: Optional[List[str]] = None) -> List[Dict]:
        """
        Scrape from 311 open data API

        Args:
            addresses: Optional list of addresses to filter

        Returns:
            List of violation dictionaries
        """
        self.logger.info("Attempting to fetch from Columbus 311 API...")

        # Try API approach
        violations = self._fetch_from_api()

        if addresses:
            # Filter to specified addresses
            violations = [v for v in violations if v.get('address') in addresses]

        return violations

    def _fetch_from_api(self) -> List[Dict]:
        """
        Fetch violation data from API

        Returns:
            List of violations
        """
        # This is a template - adjust based on actual API
        response = self._make_request(self.api_url)

        if not response or response.status_code != 200:
            self.logger.error("Failed to fetch from 311 API")
            return []

        try:
            data = response.json()
            return data.get('violations', [])
        except:
            self.logger.error("Failed to parse 311 API response")
            return []


# ========================================
# Manual Data Entry Helper
# ========================================

def create_sample_violations() -> Dict[str, List[Dict]]:
    """
    Create sample violation data for testing

    Returns:
        Dictionary mapping addresses to violations
    """
    return {
        '123 Main St, Columbus, OH 43215': [
            {
                'violation_type': 'Property Maintenance',
                'violation_date': '2023-08-15',
                'status': 'Open',
                'description': 'Overgrown vegetation',
                'severity': 'minor'
            },
            {
                'violation_type': 'Housing Code',
                'violation_date': '2023-06-22',
                'status': 'Open',
                'description': 'Broken windows',
                'severity': 'major'
            }
        ],
        '789 Elm St, Columbus, OH 43201': []
    }


# Example usage
if __name__ == '__main__':
    # Option 1: Use sample data
    print("Creating sample violation data...")
    violations = create_sample_violations()

    scraper = CodeViolationsScraper()
    aggregated = scraper.aggregate_violations(violations)

    for address, data in aggregated.items():
        print(f"\n{address}")
        print(f"  Total violations: {data['total_violations']}")
        print(f"  Critical: {data['critical_violations']}")
        print(f"  Open: {data['open_violations']}")

    # Option 2: Try real scrape (requires Chrome + chromedriver)
    # scraper = CodeViolationsScraper(headless=True)
    # addresses = ['123 Main St, Columbus, OH 43215']
    # results = scraper.scrape(addresses)
