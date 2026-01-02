"""
Ohio Secretary of State - Business Entity Scraper

Scrapes registered agent and business information from Ohio SOS.
Useful for finding contact info behind LLC/Corp entities.

Website: https://businesssearch.ohiosos.gov/
"""

import re
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from config.logging_config import get_logger

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class BusinessEntity:
    """Data class for Ohio SOS business entity information"""
    entity_name: str
    entity_number: str = ""
    entity_type: str = ""
    status: str = ""
    formation_date: str = ""
    expiration_date: str = ""

    # Registered Agent Info (this is the key data)
    agent_name: str = ""
    agent_address: str = ""

    # Principal Place of Business
    principal_address: str = ""

    def to_dict(self) -> Dict:
        return {
            'entity_name': self.entity_name,
            'entity_number': self.entity_number,
            'entity_type': self.entity_type,
            'status': self.status,
            'formation_date': self.formation_date,
            'agent_name': self.agent_name,
            'agent_address': self.agent_address,
            'principal_address': self.principal_address,
        }


class OhioSOSScraper:
    """
    Scrape business entity information from Ohio Secretary of State.

    Uses Playwright with Firefox for browser automation to handle JavaScript rendering.
    Note: Firefox is required to bypass Cloudflare protection.
    Note: headless=False is required as Cloudflare blocks headless browsers.
    """

    SEARCH_URL = "https://businesssearch.ohiosos.gov/"

    def __init__(self, headless: bool = False, delay: float = 2.0):
        self.logger = get_logger(self.__class__.__name__)
        # Force headless=False as Cloudflare blocks headless browsers
        self.headless = False  # Override - Cloudflare blocks headless
        self.delay = delay
        self.playwright = None
        self.browser = None
        self.page = None

        if not PLAYWRIGHT_AVAILABLE:
            self.logger.error("Playwright not installed. Run: pip install playwright && python -m playwright install firefox")

    def _init_browser(self):
        """Initialize browser - uses Firefox to bypass Cloudflare"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed")

        self.playwright = sync_playwright().start()
        # Use Firefox - Chromium gets blocked by Cloudflare
        self.browser = self.playwright.firefox.launch(headless=self.headless)
        self.page = self.browser.new_page()

        self.logger.info("Firefox browser initialized")

    def _close_browser(self):
        """Close browser"""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except:
            pass
        self.logger.info("Browser closed")

    def search_business(self, business_name: str) -> Optional[BusinessEntity]:
        """
        Search for a business by name.

        Args:
            business_name: Business name (e.g., "ABC PROPERTIES LLC")

        Returns:
            BusinessEntity or None
        """
        need_init = self.page is None

        try:
            if need_init:
                self._init_browser()

            self.logger.info(f"Searching Ohio SOS for: {business_name}")

            # Navigate to search page
            self.page.goto(self.SEARCH_URL, wait_until='domcontentloaded', timeout=30000)

            # Wait for Cloudflare challenge to pass (if present)
            # Cloudflare shows "Just a moment..." page
            max_cf_wait = 15
            for i in range(max_cf_wait):
                if 'Just a moment' in self.page.title():
                    self.logger.info(f"Waiting for Cloudflare challenge... ({i+1}s)")
                    time.sleep(1)
                else:
                    break

            # Wait for page to be fully ready
            time.sleep(3)
            self.page.wait_for_load_state('networkidle', timeout=15000)

            # Find search input - Ohio SOS uses id="bSearch"
            search_input = self.page.query_selector('#bSearch')

            if not search_input:
                # Try alternate selectors
                search_input = self.page.query_selector('input[placeholder*="Business"]')

            if not search_input:
                search_input = self.page.query_selector('input[type="text"]')

            if not search_input:
                self.logger.error("Could not find search input")
                return None

            # Click to focus and type business name
            search_input.click()
            time.sleep(0.3)
            search_input.fill(business_name)
            time.sleep(0.5)

            # Find and click the SEARCH button (first .srchBtn in search-button-div)
            search_btn = self.page.query_selector('.search-button-div .srchBtn')
            if search_btn:
                search_btn.click()
            else:
                # Fallback: press Enter
                self.page.keyboard.press("Enter")

            # Wait for results
            time.sleep(5)
            self.page.wait_for_load_state('networkidle', timeout=15000)

            # Check for results - look for "SHOW DETAILS" links
            show_details_links = self.page.query_selector_all('a:has-text("SHOW DETAILS")')

            if not show_details_links:
                # Try alternate: look for View Report links
                show_details_links = self.page.query_selector_all('a:has-text("View Report")')

            if not show_details_links:
                self.logger.info(f"No results found for: {business_name}")
                return None

            self.logger.info(f"Found {len(show_details_links)} results")

            # Click the first "SHOW DETAILS" link to get full info
            show_details_links[0].click()
            time.sleep(3)
            self.page.wait_for_load_state('networkidle', timeout=15000)

            # Extract entity details from the detail page
            entity = self._extract_details(business_name)

            time.sleep(self.delay)
            return entity

        except Exception as e:
            self.logger.error(f"Error searching for {business_name}: {e}")
            return None

        finally:
            if need_init:
                self._close_browser()

    def _extract_details(self, original_name: str) -> BusinessEntity:
        """Extract entity details from detail modal/page"""
        entity = BusinessEntity(entity_name=original_name)

        try:
            # Get all text content
            page_text = self.page.inner_text('body')

            # Extract from search results table (visible before clicking SHOW DETAILS)
            # Format: Entity# Name Type Filing Date Exp. Date Status Location County State
            # Example: 5040662 ARDMORE PROPERTY, LLC FOREIGN LIMITED LIABILITY COMPANY 04/26/2023 - Active

            # Try to find entity number from table row
            entity_match = re.search(r'(\d{7})\s+' + re.escape(original_name.replace(' LLC', '').replace(' Inc', '')),
                                     page_text, re.IGNORECASE)
            if entity_match:
                entity.entity_number = entity_match.group(1)

            # Alternative: look for Entity # in detail modal
            if not entity.entity_number:
                entity_num_match = re.search(r'Entity\s*#:\s*(\d+)', page_text, re.IGNORECASE)
                if entity_num_match:
                    entity.entity_number = entity_num_match.group(1)

            # Extract status
            status_match = re.search(r'\t(Active|Cancelled|Dead|Fraudulent)\t', page_text, re.IGNORECASE)
            if status_match:
                entity.status = status_match.group(1).strip()

            # Extract filing date
            date_match = re.search(r'Original Filing Date:\s*(\d{1,2}/\d{1,2}/\d{4})', page_text)
            if not date_match:
                date_match = re.search(r'(\d{2}/\d{2}/\d{4})\s*[-–]\s*(Active|Cancelled)', page_text)
            if date_match:
                entity.formation_date = date_match.group(1)

            # Extract entity type
            type_match = re.search(r'(FOREIGN LIMITED LIABILITY COMPANY|LIMITED LIABILITY COMPANY|'
                                   r'CORPORATION|DOMESTIC LIMITED LIABILITY|FOR PROFIT CORPORATION|'
                                   r'NON PROFIT CORPORATION)', page_text, re.IGNORECASE)
            if type_match:
                entity.entity_type = type_match.group(1).strip()

            # Extract AGENT/REGISTRANT INFORMATION section
            # Format in the detail modal:
            # AGENT/REGISTRANT INFORMATION
            # DBO SERVICES LLC
            # 2538 LAURELHURST RD
            # UNIVERSITY HEIGHTS OH 44118
            agent_section_match = re.search(
                r'AGENT/REGISTRANT INFORMATION\s*\n+(.+?)\n+(.+?)\n+(.+?)(?:\n|$)',
                page_text, re.IGNORECASE | re.DOTALL
            )
            if agent_section_match:
                entity.agent_name = agent_section_match.group(1).strip()
                street = agent_section_match.group(2).strip()
                city_state = agent_section_match.group(3).strip()
                entity.agent_address = f"{street}, {city_state}"
            else:
                # Try alternate patterns
                agent_match = re.search(
                    r'AGENT[/\s]*(?:REGISTRANT)?\s*(?:INFORMATION)?\s*\n+([A-Z][^\n]+)\n+([^\n]+)\n+([^\n]+)',
                    page_text, re.IGNORECASE
                )
                if agent_match:
                    entity.agent_name = agent_match.group(1).strip()
                    entity.agent_address = f"{agent_match.group(2).strip()}, {agent_match.group(3).strip()}"

            self.logger.info(f"Extracted: {entity.entity_name} | Entity#: {entity.entity_number} | Agent: {entity.agent_name}")

        except Exception as e:
            self.logger.error(f"Error extracting details: {e}")

        return entity

    def _find_section(self, text: str, headers: List[str]) -> Optional[str]:
        """Find a section of text following a header"""
        for header in headers:
            pattern = rf'{header}[:\s]*\n(.+?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_from_dom(self, entity: BusinessEntity):
        """Try to extract data from DOM elements"""
        try:
            # Look for labeled data in tables or definition lists
            rows = self.page.query_selector_all('tr, dl dt, .field-label')

            for row in rows:
                try:
                    text = row.inner_text().lower()

                    if 'agent' in text and not entity.agent_name:
                        # Get next sibling or adjacent cell
                        sibling = row.query_selector('+ td, + dd, ~ .field-value')
                        if sibling:
                            entity.agent_name = sibling.inner_text().strip()

                    elif 'status' in text and not entity.status:
                        sibling = row.query_selector('+ td, + dd, ~ .field-value')
                        if sibling:
                            entity.status = sibling.inner_text().strip()

                except:
                    continue

        except Exception as e:
            self.logger.debug(f"DOM extraction error: {e}")

    def search_batch(self, names: List[str]) -> List[BusinessEntity]:
        """Search multiple businesses"""
        results = []

        self._init_browser()

        try:
            for i, name in enumerate(names):
                self.logger.info(f"Processing {i+1}/{len(names)}: {name}")

                entity = self.search_business(name)
                if entity:
                    results.append(entity)

                time.sleep(self.delay)

        finally:
            self._close_browser()

        return results


# ========================================
# Convenience Functions
# ========================================

def lookup_llc(name: str, headless: bool = True) -> Optional[Dict]:
    """
    Look up a single LLC.

    Args:
        name: Business name (e.g., "ABC PROPERTIES LLC")
        headless: Run browser in headless mode

    Returns:
        Dict with entity info or None
    """
    scraper = OhioSOSScraper(headless=headless)
    entity = scraper.search_business(name)
    return entity.to_dict() if entity else None


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    import sys

    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not installed!")
        print("Run: pip install playwright && python -m playwright install firefox")
        sys.exit(1)

    # Test with a sample LLC
    test_name = sys.argv[1] if len(sys.argv) > 1 else "ARDMORE PROPERTY LLC"

    print(f"\nOhio SOS Lookup: {test_name}")
    print("=" * 50)

    # Note: headless=False is required - Cloudflare blocks headless browsers
    scraper = OhioSOSScraper(headless=False)
    result = scraper.search_business(test_name)

    if result:
        print(f"\nRESULTS:")
        print(f"  Entity #: {result.entity_number}")
        print(f"  Entity Type: {result.entity_type}")
        print(f"  Status: {result.status}")
        print(f"  Formation: {result.formation_date}")
        print(f"  Agent Name: {result.agent_name}")
        print(f"  Agent Address: {result.agent_address}")
        print(f"  Principal Address: {result.principal_address}")
    else:
        print("Not found or error occurred")
