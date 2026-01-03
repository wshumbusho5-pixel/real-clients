"""
Skip Tracing Module - Get phone/email contacts for property owners

Supports multiple skip tracing providers:
- Batch Skip Tracing API
- REI Skip API
- Skip Genie API
- Manual/Free methods (TruePeopleSearch, FastPeopleSearch)
"""

import os
import re
import json
import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import requests

from config.logging_config import get_logger

# Try to import playwright for free skip tracing
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class ContactInfo:
    """Contact information from skip tracing"""
    name: str
    address: str = ""

    # Phone numbers
    phones: List[str] = field(default_factory=list)
    phone_primary: str = ""
    phone_type: str = ""  # mobile, landline, voip

    # Email addresses
    emails: List[str] = field(default_factory=list)
    email_primary: str = ""

    # Additional info
    age: str = ""
    relatives: List[str] = field(default_factory=list)

    # Metadata
    source: str = ""
    confidence: float = 0.0
    lookup_date: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'address': self.address,
            'phone_primary': self.phone_primary,
            'phones': self.phones,
            'phone_type': self.phone_type,
            'email_primary': self.email_primary,
            'emails': self.emails,
            'age': self.age,
            'relatives': self.relatives,
            'source': self.source,
            'confidence': self.confidence,
            'lookup_date': self.lookup_date,
        }

    def has_contact(self) -> bool:
        """Check if we found any contact info"""
        return bool(self.phone_primary or self.email_primary or self.phones or self.emails)


class SkipTracingProvider:
    """Base class for skip tracing providers"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def lookup(self, name: str, address: str = "", city: str = "", state: str = "", zip_code: str = "") -> Optional[ContactInfo]:
        raise NotImplementedError


class BatchSkipTracing(SkipTracingProvider):
    """
    Batch Skip Tracing API integration
    Website: https://batchskiptracing.com
    Cost: ~$0.15-0.20 per record
    """

    API_URL = "https://api.batchskiptracing.com/api/v1"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv('BATCH_SKIP_API_KEY', '')

        if not self.api_key:
            self.logger.warning("Batch Skip Tracing API key not set. Set BATCH_SKIP_API_KEY env var.")

    def lookup(self, name: str, address: str = "", city: str = "", state: str = "OH", zip_code: str = "") -> Optional[ContactInfo]:
        """Look up contact info via Batch Skip Tracing API"""
        if not self.api_key:
            self.logger.error("API key not configured")
            return None

        try:
            # Parse name into first/last
            name_parts = name.strip().split()
            first_name = name_parts[0] if name_parts else ""
            last_name = name_parts[-1] if len(name_parts) > 1 else ""

            # Parse address
            if not city and address:
                # Try to extract city/state from address
                addr_match = re.search(r',\s*([^,]+),?\s*([A-Z]{2})\s*(\d{5})?', address)
                if addr_match:
                    city = addr_match.group(1).strip()
                    state = addr_match.group(2)
                    zip_code = addr_match.group(3) or ""

            payload = {
                "first_name": first_name,
                "last_name": last_name,
                "address": address.split(',')[0] if address else "",
                "city": city,
                "state": state,
                "zip": zip_code,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                f"{self.API_URL}/skip-trace",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_response(name, address, data)
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Lookup error: {e}")
            return None

    def _parse_response(self, name: str, address: str, data: Dict) -> ContactInfo:
        """Parse API response into ContactInfo"""
        contact = ContactInfo(
            name=name,
            address=address,
            source="batch_skip_tracing",
            lookup_date=datetime.now().isoformat(),
        )

        if 'phones' in data:
            contact.phones = [p.get('number', '') for p in data['phones'] if p.get('number')]
            if contact.phones:
                contact.phone_primary = contact.phones[0]
                contact.phone_type = data['phones'][0].get('type', 'unknown')

        if 'emails' in data:
            contact.emails = [e.get('address', '') for e in data['emails'] if e.get('address')]
            if contact.emails:
                contact.email_primary = contact.emails[0]

        contact.confidence = data.get('confidence', 0.0)

        return contact


class REISkip(SkipTracingProvider):
    """
    REI Skip API integration
    Website: https://reiskip.com
    Cost: ~$0.10-0.15 per record
    """

    API_URL = "https://api.reiskip.com/v1"

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv('REI_SKIP_API_KEY', '')

    def lookup(self, name: str, address: str = "", city: str = "", state: str = "OH", zip_code: str = "") -> Optional[ContactInfo]:
        """Look up contact info via REI Skip API"""
        if not self.api_key:
            self.logger.error("REI Skip API key not configured")
            return None

        try:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }

            payload = {
                "name": name,
                "street": address.split(',')[0] if address else "",
                "city": city,
                "state": state,
                "zip": zip_code,
            }

            response = requests.post(
                f"{self.API_URL}/lookup",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                contact = ContactInfo(
                    name=name,
                    address=address,
                    source="rei_skip",
                    lookup_date=datetime.now().isoformat(),
                )

                if data.get('phone'):
                    contact.phone_primary = data['phone']
                    contact.phones = [data['phone']]

                if data.get('email'):
                    contact.email_primary = data['email']
                    contact.emails = [data['email']]

                return contact

            return None

        except Exception as e:
            self.logger.error(f"REI Skip error: {e}")
            return None


class FreeSkipTracing(SkipTracingProvider):
    """
    Free skip tracing using public records sites
    Uses Playwright to scrape TruePeopleSearch, FastPeopleSearch, etc.

    Note: Slower and less reliable than paid APIs, but free.
    """

    def __init__(self):
        super().__init__()
        self.playwright = None
        self.browser = None
        self.page = None

    def _init_browser(self):
        """Initialize browser"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed")

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.firefox.launch(headless=False)
        self.page = self.browser.new_page()

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

    def lookup(self, name: str, address: str = "", city: str = "", state: str = "OH", zip_code: str = "") -> Optional[ContactInfo]:
        """Look up using free people search sites"""
        if not PLAYWRIGHT_AVAILABLE:
            self.logger.error("Playwright required for free skip tracing")
            return None

        contact = ContactInfo(
            name=name,
            address=address,
            source="free_lookup",
            lookup_date=datetime.now().isoformat(),
        )

        try:
            self._init_browser()

            # Try TruePeopleSearch
            result = self._search_truepeoplesearch(name, city, state)
            if result:
                contact.phones = result.get('phones', [])
                contact.emails = result.get('emails', [])
                contact.phone_primary = contact.phones[0] if contact.phones else ""
                contact.email_primary = contact.emails[0] if contact.emails else ""
                contact.age = result.get('age', '')
                contact.relatives = result.get('relatives', [])

            return contact if contact.has_contact() else None

        except Exception as e:
            self.logger.error(f"Free lookup error: {e}")
            return None
        finally:
            self._close_browser()

    def _search_truepeoplesearch(self, name: str, city: str = "", state: str = "OH") -> Optional[Dict]:
        """Search TruePeopleSearch.com"""
        try:
            # Format search URL
            name_formatted = name.replace(' ', '-').lower()
            location = f"{city}-{state}".lower() if city else state.lower()
            url = f"https://www.truepeoplesearch.com/results?name={name_formatted}&citystatezip={location}"

            self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            time.sleep(3)

            # Look for first result
            result_card = self.page.query_selector('.card-summary')
            if not result_card:
                return None

            # Click to view details
            result_card.click()
            time.sleep(2)

            result = {'phones': [], 'emails': [], 'age': '', 'relatives': []}

            # Extract phone numbers
            phone_elements = self.page.query_selector_all('[data-link-to-more="phone"] a, .phone')
            for el in phone_elements:
                phone = el.inner_text().strip()
                phone = re.sub(r'[^\d]', '', phone)
                if len(phone) == 10:
                    result['phones'].append(phone)

            # Extract emails
            email_elements = self.page.query_selector_all('[data-link-to-more="email"] a, .email')
            for el in email_elements:
                email = el.inner_text().strip()
                if '@' in email:
                    result['emails'].append(email)

            # Extract age
            age_el = self.page.query_selector('.age, [class*="age"]')
            if age_el:
                age_text = age_el.inner_text()
                age_match = re.search(r'(\d+)', age_text)
                if age_match:
                    result['age'] = age_match.group(1)

            return result if result['phones'] or result['emails'] else None

        except Exception as e:
            self.logger.debug(f"TruePeopleSearch error: {e}")
            return None


class SkipTracer:
    """
    Main skip tracing class that manages providers and caching
    """

    def __init__(self, provider: str = "auto", cache_file: str = None):
        """
        Initialize skip tracer

        Args:
            provider: Which provider to use - "batch", "rei", "free", or "auto"
            cache_file: Path to cache file for storing results
        """
        self.logger = get_logger("SkipTracer")
        self.provider_name = provider
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'skip_trace_cache.json'
        )
        self.cache = self._load_cache()

        # Initialize providers
        self.providers = {
            'batch': BatchSkipTracing(),
            'rei': REISkip(),
            'free': FreeSkipTracing(),
        }

    def _load_cache(self) -> Dict:
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Cache save error: {e}")

    def _get_cache_key(self, name: str, address: str) -> str:
        """Generate cache key from name and address"""
        key_str = f"{name.lower().strip()}|{address.lower().strip()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def lookup(self, name: str, address: str = "", city: str = "", state: str = "OH",
               zip_code: str = "", use_cache: bool = True) -> Optional[ContactInfo]:
        """
        Look up contact info for a person/entity

        Args:
            name: Full name or business name
            address: Street address
            city: City
            state: State (default OH)
            zip_code: ZIP code
            use_cache: Whether to use cached results

        Returns:
            ContactInfo or None
        """
        # Check cache first
        cache_key = self._get_cache_key(name, address)
        if use_cache and cache_key in self.cache:
            self.logger.info(f"Cache hit for: {name}")
            cached = self.cache[cache_key]
            return ContactInfo(**cached) if cached else None

        self.logger.info(f"Looking up: {name}")

        result = None

        if self.provider_name == "auto":
            # Try paid providers first, fall back to free
            for provider_name in ['batch', 'rei', 'free']:
                provider = self.providers[provider_name]
                try:
                    result = provider.lookup(name, address, city, state, zip_code)
                    if result and result.has_contact():
                        break
                except Exception as e:
                    self.logger.debug(f"{provider_name} failed: {e}")
        else:
            provider = self.providers.get(self.provider_name)
            if provider:
                result = provider.lookup(name, address, city, state, zip_code)

        # Cache result
        self.cache[cache_key] = result.to_dict() if result else None
        self._save_cache()

        return result

    def lookup_batch(self, records: List[Dict], name_field: str = "name",
                     address_field: str = "address") -> List[ContactInfo]:
        """
        Look up multiple records

        Args:
            records: List of dicts with name/address info
            name_field: Key for name in each record
            address_field: Key for address in each record

        Returns:
            List of ContactInfo results
        """
        results = []

        for i, record in enumerate(records):
            name = record.get(name_field, "")
            address = record.get(address_field, "")

            if not name:
                continue

            self.logger.info(f"Processing {i+1}/{len(records)}: {name}")

            result = self.lookup(name, address)
            if result:
                results.append(result)

            # Rate limiting
            time.sleep(1)

        return results


# ========================================
# Convenience Functions
# ========================================

def skip_trace(name: str, address: str = "", provider: str = "free") -> Optional[Dict]:
    """
    Quick skip trace lookup

    Args:
        name: Person or business name
        address: Address (optional but improves accuracy)
        provider: "batch", "rei", "free", or "auto"

    Returns:
        Dict with contact info or None
    """
    tracer = SkipTracer(provider=provider)
    result = tracer.lookup(name, address)
    return result.to_dict() if result else None


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python skip_tracing.py 'Name' ['Address']")
        print("\nExample:")
        print("  python skip_tracing.py 'John Smith' '123 Main St, Columbus, OH 43215'")
        sys.exit(1)

    name = sys.argv[1]
    address = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"\nSkip Tracing: {name}")
    print("=" * 50)

    tracer = SkipTracer(provider="free")
    result = tracer.lookup(name, address)

    if result and result.has_contact():
        print(f"\nRESULTS:")
        print(f"  Name: {result.name}")
        print(f"  Phone: {result.phone_primary}")
        print(f"  All Phones: {', '.join(result.phones)}")
        print(f"  Email: {result.email_primary}")
        print(f"  All Emails: {', '.join(result.emails)}")
        print(f"  Age: {result.age}")
        print(f"  Source: {result.source}")
    else:
        print("\nNo contact info found")
