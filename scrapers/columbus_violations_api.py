"""
Aerial Leads - Columbus Code Violations API Scraper

Retrieves code enforcement violations from Columbus Open Data Portal ArcGIS REST API.
Much faster and more reliable than web scraping.

API: https://maps2.columbus.gov/arcgis/rest/services/Schemas/BuildingZoning/MapServer/23
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from scrapers.base_scraper import BaseScraper
from config.logging_config import log_success, log_failure


class ColumbusViolationsAPI(BaseScraper):
    """
    Query Columbus code enforcement violations via ArcGIS REST API

    Data Source: Columbus Open Data Portal
    - Code Enforcement Cases dataset
    - Real-time official city data
    - No authentication required
    """

    def __init__(self):
        super().__init__()
        self.api_base = "https://maps2.columbus.gov/arcgis/rest/services/Schemas/BuildingZoning/MapServer/23"
        self.max_records_per_request = 2000

    def scrape(self, parcel_numbers: List[str] = None, addresses: List[str] = None, **kwargs):
        """
        Main scraping method (required by BaseScraper)

        Args:
            parcel_numbers: List of parcel IDs to query
            addresses: List of addresses to query (fallback)

        Returns:
            Dict or DataFrame of violations
        """
        if parcel_numbers:
            return self.get_violations_by_parcel(parcel_numbers)
        elif addresses:
            return self.get_violations_by_address(addresses)
        else:
            return self.get_all_violations()

    def get_violations_by_parcel(self, parcel_numbers: List[str]) -> Dict[str, List[Dict]]:
        """
        Get code violations for specific parcel numbers (RECOMMENDED METHOD)

        Args:
            parcel_numbers: List of parcel IDs to query

        Returns:
            Dict mapping parcel_number -> list of violations
        """
        self.logger.info(f"🔍 Querying violations for {len(parcel_numbers)} parcels...")

        results = {}

        for parcel in parcel_numbers:
            violations = self._query_by_parcel(parcel)
            if violations:
                results[parcel] = violations

        log_success(f"Found violations for {len(results)}/{len(parcel_numbers)} parcels")

        return results

    def get_violations_by_address(self, addresses: List[str]) -> Dict[str, List[Dict]]:
        """
        Get code violations by address (fallback method)

        Args:
            addresses: List of property addresses

        Returns:
            Dict mapping address -> list of violations
        """
        self.logger.info(f"🔍 Querying violations for {len(addresses)} addresses...")

        results = {}

        for address in addresses:
            violations = self._query_by_address(address)
            if violations:
                results[address] = violations

        log_success(f"Found violations for {len(results)}/{len(addresses)} addresses")

        return results

    def get_all_violations(self, max_records: Optional[int] = None) -> pd.DataFrame:
        """
        Download all Columbus code enforcement violations

        Args:
            max_records: Maximum records to retrieve (None = all)

        Returns:
            DataFrame with all violations
        """
        self.logger.info("📥 Downloading all Columbus code violations...")

        all_violations = []
        offset = 0

        while True:
            # Make paginated request
            params = {
                'where': '1=1',
                'outFields': '*',
                'f': 'json',
                'resultOffset': offset,
                'resultRecordCount': self.max_records_per_request,
                'returnGeometry': 'false'
            }

            response = self._make_request(f"{self.api_base}/query", params=params)

            if not response:
                break

            data = response.json()
            features = data.get('features', [])

            if not features:
                break

            # Parse features
            for feature in features:
                violation = self._parse_violation(feature.get('attributes', {}))
                all_violations.append(violation)

            self.logger.info(f"Retrieved {len(all_violations)} violations...")

            # Check if we should continue
            offset += self.max_records_per_request

            if max_records and len(all_violations) >= max_records:
                all_violations = all_violations[:max_records]
                break

        df = pd.DataFrame(all_violations)
        log_success(f"Downloaded {len(df)} total violations")

        return df

    def _query_by_parcel(self, parcel_number: str) -> List[Dict]:
        """Query violations for a specific parcel number"""

        query_url = f"{self.api_base}/query"
        params = {
            'where': f"B1_PARCEL_NBR='{parcel_number}'",
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'false'
        }

        response = self._make_request(query_url, params=params)

        if not response:
            return []

        data = response.json()
        features = data.get('features', [])

        violations = []
        for feature in features:
            violation = self._parse_violation(feature.get('attributes', {}))
            violations.append(violation)

        return violations

    def _query_by_address(self, address: str) -> List[Dict]:
        """Query violations for a specific address"""

        # Normalize address for better matching
        address_normalized = address.upper().replace(',', '').strip()

        query_url = f"{self.api_base}/query"
        params = {
            'where': f"SITE_ADDRESS LIKE '%{address_normalized}%'",
            'outFields': '*',
            'f': 'json',
            'returnGeometry': 'false'
        }

        response = self._make_request(query_url, params=params)

        if not response:
            return []

        data = response.json()
        features = data.get('features', [])

        violations = []
        for feature in features:
            violation = self._parse_violation(feature.get('attributes', {}))
            violations.append(violation)

        return violations

    def _parse_violation(self, attrs: Dict) -> Dict:
        """Parse ArcGIS feature attributes to violation dict"""

        violation_type = attrs.get('B1_PER_TYPE', '')

        return {
            'case_id': attrs.get('B1_ALT_ID'),
            'parcel_number': attrs.get('B1_PARCEL_NBR'),
            'address': attrs.get('SITE_ADDRESS'),
            'violation_group': attrs.get('B1_PER_GROUP'),
            'violation_type': violation_type,
            'violation_subtype': attrs.get('B1_PER_SUB_TYPE'),
            'violation_category': attrs.get('B1_PER_CATEGORY'),
            'status': attrs.get('B1_APPL_STATUS'),
            'file_date': self._parse_timestamp(attrs.get('B1_FILE_DD')),
            'first_inspection_date': self._parse_timestamp(attrs.get('INSP_1ST_DATE')),
            'first_inspection_result': attrs.get('INSP_1ST_RESULT'),
            'last_inspection_date': self._parse_timestamp(attrs.get('INSP_LAST_DATE')),
            'last_inspection_result': attrs.get('INSP_LAST_RESULT'),
            'severity': self._determine_severity(violation_type),
            'case_url': attrs.get('ACA_URL'),
        }

    def _parse_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert ArcGIS timestamp (milliseconds since epoch) to date string"""
        if not timestamp:
            return None

        try:
            # ArcGIS uses milliseconds since epoch
            dt = datetime.fromtimestamp(timestamp / 1000)
            return dt.strftime('%Y-%m-%d')
        except:
            return None

    def _determine_severity(self, violation_type: str) -> str:
        """
        Classify violation severity based on violation type

        Returns: 'critical', 'major', or 'minor'
        """
        if not violation_type:
            return 'minor'

        violation_type_lower = violation_type.lower()

        # Critical violations (housing, safety, structural issues)
        critical_keywords = [
            'housing', 'safety', 'structural', 'emergency',
            'health', 'nuisance', 'vacant', 'blight', 'fire',
            'demolition', 'condemnation'
        ]

        # Major violations (zoning, property maintenance)
        major_keywords = [
            'zoning', 'property maintenance', 'building',
            'electrical', 'plumbing', 'sanitation', 'mechanical',
            'occupancy', 'environmental'
        ]

        # Minor violations (cosmetic, landscaping)
        minor_keywords = [
            'graphics', 'sign', 'landscaping', 'grass',
            'cosmetic', 'weeds', 'fence', 'parking'
        ]

        if any(kw in violation_type_lower for kw in critical_keywords):
            return 'critical'
        elif any(kw in violation_type_lower for kw in major_keywords):
            return 'major'
        else:
            return 'minor'

    def aggregate_violations_by_parcel(self, violations_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Aggregate violations by parcel for scoring

        Args:
            violations_dict: Output from get_violations_by_parcel()

        Returns:
            DataFrame with aggregated violation stats per parcel
        """
        aggregated = []

        for parcel_number, violations in violations_dict.items():
            if not violations:
                continue

            # Count violations by severity
            critical_count = sum(1 for v in violations if v['severity'] == 'critical')
            major_count = sum(1 for v in violations if v['severity'] == 'major')
            minor_count = sum(1 for v in violations if v['severity'] == 'minor')

            # Count open vs closed
            open_count = sum(1 for v in violations if v['status'] and 'open' in v['status'].lower())
            closed_count = sum(1 for v in violations if v['status'] and 'closed' in v['status'].lower())

            # Get most recent violation
            dates = [v['file_date'] for v in violations if v['file_date']]
            most_recent = max(dates) if dates else None
            oldest = min(dates) if dates else None

            aggregated.append({
                'parcel_number': parcel_number,
                'total_violations': len(violations),
                'critical_violations': critical_count,
                'major_violations': major_count,
                'minor_violations': minor_count,
                'open_violations': open_count,
                'closed_violations': closed_count,
                'most_recent_violation_date': most_recent,
                'oldest_violation_date': oldest,
            })

        return pd.DataFrame(aggregated)


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Test the API scraper"""

    api = ColumbusViolationsAPI()

    # Test 1: Query specific parcels
    print("\n🧪 TEST 1: Query by parcel number")
    test_parcels = ['010-021071-00', '010-020778-00', '010-020974-00']
    violations = api.get_violations_by_parcel(test_parcels)

    for parcel, viol_list in violations.items():
        print(f"\n{parcel}: {len(viol_list)} violations")
        for v in viol_list[:2]:  # Show first 2
            print(f"  - {v['violation_type']} ({v['severity']}) - {v['status']}")

    # Test 2: Aggregate violations
    if violations:
        print("\n🧪 TEST 2: Aggregate violations")
        df_agg = api.aggregate_violations_by_parcel(violations)
        print(df_agg.to_string())

    # Test 3: Download all violations (limited)
    print("\n🧪 TEST 3: Download sample of all violations")
    df_all = api.get_all_violations(max_records=100)
    print(f"\nDownloaded {len(df_all)} violations")
    print(f"Severity breakdown:")
    print(df_all['severity'].value_counts())

    print(f"\n✅ API scraper test complete!")
    print(f"Stats: {api.get_stats()}")
