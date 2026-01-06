"""
Entity Classifier - Detect owner entity types from property records

Classifies property owners as:
- individual: Regular person (John Smith)
- llc: Limited Liability Company (ABC Properties LLC)
- corporation: Inc/Corp (XYZ Holdings Inc)
- trust: Trust/Trustee (Smith Family Trust)
- estate: Estate/Executor (Estate of John Smith)
- partnership: LP/LLP (Investment Partners LP)
- nonprofit: Churches, charities (First Baptist Church)
- government: City/County/State entities
"""

import re
from typing import Dict, Tuple, List, Optional
import pandas as pd

from config.settings import ENTITY_KEYWORDS, INVESTOR_NAME_PATTERNS
from config.logging_config import get_logger


class EntityClassifier:
    """
    Classify property owners by entity type from owner name strings.

    This is the foundation for investor identification - LLC, Corp, Trust
    owners are much more likely to be investors than individuals.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.entity_keywords = ENTITY_KEYWORDS
        self.investor_patterns = INVESTOR_NAME_PATTERNS

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.compiled_patterns = {}

        for entity_type, keywords in self.entity_keywords.items():
            # Create pattern that matches whole words
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.compiled_patterns[entity_type] = re.compile(pattern, re.IGNORECASE)

        # Pattern for investor-related names
        investor_pattern = r'\b(' + '|'.join(re.escape(p) for p in self.investor_patterns) + r')\b'
        self.investor_name_pattern = re.compile(investor_pattern, re.IGNORECASE)

        # Pattern for PO Box detection
        self.po_box_pattern = re.compile(r'\bP\.?O\.?\s*BOX\b', re.IGNORECASE)

        # Pattern for suite/floor (business address)
        self.business_addr_pattern = re.compile(
            r'\b(SUITE|STE|FLOOR|FL|UNIT|#)\s*\d+', re.IGNORECASE
        )

    def classify_entity(self, owner_name: str) -> str:
        """
        Classify a single owner name into entity type.

        Args:
            owner_name: Raw owner name string from property records

        Returns:
            Entity type: 'individual', 'llc', 'corporation', 'trust',
                        'estate', 'partnership', 'nonprofit', 'government'
        """
        if not owner_name or pd.isna(owner_name):
            return 'unknown'

        name = str(owner_name).upper().strip()

        # Check each entity type in priority order
        # (government/institution first since "CITY OF COLUMBUS" should not match "CO.")
        # business is last - catch-all for names with investor keywords but no entity suffix
        priority_order = [
            'government', 'institution', 'nonprofit', 'estate', 'trust',
            'partnership', 'corporation', 'llc', 'business'
        ]

        for entity_type in priority_order:
            if self.compiled_patterns[entity_type].search(name):
                return entity_type

        return 'individual'

    def is_investor_entity(self, entity_type: str) -> bool:
        """
        Determine if an entity type is likely an investor.

        LLCs, corporations, partnerships, trusts, and businesses are likely investors.
        Individuals, nonprofits, institutions, and government are not.
        """
        investor_types = {'llc', 'corporation', 'partnership', 'trust', 'business'}
        return entity_type in investor_types

    def has_investor_name(self, owner_name: str) -> bool:
        """
        Check if owner name contains investor-related keywords.

        Examples: "ABC Properties LLC", "Smith Investment Holdings"
        """
        if not owner_name or pd.isna(owner_name):
            return False

        return bool(self.investor_name_pattern.search(str(owner_name)))

    def has_po_box(self, address: str) -> bool:
        """Check if address contains PO Box (indicates non-local entity)"""
        if not address or pd.isna(address):
            return False
        return bool(self.po_box_pattern.search(str(address)))

    def has_business_address(self, address: str) -> bool:
        """Check if address looks like a business (suite, floor number)"""
        if not address or pd.isna(address):
            return False
        return bool(self.business_addr_pattern.search(str(address)))

    def extract_entity_details(self, owner_name: str) -> Dict:
        """
        Extract detailed entity information from owner name.

        Returns dict with:
        - entity_type: Classification
        - is_investor_entity: Boolean
        - has_investor_name: Boolean
        - normalized_name: Cleaned name
        """
        entity_type = self.classify_entity(owner_name)

        return {
            'entity_type': entity_type,
            'is_investor_entity': self.is_investor_entity(entity_type),
            'has_investor_name': self.has_investor_name(owner_name),
            'normalized_name': self._normalize_name(owner_name),
        }

    def _normalize_name(self, name: str) -> str:
        """
        Normalize owner name for matching/deduplication.

        - Uppercase
        - Remove extra whitespace
        - Standardize common abbreviations
        """
        if not name or pd.isna(name):
            return ''

        normalized = str(name).upper().strip()

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        # Standardize common abbreviations
        replacements = [
            (r'\bL\.?L\.?C\.?\b', 'LLC'),
            (r'\bINC\.?\b', 'INC'),
            (r'\bCORP\.?\b', 'CORP'),
            (r'\bL\.?P\.?\b', 'LP'),
            (r'\bTTEE\b', 'TRUSTEE'),
        ]

        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized

    def classify_dataframe(self, df: pd.DataFrame, owner_col: str = 'owner_name',
                          address_col: str = 'owner_address') -> pd.DataFrame:
        """
        Classify all owners in a DataFrame.

        Args:
            df: DataFrame with owner information
            owner_col: Column name containing owner names
            address_col: Column name containing owner addresses

        Returns:
            DataFrame with added classification columns
        """
        self.logger.info(f"Classifying {len(df)} property owners...")

        df = df.copy()

        # Classify entity type
        df['entity_type'] = df[owner_col].apply(self.classify_entity)

        # Flag investor entities
        df['is_investor_entity'] = df['entity_type'].apply(self.is_investor_entity)

        # Check for investor-related names
        df['has_investor_name'] = df[owner_col].apply(self.has_investor_name)

        # Check address indicators
        if address_col in df.columns:
            df['has_po_box'] = df[address_col].apply(self.has_po_box)
            df['has_business_address'] = df[address_col].apply(self.has_business_address)

        # Normalize names for matching
        df['normalized_owner'] = df[owner_col].apply(self._normalize_name)

        # Log summary
        entity_counts = df['entity_type'].value_counts()
        self.logger.info(f"Entity classification complete:")
        for entity_type, count in entity_counts.items():
            pct = count / len(df) * 100
            self.logger.info(f"  {entity_type}: {count:,} ({pct:.1f}%)")

        investor_count = df['is_investor_entity'].sum()
        self.logger.info(f"Total investor entities: {investor_count:,} ({investor_count/len(df)*100:.1f}%)")

        return df

    def get_entity_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics of entity classifications.
        """
        if 'entity_type' not in df.columns:
            df = self.classify_dataframe(df)

        summary = {
            'total_records': len(df),
            'entity_counts': df['entity_type'].value_counts().to_dict(),
            'investor_entities': int(df['is_investor_entity'].sum()),
            'investor_names': int(df['has_investor_name'].sum()),
        }

        if 'has_po_box' in df.columns:
            summary['po_box_addresses'] = int(df['has_po_box'].sum())

        if 'has_business_address' in df.columns:
            summary['business_addresses'] = int(df['has_business_address'].sum())

        return summary


# ========================================
# CLI Entry Point
# ========================================

if __name__ == '__main__':
    """Quick test of entity classifier"""

    test_names = [
        "JOHN SMITH",
        "ABC PROPERTIES LLC",
        "SMITH FAMILY TRUST",
        "ESTATE OF MARY JONES",
        "ACME HOLDINGS INC",
        "FIRST BAPTIST CHURCH",
        "CITY OF COLUMBUS",
        "REAL ESTATE INVESTMENT GROUP LP",
        "JANE DOE TTEE",
        "123 MAIN STREET LLC",
    ]

    classifier = EntityClassifier()

    print("\nEntity Classification Test:")
    print("-" * 60)

    for name in test_names:
        details = classifier.extract_entity_details(name)
        investor_flag = "INVESTOR" if details['is_investor_entity'] else ""
        print(f"{name:40} -> {details['entity_type']:12} {investor_flag}")
