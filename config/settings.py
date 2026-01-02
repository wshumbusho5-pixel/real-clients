"""
Investor Finder - Configuration Settings

Centralized configuration for identifying real estate investors
from public property records.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========================================
# Project Paths
# ========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / 'config'
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ========================================
# Data Source URLs
# ========================================
FRANKLIN_COUNTY_AUDITOR_URL = "https://property.franklincountyauditor.com/_web/search/commonsearch.aspx"
FRANKLIN_COUNTY_FTP_URL = "https://apps.franklincountyauditor.com/Outside_User_Files/"

# ========================================
# Scraping Settings
# ========================================
REQUEST_DELAY = int(os.getenv('REQUEST_DELAY', 2))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))

USER_AGENT = os.getenv(
    'USER_AGENT',
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Proxy settings
USE_PROXY = os.getenv('USE_PROXY', 'false').lower() == 'true'
PROXY_URL = os.getenv('PROXY_URL', '')

# ========================================
# Entity Classification Keywords
# ========================================
ENTITY_KEYWORDS = {
    'llc': ['LLC', 'L.L.C.', 'LIMITED LIABILITY'],
    'corporation': ['INC', 'INCORPORATED', 'CORP', 'CORPORATION'],
    'trust': ['TRUST', 'TRUSTEE', 'TTEE', 'LIVING TRUST', 'REVOCABLE TRUST'],
    'estate': ['ESTATE OF', 'EXECUTOR', 'EXECUTRIX', 'CONSERVATOR', 'ADMINISTRATOR'],
    'partnership': ['LP', 'LLP', 'PLLC', 'PARTNERSHIP', 'PARTNERS'],
    'nonprofit': ['CHURCH', 'NONPROFIT', 'FOUNDATION', 'CHARITY', 'ASSOCIATION', 'MINISTRY'],
    'government': ['CITY OF', 'COUNTY OF', 'STATE OF', 'SCHOOL', 'BOARD OF', 'HOUSING AUTHORITY'],
}

# Investor name patterns (high confidence)
INVESTOR_NAME_PATTERNS = [
    'PROPERTIES', 'PROPERTY', 'REAL ESTATE', 'REALTY',
    'INVESTMENTS', 'INVESTING', 'INVESTMENT',
    'HOLDINGS', 'HOLDING', 'CAPITAL',
    'RENTALS', 'RENTAL', 'HOMES', 'HOUSES',
    'ACQUISITIONS', 'VENTURES', 'GROUP',
]

# ========================================
# Investor Scoring Weights
# ========================================
INVESTOR_SCORE_WEIGHTS = {
    'entity_type': 30,           # LLC/Corp/Trust = likely investor
    'portfolio_size': 25,        # Multiple properties = active investor
    'absentee_owner': 15,        # Doesn't live at property
    'business_address': 10,      # PO Box or suite number
    'recent_activity': 10,       # Recent purchases
    'investor_name': 10,         # Name contains investor keywords
}

# Portfolio thresholds
PORTFOLIO_THRESHOLDS = {
    'small': 2,      # 2-4 properties
    'medium': 5,     # 5-9 properties
    'large': 10,     # 10-24 properties
    'institutional': 25,  # 25+ properties
}

# ========================================
# Investor Tier Thresholds
# ========================================
# Score 0-100 based on likelihood of being an active investor
TIER_1_THRESHOLD = 80   # High-confidence investor (likely buyer)
TIER_2_THRESHOLD = 60   # Probable investor
TIER_3_THRESHOLD = 40   # Possible investor
TIER_4_THRESHOLD = 20   # Unlikely investor

# ========================================
# Export Settings
# ========================================
DEFAULT_EXPORT_FORMAT = os.getenv('DEFAULT_EXPORT_FORMAT', 'csv')

# CSV export columns for investor prospects
CSV_EXPORT_COLUMNS = [
    'owner_name',
    'entity_type',
    'is_investor_entity',
    'investor_score',
    'investor_tier',
    'portfolio_size',
    'properties_owned',
    'total_property_value',
    'owner_address',
    'has_po_box',
    'is_absentee',
    'recent_purchase',
    'last_purchase_date',
    'sample_properties',
]

# ========================================
# Logging Settings
# ========================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = Path(os.getenv('LOG_FILE', str(LOGS_DIR / 'investor_finder.log')))
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
RICH_LOG_FORMAT = "[%(asctime)s] %(message)s"

# ========================================
# Business Logic Constants
# ========================================
# Minimum properties to consider someone a portfolio investor
MIN_PORTFOLIO_SIZE = 2

# Maximum results to process in one session
MAX_PROCESS_RESULTS = 500000  # Process all ~486K records

# Columbus zip codes
COLUMBUS_ZIP_CODES = [
    '43201', '43202', '43203', '43204', '43205',
    '43206', '43207', '43209', '43210', '43211',
    '43212', '43213', '43214', '43215', '43219',
    '43220', '43221', '43222', '43223', '43224',
    '43227', '43229', '43230', '43231', '43232',
    '43235', '43240'
]

# ========================================
# Environment
# ========================================
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

# ========================================
# Version
# ========================================
VERSION = '1.0.0'
APP_NAME = 'Investor Finder'
DESCRIPTION = 'Identify real estate investors from public property records'
