# Investor Finder

**Identify Real Estate Investors from Property Records**

Investor Finder analyzes public property records to identify active real estate investors who could be potential customers for your lead generation services.

---

## What It Does

Investor Finder automatically:
- **Classifies** property owners by entity type (LLC, Corp, Trust, Individual)
- **Detects** portfolio investors (owners with multiple properties)
- **Scores** each owner's investor likelihood (0-100)
- **Exports** ranked investor prospects for your outreach

## Use Case

You have a lead generation product (like Aerial Leads) that sells property leads to real estate investors. Instead of cold-calling random agents or running ads, use Investor Finder to:

1. Find active investors in your market
2. Identify who owns multiple properties (serious investors)
3. Prioritize outreach to high-confidence prospects
4. Build a targeted customer acquisition list

---

## Investor Scoring Algorithm

```python
INVESTOR_SCORE = (
    Entity Type (0-30 points)      # LLC/Corp = likely investor
    + Portfolio Size (0-25 points)  # Multiple properties = active investor
    + Absentee Owner (0-15 points)  # Doesn't live at property
    + Business Address (0-10 points) # PO Box or suite number
    + Recent Activity (0-10 points)  # Recent purchases
    + Investor Name (0-10 points)    # "Properties", "Holdings" in name
) capped at 100
```

### Tier Classification

| Tier | Score | Description |
|------|-------|-------------|
| **Tier 1** | 80-100 | High-confidence investor - Prime prospect |
| **Tier 2** | 60-79 | Probable investor - Strong prospect |
| **Tier 3** | 40-59 | Possible investor - Worth contacting |
| **Tier 4** | 20-39 | Unlikely investor - Low priority |

---

## Quick Start

### Installation

```bash
cd investor-finder

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Download Franklin County property data files and place in `data/raw/`:
- `Parcel.xlsx` (required) - Property and owner information
- `Value.xlsx` (required) - Property values
- `TaxDetail.xlsx` (optional) - Tax payment details

Download from: https://apps.franklincountyauditor.com/Outside_User_Files/

### Basic Usage

```bash
# Check if data files are present
python main.py check

# Find investors (main command)
python main.py find --min-portfolio 2 --min-score 60

# Show top 50 investors
python main.py top --top 50

# Show statistics
python main.py stats

# Look up specific owner
python main.py lookup "ABC PROPERTIES LLC"

# Export to CSV
python main.py export --tier tier_1 --format csv

# Test entity classification
python main.py test-classify
```

---

## Project Structure

```
investor-finder/
├── config/                     # Configuration
│   ├── settings.py            # Main settings
│   └── logging_config.py      # Logging setup
│
├── investor_identification/    # Core identification logic
│   ├── entity_classifier.py   # LLC/Trust/Corp detection
│   ├── portfolio_detector.py  # Multi-property owner detection
│   └── investor_scorer.py     # Investor likelihood scoring
│
├── scrapers/                   # Data loading
│   ├── franklin_county_excel.py # Excel file loader
│   └── base_scraper.py        # Base scraper utilities
│
├── data_processing/            # Pipeline orchestration
│   └── investor_pipeline.py   # Main processing pipeline
│
├── cli/                        # Command-line interface
│   └── commands.py
│
├── data/                       # Data storage
│   ├── raw/                   # Input Excel files
│   └── processed/             # Output CSV files
│
└── main.py                    # Entry point
```

---

## Entity Classification

The system detects these entity types:

| Type | Keywords | Investor? |
|------|----------|-----------|
| **LLC** | LLC, L.L.C., Limited Liability | Yes |
| **Corporation** | INC, CORP, INCORPORATED | Yes |
| **Trust** | TRUST, TRUSTEE, TTEE | Yes |
| **Partnership** | LP, LLP, PLLC | Yes |
| **Estate** | ESTATE, EXECUTOR | Maybe |
| **Nonprofit** | CHURCH, FOUNDATION | No |
| **Government** | CITY OF, COUNTY OF | No |
| **Individual** | (default) | Depends on portfolio |

---

## Output Format

### CSV Export

```csv
owner_name,entity_type,portfolio_size,investor_score,investor_tier,owner_address,score_reasons
"ABC PROPERTIES LLC",llc,15,92,tier_1,"123 Main St Suite 100","LLC entity; Large portfolio (15 properties); Absentee owner"
```

### Columns

| Column | Description |
|--------|-------------|
| `owner_name` | Property owner name |
| `entity_type` | LLC, corporation, trust, individual, etc. |
| `portfolio_size` | Number of properties owned |
| `investor_score` | 0-100 likelihood score |
| `investor_tier` | tier_1, tier_2, tier_3, tier_4 |
| `owner_address` | Owner's mailing address |
| `sample_properties` | List of properties owned |
| `score_reasons` | Why they scored high/low |

---

## CLI Commands

```bash
# Main discovery command
python main.py find [OPTIONS]
  --columbus-only / --all-counties  Filter to Columbus (default: Columbus only)
  --min-portfolio INT               Minimum properties (default: 2)
  --min-score INT                   Minimum score (default: 40)
  --max-results INT                 Limit results
  --export / --no-export            Export to CSV (default: export)

# View top investors
python main.py top --top 50

# View statistics
python main.py stats

# Look up specific owner
python main.py lookup "OWNER NAME"

# Export specific tier
python main.py export --tier tier_1 --format csv

# Check data files
python main.py check

# Test classification
python main.py test-classify
```

---

## Configuration

### Scoring Weights (config/settings.py)

```python
INVESTOR_SCORE_WEIGHTS = {
    'entity_type': 30,        # LLC/Corp/Trust
    'portfolio_size': 25,     # Multiple properties
    'absentee_owner': 15,     # Doesn't live at property
    'business_address': 10,   # PO Box or suite
    'recent_activity': 10,    # Recent purchases
    'investor_name': 10,      # Keywords in name
}
```

### Portfolio Thresholds

```python
PORTFOLIO_THRESHOLDS = {
    'small': 2,           # 2-4 properties
    'medium': 5,          # 5-9 properties
    'large': 10,          # 10-24 properties
    'institutional': 25,  # 25+ properties
}
```

---

## How to Use Results

1. **Export tier_1 investors** - These are your prime prospects
2. **Look up contact info** - Use the owner address to find contact details
3. **Personalize outreach** - Reference their portfolio size and properties
4. **Offer your lead service** - They already buy properties, now sell them leads

Example outreach:

> "I noticed you own 15 properties in Franklin County. I have a service that provides pre-qualified motivated seller leads. Would you be interested in seeing what's available in your area?"

---

## Data Sources

Currently supports:
- Franklin County, Ohio (Columbus metro)

The same methodology can be applied to any county with public property records.

---

## Requirements

- Python 3.8+
- pandas
- openpyxl (for Excel files)
- click (CLI)
- rich (terminal UI)

See `requirements.txt` for full list.

---

## License

Proprietary - All Rights Reserved
