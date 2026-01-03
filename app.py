"""
Investor Finder - Web Interface
Simple Flask app to view investor data and lookup Ohio SOS info
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
from datetime import datetime

app = Flask(__name__)

# Saved contacts file
SAVED_CONTACTS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'saved_contacts.json')


def load_saved_contacts():
    """Load saved contacts from file"""
    try:
        if os.path.exists(SAVED_CONTACTS_FILE):
            with open(SAVED_CONTACTS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return []


def save_contacts_to_file(contacts):
    """Save contacts to file"""
    os.makedirs(os.path.dirname(SAVED_CONTACTS_FILE), exist_ok=True)
    with open(SAVED_CONTACTS_FILE, 'w') as f:
        json.dump(contacts, f, indent=2)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')


def load_investors(tier=None):
    """Load investor data from CSV files"""
    investors = []

    files = {
        'tier_1': 'investor_prospects_tier_1.csv',
        'tier_2': 'investor_prospects_tier_2.csv',
        'tier_3': 'investor_prospects_tier_3.csv',
    }

    if tier and tier in files:
        files = {tier: files[tier]}

    for tier_name, filename in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['tier'] = tier_name
            investors.append(df)

    if investors:
        df = pd.concat(investors, ignore_index=True)
        # Replace NaN with empty string to avoid JSON issues
        df = df.fillna('')
        return df
    return pd.DataFrame()


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/investors')
def get_investors():
    """API endpoint for investor data"""
    tier = request.args.get('tier', None)
    entity_type = request.args.get('entity_type', None)
    limit = int(request.args.get('limit', 100))

    df = load_investors(tier)

    if df.empty:
        return jsonify({'error': 'No data found. Run the pipeline first.', 'investors': []})

    # Filter by entity type if specified
    if entity_type and 'entity_type' in df.columns:
        df = df[df['entity_type'] == entity_type]

    # Select key columns
    columns = ['owner_name', 'portfolio_size', 'total_market_value', 'entity_type',
               'investor_score', 'investor_tier', 'owner_address', 'mailing_address']

    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols].head(limit)

    return jsonify({
        'count': len(df),
        'investors': df.to_dict('records')
    })


@app.route('/api/entity-types')
def get_entity_types():
    """Get list of unique entity types"""
    df = load_investors()
    if df.empty or 'entity_type' not in df.columns:
        return jsonify({'types': []})

    types = df['entity_type'].dropna().unique().tolist()
    return jsonify({'types': sorted(types)})


@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    """Run the investor identification pipeline"""
    try:
        from data_processing.investor_pipeline import InvestorPipeline

        pipeline = InvestorPipeline()
        df = pipeline.run()

        # Export investors to CSV files
        pipeline.export_investors(df)

        # Count tiers
        tier_counts = {'tier_1': 0, 'tier_2': 0, 'tier_3': 0}
        if 'investor_tier' in df.columns:
            tier_counts = df['investor_tier'].value_counts().to_dict()

        return jsonify({
            'success': True,
            'message': 'Pipeline completed successfully',
            'results': {
                'total_parcels': len(df) if df is not None else 0,
                'total_investors': len(df) if df is not None else 0,
                'tier_1': tier_counts.get('tier_1', 0),
                'tier_2': tier_counts.get('tier_2', 0),
                'tier_3': tier_counts.get('tier_3', 0),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sos-lookup', methods=['POST'])
def sos_lookup():
    """Look up business on Ohio SOS"""
    data = request.get_json()
    business_name = data.get('name', '')

    if not business_name:
        return jsonify({'error': 'Business name required'})

    try:
        from scrapers.ohio_sos import OhioSOSScraper
        scraper = OhioSOSScraper(headless=False)
        result = scraper.search_business(business_name)

        if result:
            return jsonify({
                'success': True,
                'data': result.to_dict()
            })
        else:
            return jsonify({'success': False, 'error': 'Not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/skip-trace', methods=['POST'])
def skip_trace_lookup():
    """Skip trace to get phone/email contacts"""
    data = request.get_json()
    name = data.get('name', '')
    address = data.get('address', '')

    if not name:
        return jsonify({'error': 'Name required'})

    try:
        from scrapers.skip_tracing import SkipTracer
        tracer = SkipTracer(provider="free")
        result = tracer.lookup(name, address)

        if result and result.has_contact():
            return jsonify({
                'success': True,
                'data': result.to_dict()
            })
        else:
            return jsonify({'success': False, 'error': 'No contact info found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/full-lookup', methods=['POST'])
def full_lookup():
    """
    Smart lookup based on entity type:
    - LLC/Corp/Trust → Ohio SOS → Agent Name → Skip Trace
    - Individual → Skip Trace directly using name + address
    """
    data = request.get_json()
    owner_name = data.get('name', '')
    owner_address = data.get('address', '')
    entity_type = data.get('entity_type', '').lower()

    if not owner_name:
        return jsonify({'error': 'Name required'})

    result = {
        'llc_name': owner_name,
        'entity_type': entity_type,
        'sos_data': None,
        'agent_name': None,
        'agent_address': None,
        'contact_data': None,
        'phone': None,
        'email': None,
        'steps': []
    }

    # Determine if this is a business entity or individual
    is_business = entity_type in ['llc', 'corporation', 'trust', 'partnership']

    # Also check name patterns if entity_type not provided
    if not entity_type:
        business_keywords = ['LLC', 'L.L.C', 'INC', 'CORP', 'TRUST', 'LP', 'LTD', 'COMPANY', 'PARTNERS']
        is_business = any(kw in owner_name.upper() for kw in business_keywords)

    try:
        if is_business:
            # BUSINESS FLOW: Ohio SOS → Agent → Skip Trace
            result['steps'].append(f'Detected business entity ({entity_type or "LLC/Corp"})')
            result['steps'].append('Looking up on Ohio SOS...')

            from scrapers.ohio_sos import OhioSOSScraper
            scraper = OhioSOSScraper(headless=False)
            sos_result = scraper.search_business(owner_name)

            if not sos_result:
                result['steps'].append('Not found on Ohio SOS')
                return jsonify({'success': False, 'error': 'Business not found on Ohio SOS', 'data': result})

            result['sos_data'] = sos_result.to_dict()
            result['agent_name'] = sos_result.agent_name
            result['agent_address'] = sos_result.agent_address
            result['steps'].append(f'Found agent: {sos_result.agent_name}')

            if not sos_result.agent_name:
                result['steps'].append('No agent name found')
                return jsonify({'success': True, 'data': result, 'message': 'Found business but no agent name'})

            # Check if agent is a service company
            service_companies = ['NATIONAL REGISTERED AGENTS', 'CT CORPORATION', 'CSC',
                               'REGISTERED AGENT SOLUTIONS', 'INCORP SERVICES', 'NORTHWEST REGISTERED',
                               'COGENCY GLOBAL', 'LEGALINC', 'HARVARD BUSINESS']
            is_service = any(svc in sos_result.agent_name.upper() for svc in service_companies)

            if is_service:
                result['steps'].append('Agent is a registered agent service - skip tracing not useful')
                return jsonify({
                    'success': True,
                    'data': result,
                    'message': 'Agent is a service company, not a person'
                })

            # Skip Trace the Agent
            result['steps'].append(f'Skip tracing agent: {sos_result.agent_name}')
            skip_name = sos_result.agent_name
            skip_address = sos_result.agent_address

        else:
            # INDIVIDUAL FLOW: Skip Trace directly
            result['steps'].append('Detected individual owner')
            result['steps'].append(f'Skip tracing: {owner_name}')
            skip_name = owner_name
            skip_address = owner_address
            result['agent_name'] = owner_name
            result['agent_address'] = owner_address

        # Perform Skip Trace
        from scrapers.skip_tracing import SkipTracer
        tracer = SkipTracer(provider="free")

        # Parse city/state from address
        city, state = "", "OH"
        if skip_address:
            import re
            addr_match = re.search(r'([A-Za-z\s]+)\s+([A-Z]{2})\s*\d*', skip_address)
            if addr_match:
                city = addr_match.group(1).strip()
                state = addr_match.group(2)

        contact = tracer.lookup(skip_name, skip_address, city, state)

        if contact and contact.has_contact():
            result['contact_data'] = contact.to_dict()
            result['phone'] = contact.phone_primary
            result['email'] = contact.email_primary
            result['steps'].append(f'Found contact: {contact.phone_primary or contact.email_primary}')
            return jsonify({'success': True, 'data': result})
        else:
            result['steps'].append('No contact info found')
            return jsonify({
                'success': True,
                'data': result,
                'message': 'Could not find contact info'
            })

    except Exception as e:
        result['steps'].append(f'Error: {str(e)}')
        return jsonify({'success': False, 'error': str(e), 'data': result})


@app.route('/api/saved-contacts')
def get_saved_contacts():
    """Get all saved contacts"""
    contacts = load_saved_contacts()
    return jsonify({'contacts': contacts, 'count': len(contacts)})


@app.route('/api/save-contact', methods=['POST'])
def save_contact():
    """Save a contact from lookup results"""
    data = request.get_json()

    contact = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
        'saved_at': datetime.now().isoformat(),
        'llc_name': data.get('llc_name', ''),
        'entity_number': data.get('entity_number', ''),
        'status': data.get('status', ''),
        'agent_name': data.get('agent_name', ''),
        'agent_address': data.get('agent_address', ''),
        'phone': data.get('phone', ''),
        'email': data.get('email', ''),
        'notes': data.get('notes', ''),
    }

    contacts = load_saved_contacts()
    contacts.append(contact)
    save_contacts_to_file(contacts)

    return jsonify({'success': True, 'contact': contact})


@app.route('/api/delete-contact/<contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    """Delete a saved contact"""
    contacts = load_saved_contacts()
    contacts = [c for c in contacts if c.get('id') != contact_id]
    save_contacts_to_file(contacts)
    return jsonify({'success': True})


@app.route('/api/export-contacts')
def export_contacts():
    """Export saved contacts as CSV to Downloads folder"""
    contacts = load_saved_contacts()
    if not contacts:
        return jsonify({'error': 'No contacts to export'})

    df = pd.DataFrame(contacts)
    downloads_folder = os.path.expanduser('~/Downloads')
    csv_path = os.path.join(downloads_folder, 'investor_contacts.csv')
    df.to_csv(csv_path, index=False)

    return jsonify({'success': True, 'path': csv_path, 'count': len(contacts)})


@app.route('/api/export-for-skip-tracing')
def export_for_skip_tracing():
    """Export investor data formatted for skip tracing platforms"""
    import re

    tier = request.args.get('tier', None)
    entity_type = request.args.get('entity_type', None)

    df = load_investors(tier)

    if df.empty:
        return jsonify({'error': 'No data to export'})

    # Filter by entity type if specified
    if entity_type and 'entity_type' in df.columns:
        df = df[df['entity_type'] == entity_type]

    # Prepare skip tracing format
    skip_data = []

    for _, row in df.iterrows():
        name = row.get('owner_name', '')
        address = row.get('mailing_address', '') or row.get('owner_address', '')

        # Parse name into first/last (for individuals)
        name_parts = name.strip().split()
        if len(name_parts) >= 2 and row.get('entity_type') == 'individual':
            # Assume format: LASTNAME FIRSTNAME or FIRSTNAME LASTNAME
            # Most property records use LASTNAME FIRSTNAME
            last_name = name_parts[0]
            first_name = ' '.join(name_parts[1:])
        else:
            first_name = name
            last_name = ''

        # Parse address into components
        street = address
        city = ''
        state = 'OH'
        zip_code = ''

        # Try to extract city, state, zip from address
        addr_match = re.search(r'^(.+?),?\s+([A-Za-z\s]+),?\s*([A-Z]{2})?\s*(\d{5})?', address)
        if addr_match:
            street = addr_match.group(1).strip()
            city = addr_match.group(2).strip() if addr_match.group(2) else ''
            state = addr_match.group(3) if addr_match.group(3) else 'OH'
            zip_code = addr_match.group(4) if addr_match.group(4) else ''

        skip_data.append({
            'full_name': name,
            'first_name': first_name,
            'last_name': last_name,
            'address': street,
            'city': city,
            'state': state,
            'zip': zip_code,
            'full_address': address,
            'entity_type': row.get('entity_type', ''),
            'portfolio_size': row.get('portfolio_size', ''),
            'investor_score': row.get('investor_score', ''),
            'investor_tier': row.get('investor_tier', ''),
        })

    export_df = pd.DataFrame(skip_data)
    downloads_folder = os.path.expanduser('~/Downloads')
    csv_path = os.path.join(downloads_folder, 'investors_for_skip_tracing.csv')
    export_df.to_csv(csv_path, index=False)

    return jsonify({
        'success': True,
        'path': csv_path,
        'count': len(export_df),
        'message': f'Exported {len(export_df)} investors to Downloads folder'
    })


@app.route('/api/stats')
def get_stats():
    """Get summary statistics"""
    df = load_investors()

    if df.empty:
        return jsonify({'error': 'No data found'})

    stats = {
        'total_investors': len(df),
        'tier_1_count': len(df[df['investor_tier'] == 'tier_1']) if 'investor_tier' in df.columns else 0,
        'tier_2_count': len(df[df['investor_tier'] == 'tier_2']) if 'investor_tier' in df.columns else 0,
        'tier_3_count': len(df[df['investor_tier'] == 'tier_3']) if 'investor_tier' in df.columns else 0,
        'total_properties': int(df['portfolio_size'].sum()) if 'portfolio_size' in df.columns else 0,
        'total_value': float(df['total_market_value'].sum()) if 'total_market_value' in df.columns else 0,
    }

    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
