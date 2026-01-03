"""
Investor Finder - Web Interface
Simple Flask app to view investor data and lookup Ohio SOS info
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

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
        return pd.concat(investors, ignore_index=True)
    return pd.DataFrame()


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/investors')
def get_investors():
    """API endpoint for investor data"""
    tier = request.args.get('tier', None)
    limit = int(request.args.get('limit', 100))

    df = load_investors(tier)

    if df.empty:
        return jsonify({'error': 'No data found. Run the pipeline first.', 'investors': []})

    # Select key columns
    columns = ['owner_name', 'portfolio_size', 'total_market_value', 'entity_type',
               'investor_score', 'investor_tier', 'owner_address', 'mailing_address']

    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols].head(limit)

    return jsonify({
        'count': len(df),
        'investors': df.to_dict('records')
    })


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
    app.run(debug=True, port=5000)
