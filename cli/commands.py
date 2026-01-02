"""
Investor Finder - Command-Line Interface

CLI for identifying real estate investors from property records.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path

from config.settings import VERSION, APP_NAME, RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import setup_logging, log_success, log_failure

console = Console()


@click.group()
@click.version_option(version=VERSION, prog_name=APP_NAME)
def cli():
    """
    Investor Finder - Identify real estate investors from property records.

    Find portfolio investors, LLC owners, and active buyers to market your
    lead generation services to.
    """
    setup_logging()
    console.print(Panel.fit(
        f"[bold cyan]{APP_NAME}[/bold cyan] v{VERSION}\n"
        "[dim]Find Real Estate Investors from Property Records[/dim]",
        border_style="cyan"
    ))


@cli.command()
@click.option('--columbus-only/--all-counties', default=True,
              help='Filter to Columbus zip codes only')
@click.option('--min-portfolio', default=2, type=int,
              help='Minimum properties to be considered an investor')
@click.option('--min-score', default=40, type=int,
              help='Minimum investor score (0-100)')
@click.option('--max-results', default=None, type=int,
              help='Maximum investors to return')
@click.option('--export/--no-export', default=True,
              help='Export results to CSV')
def find(columbus_only, min_portfolio, min_score, max_results, export):
    """
    Find real estate investors from property records.

    Analyzes property ownership data to identify:
    - LLC/Corp/Trust entities (likely investors)
    - Portfolio owners (multiple properties)
    - Absentee owners
    - Recent buyers

    Example:
        python main.py find --min-portfolio 3 --min-score 60
    """
    from data_processing.investor_pipeline import InvestorPipeline

    console.print("\n[bold green]Searching for investors...[/bold green]\n")
    console.print(f"  Columbus only: {columbus_only}")
    console.print(f"  Min portfolio size: {min_portfolio}")
    console.print(f"  Min score: {min_score}")
    console.print()

    pipeline = InvestorPipeline()

    # Run pipeline
    investors = pipeline.run(
        columbus_only=columbus_only,
        min_portfolio_size=min_portfolio,
        min_score=min_score,
        max_results=max_results
    )

    # Print summary
    pipeline.print_summary(investors)

    # Export if requested
    if export and not investors.empty:
        exported = pipeline.export_investors(investors)
        console.print(f"\n[bold green]Exported to:[/bold green]")
        for f in exported:
            console.print(f"  [cyan]{f}[/cyan]")


@cli.command()
@click.option('--top', default=50, type=int, help='Number of top investors to show')
def top(top):
    """
    Show top investors by score.

    Quick view of the highest-scoring investor prospects.

    Example:
        python main.py top --top 20
    """
    from data_processing.investor_pipeline import InvestorPipeline

    console.print(f"\n[bold cyan]Loading data...[/bold cyan]")

    pipeline = InvestorPipeline()
    investors = pipeline.run(columbus_only=True, min_portfolio_size=2, min_score=0)

    if investors.empty:
        console.print("[red]No investors found.[/red]")
        return

    top_investors = investors.nlargest(top, 'investor_score')

    # Create table
    table = Table(title=f"Top {top} Investors by Score", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Owner Name", style="white", width=42)
    table.add_column("Props", justify="right", style="cyan", width=6)
    table.add_column("Score", justify="right", style="green", width=6)
    table.add_column("Tier", style="yellow", width=8)
    table.add_column("Type", style="blue", width=12)

    for i, (_, row) in enumerate(top_investors.iterrows(), 1):
        name = row['owner_name'][:40] if len(row['owner_name']) > 40 else row['owner_name']
        table.add_row(
            str(i),
            name,
            str(row.get('portfolio_size', 0)),
            str(row.get('investor_score', 0)),
            row.get('investor_tier', 'N/A'),
            row.get('entity_type', 'N/A')
        )

    console.print("\n")
    console.print(table)
    console.print()


@cli.command()
def stats():
    """
    Show statistics about property data.

    Displays summary of property records, entity types,
    and portfolio distributions.
    """
    from data_processing.investor_pipeline import InvestorPipeline

    console.print("\n[bold cyan]Loading property data...[/bold cyan]")

    pipeline = InvestorPipeline()
    investors = pipeline.run(columbus_only=True, min_portfolio_size=1, min_score=0)

    if investors.empty:
        console.print("[red]No data available.[/red]")
        return

    stats = pipeline.get_pipeline_stats(investors)

    # Create stats table
    table = Table(title="Property Data Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", justify="right", style="green", width=20)

    table.add_row("Total Unique Owners", f"{stats['total_investors']:,}")
    table.add_row("Total Properties Owned", f"{stats.get('total_properties_owned', 0):,}")
    table.add_row("Average Portfolio Size", f"{stats.get('avg_portfolio_size', 0):.1f}")
    table.add_row("Average Investor Score", f"{stats.get('avg_score', 0):.1f}")

    table.add_row("", "")
    table.add_row("[bold]Tier Distribution", "")
    for tier, count in sorted(stats.get('tier_distribution', {}).items()):
        table.add_row(f"  {tier}", f"{count:,}")

    table.add_row("", "")
    table.add_row("[bold]Entity Types", "")
    for entity, count in stats.get('entity_distribution', {}).items():
        table.add_row(f"  {entity.title()}", f"{count:,}")

    table.add_row("", "")
    table.add_row("[bold]Portfolio Sizes", "")
    for category, count in stats.get('portfolio_distribution', {}).items():
        table.add_row(f"  {category.title()}", f"{count:,}")

    console.print("\n")
    console.print(table)
    console.print()


@cli.command()
@click.argument('owner_name')
def lookup(owner_name):
    """
    Look up a specific owner's investor profile.

    Example:
        python main.py lookup "ABC PROPERTIES LLC"
    """
    from data_processing.investor_pipeline import InvestorPipeline

    console.print(f"\n[bold cyan]Searching for: {owner_name}[/bold cyan]\n")

    pipeline = InvestorPipeline()
    investors = pipeline.run(columbus_only=True, min_portfolio_size=1, min_score=0)

    if investors.empty:
        console.print("[red]No data available.[/red]")
        return

    # Search for owner (case-insensitive)
    matches = investors[
        investors['owner_name'].str.upper().str.contains(owner_name.upper(), na=False)
    ]

    if matches.empty:
        console.print(f"[yellow]No investor found matching: {owner_name}[/yellow]")
        return

    console.print(f"[green]Found {len(matches)} match(es):[/green]\n")

    for _, row in matches.iterrows():
        console.print(Panel(
            f"[bold]{row['owner_name']}[/bold]\n\n"
            f"Entity Type: {row.get('entity_type', 'N/A')}\n"
            f"Portfolio Size: {row.get('portfolio_size', 'N/A')} properties\n"
            f"Investor Score: {row.get('investor_score', 'N/A')}/100\n"
            f"Tier: {row.get('investor_tier', 'N/A')}\n"
            f"Address: {row.get('owner_address', 'N/A')}\n\n"
            f"[dim]Reasons: {row.get('score_reasons', 'N/A')}[/dim]",
            title="Investor Profile",
            border_style="cyan"
        ))


@cli.command()
@click.option('--tier', type=click.Choice(['tier_1', 'tier_2', 'tier_3', 'all']),
              default='all', help='Export specific tier')
@click.option('--format', 'fmt', type=click.Choice(['csv', 'json']),
              default='csv', help='Export format')
def export(tier, fmt):
    """
    Export investor prospects to file.

    Example:
        python main.py export --tier tier_1 --format csv
    """
    from data_processing.investor_pipeline import InvestorPipeline

    console.print(f"\n[bold yellow]Exporting investors...[/bold yellow]\n")

    pipeline = InvestorPipeline()
    investors = pipeline.run(columbus_only=True, min_portfolio_size=2, min_score=40)

    if investors.empty:
        console.print("[red]No investors to export.[/red]")
        return

    # Filter by tier if specified
    if tier != 'all':
        investors = investors[investors['investor_tier'] == tier]
        if investors.empty:
            console.print(f"[yellow]No {tier} investors found.[/yellow]")
            return

    # Export
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if fmt == 'csv':
        filename = f"investors_{tier}.csv" if tier != 'all' else "investors_all.csv"
        filepath = PROCESSED_DATA_DIR / filename
        investors.to_csv(filepath, index=False)
    else:
        filename = f"investors_{tier}.json" if tier != 'all' else "investors_all.json"
        filepath = PROCESSED_DATA_DIR / filename
        investors.to_json(filepath, orient='records', indent=2)

    console.print(f"[green]Exported {len(investors):,} investors to:[/green]")
    console.print(f"  [cyan]{filepath}[/cyan]")


@cli.command()
def check():
    """
    Check if required data files exist.

    Verifies that Franklin County Excel files are present.
    """
    console.print("\n[bold cyan]Checking data files...[/bold cyan]\n")

    required_files = ['Parcel.xlsx', 'Value.xlsx']
    optional_files = ['TaxDetail.xlsx']

    all_present = True

    for filename in required_files:
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            console.print(f"  [green][OK][/green] {filename} ({size_mb:.1f} MB)")
        else:
            console.print(f"  [red][MISSING][/red] {filename}")
            all_present = False

    for filename in optional_files:
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            console.print(f"  [green][OK][/green] {filename} ({size_mb:.1f} MB) [dim](optional)[/dim]")
        else:
            console.print(f"  [dim][--][/dim] {filename} [dim](optional, not present)[/dim]")

    if all_present:
        console.print("\n[bold green]All required data files present![/bold green]")
    else:
        console.print("\n[bold red]Missing required files![/bold red]")
        console.print(f"Please place files in: [cyan]{RAW_DATA_DIR}[/cyan]")
        console.print("\nDownload from: https://apps.franklincountyauditor.com/Outside_User_Files/")


@cli.command()
def test_classify():
    """
    Test entity classification on sample names.

    Quick test to verify entity classifier is working.
    """
    from investor_identification import EntityClassifier

    test_names = [
        "JOHN SMITH",
        "ABC PROPERTIES LLC",
        "SMITH FAMILY TRUST",
        "ESTATE OF MARY JONES",
        "ACME HOLDINGS INC",
        "FIRST BAPTIST CHURCH",
        "CITY OF COLUMBUS",
        "REAL ESTATE INVESTMENT GROUP LP",
    ]

    classifier = EntityClassifier()

    table = Table(title="Entity Classification Test", show_header=True, header_style="bold cyan")
    table.add_column("Owner Name", style="white", width=40)
    table.add_column("Entity Type", style="yellow", width=15)
    table.add_column("Investor?", style="green", width=10)

    for name in test_names:
        details = classifier.extract_entity_details(name)
        investor = "YES" if details['is_investor_entity'] else "no"
        table.add_row(name, details['entity_type'], investor)

    console.print("\n")
    console.print(table)
    console.print()


@cli.command()
@click.argument('business_name')
@click.option('--show-browser', is_flag=True, help='Show browser window (for debugging)')
def sos_lookup(business_name, show_browser):
    """
    Look up LLC/Corp details from Ohio Secretary of State.

    Gets registered agent name and address - the actual person behind the LLC.

    Example:
        python main.py sos-lookup "ABC PROPERTIES LLC"
    """
    try:
        from scrapers.ohio_sos import OhioSOSScraper, PLAYWRIGHT_AVAILABLE
    except ImportError:
        console.print("[red]Ohio SOS scraper not available[/red]")
        return

    if not PLAYWRIGHT_AVAILABLE:
        console.print("[red]Playwright not installed![/red]")
        console.print("Run: [cyan]pip install playwright && playwright install chromium[/cyan]")
        return

    console.print(f"\n[bold cyan]Looking up: {business_name}[/bold cyan]\n")

    scraper = OhioSOSScraper(headless=not show_browser, delay=2.0)
    result = scraper.search_business(business_name)

    if result:
        console.print(Panel(
            f"[bold]{result.entity_name}[/bold]\n\n"
            f"Entity Number: {result.entity_number or 'N/A'}\n"
            f"Type: {result.entity_type or 'N/A'}\n"
            f"Status: {result.status or 'N/A'}\n"
            f"Formation Date: {result.formation_date or 'N/A'}\n\n"
            f"[bold cyan]Registered Agent:[/bold cyan]\n"
            f"  {result.agent_name or 'N/A'}\n"
            f"  {result.agent_address or 'N/A'}\n\n"
            f"[bold cyan]Principal Address:[/bold cyan]\n"
            f"  {result.principal_address or 'N/A'}",
            title="Ohio Secretary of State",
            border_style="green"
        ))
    else:
        console.print(f"[yellow]No results found for: {business_name}[/yellow]")


@cli.command()
@click.option('--tier', type=click.Choice(['tier_1', 'tier_2', 'all']),
              default='tier_1', help='Which tier of investors to enrich')
@click.option('--limit', default=10, type=int, help='Max number of lookups')
@click.option('--show-browser', is_flag=True, help='Show browser window')
def enrich_sos(tier, limit, show_browser):
    """
    Enrich investor prospects with Ohio SOS data.

    Looks up registered agent info for LLC/Corp investors.

    Example:
        python main.py enrich-sos --tier tier_1 --limit 20
    """
    import pandas as pd

    try:
        from scrapers.ohio_sos import OhioSOSScraper, PLAYWRIGHT_AVAILABLE
    except ImportError:
        console.print("[red]Ohio SOS scraper not available[/red]")
        return

    if not PLAYWRIGHT_AVAILABLE:
        console.print("[red]Playwright not installed![/red]")
        console.print("Run: [cyan]pip install playwright && playwright install chromium[/cyan]")
        return

    # Load existing investor prospects
    if tier == 'all':
        filepath = PROCESSED_DATA_DIR / 'investor_prospects.csv'
    else:
        filepath = PROCESSED_DATA_DIR / f'investor_prospects_{tier}.csv'

    if not filepath.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        console.print("Run [cyan]python main.py find[/cyan] first to generate investor prospects.")
        return

    df = pd.read_csv(filepath)

    # Filter to only LLC/Corp entities (worth looking up)
    entity_types = ['llc', 'corporation', 'partnership']
    df_entities = df[df['entity_type'].isin(entity_types)].head(limit)

    if df_entities.empty:
        console.print("[yellow]No LLC/Corp entities to look up[/yellow]")
        return

    console.print(f"\n[bold cyan]Enriching {len(df_entities)} entities with Ohio SOS data...[/bold cyan]\n")

    scraper = OhioSOSScraper(headless=not show_browser, delay=3.0)

    # Results storage
    enriched_data = []

    scraper._init_browser()

    try:
        for i, (_, row) in enumerate(df_entities.iterrows()):
            name = row['owner_name']
            console.print(f"[{i+1}/{len(df_entities)}] Looking up: {name[:50]}...")

            entity = scraper.search_business(name)

            enriched = row.to_dict()
            if entity:
                enriched['sos_entity_number'] = entity.entity_number
                enriched['sos_status'] = entity.status
                enriched['sos_agent_name'] = entity.agent_name
                enriched['sos_agent_address'] = entity.agent_address
                enriched['sos_principal_address'] = entity.principal_address
                console.print(f"    [green]Found: Agent = {entity.agent_name or 'N/A'}[/green]")
            else:
                enriched['sos_entity_number'] = ''
                enriched['sos_status'] = ''
                enriched['sos_agent_name'] = ''
                enriched['sos_agent_address'] = ''
                enriched['sos_principal_address'] = ''
                console.print(f"    [yellow]Not found[/yellow]")

            enriched_data.append(enriched)

    finally:
        scraper._close_browser()

    # Save enriched data
    if enriched_data:
        enriched_df = pd.DataFrame(enriched_data)
        output_path = PROCESSED_DATA_DIR / f'investors_enriched_sos.csv'
        enriched_df.to_csv(output_path, index=False)

        console.print(f"\n[bold green]Enriched {len(enriched_data)} investors![/bold green]")
        console.print(f"Saved to: [cyan]{output_path}[/cyan]")

        # Show sample
        found_count = sum(1 for d in enriched_data if d.get('sos_agent_name'))
        console.print(f"\nFound agent info for {found_count}/{len(enriched_data)} entities")


# Entry point
if __name__ == '__main__':
    cli()
