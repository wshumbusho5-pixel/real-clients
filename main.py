#!/usr/bin/env python3
"""
Investor Finder - Main Entry Point

Identify real estate investors from property records.
"""

import sys
from cli.commands import cli

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print("\n\nInvestor Finder stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
