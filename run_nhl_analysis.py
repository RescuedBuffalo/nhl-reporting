#!/usr/bin/env python3
"""
NHL xG Analysis Runner
Simple entry point for running NHL Expected Goals analysis using the src package.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.run_analysis import main

if __name__ == "__main__":
    print("🏒 NHL xG Analysis - Using src/ package")
    print("=" * 50)
    main() 