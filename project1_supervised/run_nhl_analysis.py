#!/usr/bin/env python3
"""
NHL xG Analysis Runner
Simple entry point for running NHL Expected Goals analysis using the data_pipeline package.
"""

import sys
import os

# Add data_pipeline/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_pipeline', 'src'))

from analysis.run_analysis import main

if __name__ == "__main__":
    print("🏒 NHL xG Analysis - Using data_pipeline/src/ package")
    print("=" * 50)
    main() 