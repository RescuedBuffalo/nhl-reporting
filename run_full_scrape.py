#!/usr/bin/env python3

import sys
import os
sys.path.append('Python')

from Python.functions import init_db, get_game_ids_for_season, fetch_and_store_game_data, fetch_nhl_api
from Python.scrape_nhl_data import scrape_full_seasons

def main():
    print("=== NHL FULL SEASON SCRAPING ===")
    
    # Configuration
    seasons = ['20232024']  # Full 2023-24 season
    db_path = 'nhl_stats.db'
    test_mode = False  # Full scraping
    
    print(f"Scraping full season: {seasons[0]}")
    print(f"Database: {db_path}")
    print(f"This will take approximately 1-2 hours...")
    
    # Initialize database
    init_db(db_path)
    
    try:
        games_processed, games_failed = scrape_full_seasons(
            seasons=seasons,
            db_path=db_path,
            delay=1.0,  # 1 second delay between requests
            test_mode=test_mode
        )
        
        print(f"\n🎉 Full season scraping completed!")
        print(f"Games processed: {games_processed}")
        print(f"Games failed: {games_failed}")
        print(f"Run 'python Python/verify_data.py' to check your data.")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Scraping interrupted by user.")
        print(f"Partial data has been saved to {db_path}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"Partial data may have been saved to {db_path}")

if __name__ == '__main__':
    main() 