import sys
import time
from datetime import datetime
from functions import init_db, get_game_ids_for_season, fetch_and_store_game_data, fetch_nhl_api

def test_api_connection():
    '''Test if the NHL API is working and what the data structure looks like'''
    try:
        # Test basic API connection with a simple endpoint
        print("Testing NHL API connection...")
        data = fetch_nhl_api('/schedule/now')
        print("✓ API connection successful")
        
        # Test getting a specific date's schedule
        print("Testing schedule endpoint...")
        test_date = '2024-01-15'  # Use a date that likely had games
        schedule_data = fetch_nhl_api(f'/schedule/{test_date}')
        print(f"✓ Schedule endpoint working")
        
        # Try to find a game ID from recent schedule
        print("Looking for a test game ID...")
        
        # Try current schedule first
        current_schedule = fetch_nhl_api('/schedule/now')
        test_game_id = None
        
        # Look for games in current schedule
        if 'gameWeek' in current_schedule:
            for week in current_schedule['gameWeek']:
                for date_entry in week.get('games', []):
                    for game in date_entry.get('games', []):
                        test_game_id = game.get('id')
                        if test_game_id:
                            break
                    if test_game_id:
                        break
                if test_game_id:
                    break
        
        # If no current games, try a known game ID from recent season
        if not test_game_id:
            test_game_id = 2023020001  # First game of 2023-24 season
            print(f"Using known game ID for testing: {test_game_id}")
        else:
            print(f"Found current game ID for testing: {test_game_id}")
        
        # Test game data fetch
        print("Testing game data fetch...")
        game_data = fetch_nhl_api(f'/gamecenter/{test_game_id}/play-by-play')
        print("✓ Game data fetch successful")
        print(f"Game data keys: {list(game_data.keys())}")
        
        # Test team schedule endpoint
        print("Testing team schedule endpoint...")
        team_schedule = fetch_nhl_api('/club-schedule-season/TOR/20232024')
        print("✓ Team schedule endpoint working")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def test_season_data_fetch(season='20232024'):
    '''Test fetching game IDs for a specific season'''
    try:
        print(f"Testing season data fetch for {season}...")
        game_ids = get_game_ids_for_season(season)
        print(f"✓ Found {len(game_ids)} games for season {season}")
        
        if len(game_ids) > 0:
            print(f"Sample game IDs: {game_ids[:5]}")
            return True
        else:
            print("⚠ No game IDs found")
            return False
            
    except Exception as e:
        print(f"✗ Season data fetch failed: {e}")
        return False

def scrape_full_seasons(seasons, db_path='nhl_stats.db', delay=1.0, test_mode=False):
    '''Scrape full seasons of NHL data with progress tracking'''
    
    total_games_processed = 0
    total_games_failed = 0
    start_time = datetime.now()
    
    for season_idx, season in enumerate(seasons):
        print(f'\n{"="*60}')
        print(f'PROCESSING SEASON {season} ({season_idx + 1}/{len(seasons)})')
        print(f'{"="*60}')
        
        season_start_time = datetime.now()
        
        try:
            # Get all game IDs for the season
            print(f'Fetching game IDs for season {season}...')
            game_ids = get_game_ids_for_season(season)
            print(f'Found {len(game_ids)} games for season {season}')
            
            if len(game_ids) == 0:
                print(f"⚠ No games found for season {season}, skipping...")
                continue
            
            # In test mode, limit to first 20 games
            if test_mode:
                game_ids = game_ids[:20]
                print(f'TEST MODE: Processing first {len(game_ids)} games only')
            
            # Process each game
            games_processed = 0
            games_failed = 0
            
            for i, gamePk in enumerate(game_ids):
                try:
                    # Progress indicator
                    progress = (i + 1) / len(game_ids) * 100
                    print(f'[{season}] Game {i+1:4d}/{len(game_ids)} ({progress:5.1f}%) - ID: {gamePk}', end=' ')
                    
                    # Fetch and store game data
                    fetch_and_store_game_data(gamePk, db_path=db_path, delay=delay)
                    print('✓')
                    
                    games_processed += 1
                    total_games_processed += 1
                    
                    # Progress summary every 50 games
                    if (i + 1) % 50 == 0:
                        elapsed = datetime.now() - season_start_time
                        rate = games_processed / elapsed.total_seconds() * 60  # games per minute
                        remaining = len(game_ids) - (i + 1)
                        eta_minutes = remaining / rate if rate > 0 else 0
                        print(f'    Progress: {games_processed}/{len(game_ids)} games, {rate:.1f} games/min, ETA: {eta_minutes:.0f}min')
                    
                except Exception as e:
                    print(f'✗ Error: {e}')
                    games_failed += 1
                    total_games_failed += 1
                    
                    # If too many failures, pause longer
                    if games_failed > 5:
                        print(f'    Multiple failures detected, increasing delay...')
                        time.sleep(delay * 3)
                    
                    continue
            
            # Season summary
            season_elapsed = datetime.now() - season_start_time
            print(f'\nSeason {season} Summary:')
            print(f'  ✓ Games processed: {games_processed}')
            print(f'  ✗ Games failed: {games_failed}')
            print(f'  ⏱ Time elapsed: {season_elapsed}')
            print(f'  📊 Success rate: {games_processed/(games_processed+games_failed)*100:.1f}%')
                    
        except Exception as e:
            print(f'✗ Error processing season {season}: {e}')
            continue
    
    # Final summary
    total_elapsed = datetime.now() - start_time
    print(f'\n{"="*60}')
    print(f'SCRAPING COMPLETE')
    print(f'{"="*60}')
    print(f'Total games processed: {total_games_processed}')
    print(f'Total games failed: {total_games_failed}')
    print(f'Total time elapsed: {total_elapsed}')
    print(f'Overall success rate: {total_games_processed/(total_games_processed+total_games_failed)*100:.1f}%')
    
    return total_games_processed, total_games_failed

if __name__ == '__main__':
    print("=== NHL DATA COLLECTION - SCALED UP ===")
    
    # Test API connection first
    print("\n1. Testing API Connection...")
    if not test_api_connection():
        print("API connection test failed. Please check the API structure before proceeding.")
        sys.exit(1)
    
    print("\n2. Testing Season Data Fetch...")
    if not test_season_data_fetch():
        print("Season data fetch failed. Please check the season endpoint.")
        sys.exit(1)
    
    # Configuration
    print("\n3. Configuration...")
    
    # Full seasons to scrape (last 5 seasons)
    full_seasons = ['20192020', '20202021', '20212022', '20222023', '20232024']
    
    # Ask user for scraping mode
    print(f"Available seasons: {full_seasons}")
    print("\nScraping options:")
    print("1. Test mode (20 games per season)")
    print("2. Full scraping (all games)")
    print("3. Single season")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == '1':
        test_mode = True
        seasons = full_seasons
        print(f"TEST MODE: Will scrape 20 games from each of {len(seasons)} seasons")
    elif choice == '2':
        test_mode = False
        seasons = full_seasons
        print(f"FULL MODE: Will scrape ALL games from {len(seasons)} seasons")
        confirm = input("This will take several hours. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    elif choice == '3':
        season = input("Enter season (e.g., 20232024): ").strip()
        seasons = [season]
        test_mode = False
        print(f"SINGLE SEASON: Will scrape all games from {season}")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    # Initialize database
    db_path = 'nhl_stats.db'
    print(f"\n4. Initializing database: {db_path}")
    init_db(db_path)
    
    # Start scraping
    print(f"\n5. Starting data collection...")
    print(f"Seasons: {seasons}")
    print(f"Test mode: {test_mode}")
    print(f"Database: {db_path}")
    
    try:
        games_processed, games_failed = scrape_full_seasons(
            seasons=seasons,
            db_path=db_path,
            delay=1.0,  # 1 second delay between requests
            test_mode=test_mode
        )
        
        print(f"\n🎉 Data collection completed successfully!")
        print(f"Run 'python Python/verify_data.py' to check your data.")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Scraping interrupted by user.")
        print(f"Partial data has been saved to {db_path}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"Partial data may have been saved to {db_path}") 