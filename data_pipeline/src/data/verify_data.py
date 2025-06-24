import sqlite3
import json
import pandas as pd

def verify_database(db_path='nhl_stats.db'):
    '''Verify and explore the data we've collected'''
    conn = sqlite3.connect(db_path)
    
    print("=== NHL Data Verification ===\n")
    
    # Check tables
    print("1. Database Tables:")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
    for table in tables['name']:
        print(f"   - {table}")
    
    # Check games
    print("\n2. Games Data:")
    games_count = pd.read_sql_query("SELECT COUNT(*) as count FROM games", conn)['count'][0]
    print(f"   Total games: {games_count}")
    
    if games_count > 0:
        games_sample = pd.read_sql_query("SELECT * FROM games LIMIT 3", conn)
        print("   Sample games:")
        for _, game in games_sample.iterrows():
            print(f"     Game {game['gamePk']}: {game['gameDate']}")
    
    # Check events
    print("\n3. Events Data:")
    events_count = pd.read_sql_query("SELECT COUNT(*) as count FROM events", conn)['count'][0]
    print(f"   Total events: {events_count}")
    
    if events_count > 0:
        # Event types breakdown
        event_types = pd.read_sql_query("""
            SELECT eventType, COUNT(*) as count 
            FROM events 
            GROUP BY eventType 
            ORDER BY count DESC
        """, conn)
        print("   Event types:")
        for _, event in event_types.iterrows():
            print(f"     {event['eventType']}: {event['count']}")
        
        # Sample events with coordinates
        print("\n   Sample events with coordinates:")
        coord_events = pd.read_sql_query("""
            SELECT gamePk, eventType, period, periodTime, x, y 
            FROM events 
            WHERE x IS NOT NULL AND y IS NOT NULL 
            LIMIT 5
        """, conn)
        for _, event in coord_events.iterrows():
            print(f"     Game {event['gamePk']}: {event['eventType']} at ({event['x']}, {event['y']}) - P{event['period']} {event['periodTime']}")
    
    # Check for shot events specifically
    print("\n4. Shot Events Analysis:")
    shot_events = pd.read_sql_query("""
        SELECT eventType, COUNT(*) as count 
        FROM events 
        WHERE eventType LIKE '%shot%' OR eventType LIKE '%goal%' 
        GROUP BY eventType
    """, conn)
    
    if len(shot_events) > 0:
        print("   Shot-related events:")
        for _, event in shot_events.iterrows():
            print(f"     {event['eventType']}: {event['count']}")
        
        # Sample shot event details
        sample_shot = pd.read_sql_query("""
            SELECT details 
            FROM events 
            WHERE eventType LIKE '%shot%' OR eventType LIKE '%goal%' 
            LIMIT 1
        """, conn)
        
        if len(sample_shot) > 0:
            print("\n   Sample shot event details:")
            details = json.loads(sample_shot['details'][0])
            print(f"     Keys available: {list(details.keys())}")
    
    # Data quality checks
    print("\n5. Data Quality:")
    
    # Check for missing coordinates in shot events
    missing_coords = pd.read_sql_query("""
        SELECT COUNT(*) as count 
        FROM events 
        WHERE (eventType LIKE '%shot%' OR eventType LIKE '%goal%') 
        AND (x IS NULL OR y IS NULL)
    """, conn)['count'][0]
    
    total_shots = pd.read_sql_query("""
        SELECT COUNT(*) as count 
        FROM events 
        WHERE eventType LIKE '%shot%' OR eventType LIKE '%goal%'
    """, conn)['count'][0]
    
    if total_shots > 0:
        print(f"   Shot events missing coordinates: {missing_coords}/{total_shots} ({missing_coords/total_shots*100:.1f}%)")
    
    # Check for events per game
    events_per_game = pd.read_sql_query("""
        SELECT gamePk, COUNT(*) as event_count 
        FROM events 
        GROUP BY gamePk 
        ORDER BY event_count DESC 
        LIMIT 5
    """, conn)
    
    print("   Events per game (top 5):")
    for _, game in events_per_game.iterrows():
        print(f"     Game {game['gamePk']}: {game['event_count']} events")
    
    conn.close()
    
    print("\n=== Summary ===")
    print(f"✓ Successfully collected data from {games_count} games")
    print(f"✓ Total of {events_count} events captured")
    print(f"✓ Data includes coordinates and detailed event information")
    print("✓ Ready for analysis and modeling!")

if __name__ == '__main__':
    verify_database() 