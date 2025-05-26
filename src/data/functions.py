import json
import requests
import numpy as np
import time
import sqlite3
from definitions import *

def createFullName(row):
    '''
    Function to format first name and last name as "last_name, first_name"
    @param row: row of a dataframe
    '''
    return row['last_name'] + ', ' + row['first_name']

def convertDuration(row):
    '''
    Function to convert duration from MM:SS format to seconds.
    @param row: row of a dataframe
    '''
    if row['duration'] is not None:
        splits = row['duration'].split(':')
        return int(splits[0]) * 60 + int(splits[1])
    else:
        return 0
    
def getXCoord(row):
    '''
    Function to get the x coordinate of the shot
    @param row: row of a dataframe
    '''
    event_details = json.loads(row['event_details'])
    return event_details.get('xCoord')
    
def getYCoord(row):
    '''
    Function to get the y coordinate of the shot
    @param row: row of a dataframe
    '''
    event_details = json.loads(row['event_details'])
    return event_details.get('yCoord')

def getShootingPlayerId(row):
    '''
    Function to get the player id of the player who took the shot
    @param row: row of a dataframe
    '''
    event_details = json.loads(row['event_details'])
    if row['type_code'] == GOAL:
        return event_details.get('scoringPlayerId', '')
    else:
        return event_details.get('shootingPlayerId', '')
    
def getEventOwner(row):
    '''
    Function to get the team that owns the event
    @param row: row of a dataframe
    '''

    if row['event_details'] is not None:
        event_details = json.loads(row['event_details'])
    else: 
        return None
    keys = ['hitteePlayerId', 'blockingPlayerId', 'servedByPlayerId', 'scoringPlayerId', 'assist1PlayerId', 'shootingPlayerId', 
     'drawnByPlayerId', 'hittingPlayerId', 'committedByPlayerId', 'assist2PlayerId', 'winningPlayerId', 'losingPlayerId']
    
    for key in keys:
        if event_details.get(key) is not None:
            return event_details.get(key)
        
def fetch_nhl_api(endpoint, params=None):
    '''
    Fetch data from the NHL API (updated for new API).
    @param endpoint: API endpoint (e.g., '/schedule/now')
    @param params: dict of query parameters
    '''
    base_url = 'https://api-web.nhle.com/v1'
    url = base_url + endpoint
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def rate_limited_get(url, params=None, delay=0.5, max_retries=3):
    '''
    Make a GET request with rate limiting and retries.
    '''
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(delay)
            return response
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (attempt + 1))

def get_game_ids_for_season(season):
    '''
    Fetch all game IDs for a given NHL season using team schedule endpoints.
    @param season: Season in format '20232024'
    '''
    # Get all teams first
    teams_data = fetch_nhl_api('/season')
    
    # Get a list of team abbreviations (we'll use one team to get all games)
    # For now, let's use a known team like TOR to get the season schedule
    team_code = 'TOR'  # We can use any team since schedule shows all games
    
    endpoint = f'/club-schedule-season/{team_code}/{season}'
    try:
        data = fetch_nhl_api(endpoint)
        game_ids = []
        
        # Extract game IDs from the schedule
        if 'games' in data:
            for game in data['games']:
                game_ids.append(game['id'])
        
        return game_ids
    except Exception as e:
        print(f"Error getting schedule for {season}: {e}")
        return []

def get_all_game_ids_for_date_range(start_date, end_date):
    '''
    Get all game IDs for a date range using the schedule endpoint.
    @param start_date: Start date in YYYY-MM-DD format
    @param end_date: End date in YYYY-MM-DD format
    '''
    game_ids = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            endpoint = f'/schedule/{current_date}'
            data = fetch_nhl_api(endpoint)
            
            # Extract game IDs from daily schedule
            if 'gameWeek' in data:
                for week in data['gameWeek']:
                    for date_entry in week.get('games', []):
                        for game in date_entry.get('games', []):
                            game_ids.append(game['id'])
            
            # Move to next date (simplified - just increment day)
            # In production, you'd want proper date handling
            year, month, day = map(int, current_date.split('-'))
            day += 1
            if day > 31:  # Simplified - would need proper month/year handling
                break
            current_date = f"{year}-{month:02d}-{day:02d}"
            
        except Exception as e:
            print(f"Error getting schedule for {current_date}: {e}")
            break
            
    return game_ids

def init_db(db_path='nhl_stats.db'):
    '''
    Initialize SQLite database with tables for games, events, players, and teams.
    '''
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS games (
        gamePk INTEGER PRIMARY KEY,
        season TEXT,
        gameType TEXT,
        gameDate TEXT,
        homeTeamId INTEGER,
        awayTeamId INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gamePk INTEGER,
        eventIdx INTEGER,
        eventType TEXT,
        period INTEGER,
        periodTime TEXT,
        teamId INTEGER,
        playerId INTEGER,
        x REAL,
        y REAL,
        details TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS players (
        playerId INTEGER PRIMARY KEY,
        fullName TEXT,
        position TEXT,
        shootsCatches TEXT,
        birthDate TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS teams (
        teamId INTEGER PRIMARY KEY,
        name TEXT,
        abbreviation TEXT
    )''')
    conn.commit()
    conn.close()

def insert_game(conn, game):
    c = conn.cursor()
    c.execute('''INSERT OR IGNORE INTO games (gamePk, season, gameType, gameDate, homeTeamId, awayTeamId)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (game['gamePk'], game['season'], game['gameType'], game['gameDate'], game['homeTeamId'], game['awayTeamId']))
    conn.commit()

def insert_event(conn, event):
    c = conn.cursor()
    c.execute('''INSERT INTO events (gamePk, eventIdx, eventType, period, periodTime, teamId, playerId, x, y, details)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (event['gamePk'], event['eventIdx'], event['eventType'], event['period'], event['periodTime'], event['teamId'], event['playerId'], event['x'], event['y'], json.dumps(event['details'])))
    conn.commit()

def insert_player(conn, player):
    c = conn.cursor()
    c.execute('''INSERT OR IGNORE INTO players (playerId, fullName, position, shootsCatches, birthDate)
                 VALUES (?, ?, ?, ?, ?)''',
              (player['id'], player['fullName'], player['position'], player['shootsCatches'], player['birthDate']))
    conn.commit()

def insert_team(conn, team):
    c = conn.cursor()
    c.execute('''INSERT OR IGNORE INTO teams (teamId, name, abbreviation)
                 VALUES (?, ?, ?)''',
              (team['id'], team['name'], team['abbreviation']))
    conn.commit()

def fetch_and_store_game_data(gamePk, db_path='nhl_stats.db', delay=0.5):
    '''
    Fetch play-by-play and meta data for a game using new API and store in SQLite DB.
    '''
    conn = sqlite3.connect(db_path)
    
    try:
        # Use new API endpoint for play-by-play
        endpoint = f'/gamecenter/{gamePk}/play-by-play'
        feed = fetch_nhl_api(endpoint)
        
        # Extract game info from the new API structure
        game = {
            'gamePk': gamePk,
            'season': feed.get('season', ''),
            'gameType': feed.get('gameType', ''),
            'gameDate': feed.get('gameDate', ''),
            'homeTeamId': feed.get('homeTeam', {}).get('id', 0),
            'awayTeamId': feed.get('awayTeam', {}).get('id', 0)
        }
        insert_game(conn, game)
        
        # Insert events from the new API structure
        for event in feed.get('plays', []):
            # Extract coordinates from the event details
            coords = event.get('details', {})
            
            event_dict = {
                'gamePk': gamePk,
                'eventIdx': event.get('eventId', 0),
                'eventType': event.get('typeDescKey', ''),
                'period': event.get('periodDescriptor', {}).get('number', 0),
                'periodTime': event.get('timeInPeriod', ''),
                'teamId': event.get('details', {}).get('eventOwnerTeamId'),
                'playerId': None,  # Will need to extract from players array if available
                'x': coords.get('xCoord'),
                'y': coords.get('yCoord'),
                'details': event
            }
            insert_event(conn, event_dict)
            
    except Exception as e:
        print(f"Error processing game {gamePk}: {e}")
    finally:
        conn.close()
        time.sleep(delay)

# Legacy functions for old API (keeping for compatibility)
def normalize_shot_coords(x, y, team_side='left'):
    '''
    Normalize shot coordinates so all shots are toward the same net (right side).
    @param x: x coordinate
    @param y: y coordinate
    @param team_side: 'left' or 'right' (attacking direction)
    '''
    if team_side == 'left':
        return x, y
    else:
        return -x, -y

def calc_distance_to_net(x, y):
    '''
    Calculate distance from (x, y) to the center of the net (NHL standard: x=89, y=0).
    '''
    NET_X = 89
    NET_Y = 0
    return np.sqrt((x - NET_X) ** 2 + (y - NET_Y) ** 2)

def calc_angle_to_net(x, y):
    '''
    Calculate the angle (in degrees) from (x, y) to the center of the net.
    '''
    NET_X = 89
    NET_Y = 0
    dx = NET_X - x
    dy = NET_Y - y
    return np.degrees(np.arctan2(dy, dx))

def extract_event_context(event_json):
    '''
    Extract period, strength, score state, and time remaining from event JSON.
    '''
    period = event_json.get('about', {}).get('period')
    strength = event_json.get('result', {}).get('strength', {}).get('code')
    home_goals = event_json.get('about', {}).get('goals', {}).get('home')
    away_goals = event_json.get('about', {}).get('goals', {}).get('away')
    time_remaining = event_json.get('about', {}).get('periodTimeRemaining')
    return period, strength, home_goals, away_goals, time_remaining

def get_player_info(player_id):
    '''
    Fetch player info from NHL API by player ID (updated for new API).
    '''
    endpoint = f'/player/{player_id}/landing'
    data = fetch_nhl_api(endpoint)
    return data

def get_team_info(team_id):
    '''
    Fetch team info from NHL API by team ID (updated for new API).
    '''
    endpoint = f'/team/{team_id}'
    data = fetch_nhl_api(endpoint)
    return data

def parse_live_game_feed(game_pk):
    '''
    Fetch and parse the live game feed for a given gamePk (updated for new API).
    '''
    endpoint = f'/gamecenter/{game_pk}/play-by-play'
    return fetch_nhl_api(endpoint)
        