# NHL Data Collection Constants

# Event type constants
GOAL = 'goal'
SHOT = 'shot-on-goal'
HIT = 'hit'
BLOCKED_SHOT = 'blocked-shot'
MISSED_SHOT = 'missed-shot'
PENALTY = 'penalty'
FACEOFF = 'faceoff'
GIVEAWAY = 'giveaway'
TAKEAWAY = 'takeaway'

# Game type constants
PRESEASON = 1
REGULAR_SEASON = 2
PLAYOFFS = 3
ALL_STAR = 4

# Team constants
NHL_TEAMS = [
    'ANA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET',
    'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT',
    'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH',
    'WPG', 'ARI'  # Arizona (now Utah)
]

# API endpoints
NHL_API_BASE = 'https://api-web.nhle.com/v1'

# Coordinate constants for NHL rink
RINK_LENGTH = 200
RINK_WIDTH = 85
NET_X = 89
NET_Y = 0

# Time constants
PERIOD_LENGTH_SECONDS = 1200  # 20 minutes
OVERTIME_LENGTH_SECONDS = 300  # 5 minutes 