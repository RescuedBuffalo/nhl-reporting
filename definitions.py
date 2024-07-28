# Play-by-play event types
FACEOFF = 502
HIT = 503
GIVEAWAY = 504
GOAL = 505
SHOT_ON_GOAL = 506
MISSED_SHOT = 507
BLOCKED_SHOT = 508
PENALTY = 509
STOPPAGE = 516
PERIOD_START = 520
PERIOD_END = 521
SHOOTOUT_COMPLETE = 523
GAME_END = 524
TAKEAWAY = 525
DELAYED_PENALTY = 535
FAILED_SHOT_ATTEMPT = 537

# Home and Away Codes
HOME = 1
AWAY = 2

# Situation Object
class Situation:

    def __init__(self, json):
        self.homeTeam = json['homeTeam']['abbrev']
        self.awayTeam = json['awayTeam']['abbrev']
        self.situation = self.getSituation(json)
        self.situationCode = json['situationCode']
        self.secondsRemaining = json['secondsRemaining']

    def getSituation(self, json):
        if 'situationDescriptions' in json['homeTeam']:
            return HOME, json['homeTeam']['situationDescriptions'][0], json['homeTeam']['strength'], json['awayTeam']['strength']
        elif 'situationDescriptions' in json['awayTeam']:
            return AWAY, json['awayTeam']['situationDescriptions'][0], json['homeTeam']['strength'], json['awayTeam']['strength']
        return None