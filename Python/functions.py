import json
from functions import *

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
        