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