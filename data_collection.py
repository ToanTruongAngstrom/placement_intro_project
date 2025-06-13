import os
import time
import requests
import urllib.parse
import json
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail, playerindex

base_url = "https://d2c6afifpk.execute-api.eu-west-2.amazonaws.com/dev"

# Queries API endpoint for information on all players and stores it in ./data/players.json
def load_players():
    first_page = requests.get(base_url+"/players").json()
    players = first_page["results"]
    queryExecutionId = first_page["queryExecutionId"]
    if first_page["nextToken"]:
        nextToken = urllib.parse.quote(first_page["nextToken"])
    columns = urllib.parse.quote(json.dumps(first_page["columns"]).replace(" ", ""))

    while True:
        url = f"{base_url}/players?queryExecutionId={queryExecutionId}&nextToken={nextToken}&columns={columns}"
        next_page = requests.get(url).json()
        players.extend(next_page["results"])
        if next_page["nextToken"]:
            nextToken = urllib.parse.quote_plus(next_page["nextToken"])
        else:
            break

    with open("data/players.json", "w") as f:
        json.dump(players, f)

# Queries API endpoint for player performance statistics and stores them in ./data/boxscores.json
def load_boxscores():
    first_page = requests.get(base_url+"/boxscores").json()
    boxscores = first_page["results"]
    queryExecutionId = first_page["queryExecutionId"]
    if first_page["nextToken"]:
        nextToken = urllib.parse.quote(first_page["nextToken"])
    columns = urllib.parse.quote(json.dumps(first_page["columns"]).replace(" ", ""))

    while True:
        url = f"{base_url}/boxscores?queryExecutionId={queryExecutionId}&nextToken={nextToken}&columns={columns}"
        next_page = requests.get(url).json()
        boxscores.extend(next_page["results"])
        if next_page["nextToken"]:
            nextToken = urllib.parse.quote_plus(next_page["nextToken"])
        else:
            break

    with open("data/boxscores.json", "w") as f:
        json.dump(boxscores, f)

# Searches the data for the required participants, given by their first and last names.
# As of the current state of the data, this is enough to uniquely identify each participant.
def load_participant_info():
    try:
        f = open("data/players.json")
    except FileNotFoundError:
        load_players()
        f = open("data/players.json")

    players = json.load(f)
    f.close()

    participants = [{"firstname": "Jalen", "surname": "Brunson"}, 
                    {"firstname": "Cade", "surname": "Cunningham"},
                    {"firstname": "Darius", "surname": "Garland"},
                    {"firstname": "Tyler", "surname": "Herro"},
                    {"firstname": "Buddy", "surname": "Hield"},
                    {"firstname": "Cameron", "surname": "Johnson"},
                    {"firstname": "Damian", "surname": "Lillard"},
                    {"firstname": "Norman", "surname": "Powell"}]

    participant_info = []
    for player in players:
        for participant in participants:
            if player["firstname"] == participant["firstname"] and player["surname"] == participant["surname"]:
                participant_info.append(player)

    with open("data/participants.json", "w") as f:
        json.dump(participant_info, f)

load_participant_info()

def load_results():
    first_page = requests.get(base_url+"/three_pt_contest_historical_results").json()
    results = first_page["results"]
    queryExecutionId = first_page["queryExecutionId"]
    columns = urllib.parse.quote(json.dumps(first_page["columns"]).replace(" ", ""))

    if first_page["nextToken"]:
        nextToken = urllib.parse.quote(first_page["nextToken"])

        while True:
            url = f"{base_url}/three_pt_contest_historical_results?queryExecutionId={queryExecutionId}&nextToken={nextToken}&columns={columns}"
            next_page = requests.get(url).json()
            results.extend(next_page["results"])
            if next_page["nextToken"]:
                nextToken = urllib.parse.quote_plus(next_page["nextToken"])
            else:
                break

    with open("data/results.json", "w") as f:
        json.dump(results, f)

def load_3ptfg_last_100(playerid):
    try:
        boxscores = pd.read_json("data/boxscores.json")
    except FileNotFoundError:
        load_boxscores()
        boxscores = pd.read_json("data/boxscores.json")

    if not os.path.exists("data/3ptfg_last_100.json"):
        with open("data/3ptfg_last_100.json", "w") as f:
            json.dump({}, f)

    player_boxscores = boxscores[boxscores['player'] == playerid].tail(100)
    player_boxscores["threepointers"] = pd.to_numeric(player_boxscores["threepointers"])
    player_boxscores["threepointersattempted"] = pd.to_numeric(player_boxscores["threepointersattempted"])
    made = player_boxscores["threepointers"].sum()
    att = player_boxscores["threepointersattempted"].sum()
    try:    
        pc = round(100 * float(made) / float(att), 1)
    except ZeroDivisionError:
        pc = 0
    
    new_data = {"made": int(made), "att": int(att), "3pt%": pc}
    with open("data/3ptfg_last_100.json", "r") as f:
        data = json.load(f)
        data[str(playerid)] = new_data

    with open("data/3ptfg_last_100.json", "w") as f:
        json.dump(data, f)
    
    return new_data

def get_name_by_id(playerid):
    # Search participants.json first (kind of like a cache)
    try:
        f = open("data/participants.json")
    except FileNotFoundError:
        load_participant_info()
        f = open("data/participants.json")
    
    players = json.load(f)
    f.close()

    for player in players:
        if player["playerid"] == str(playerid):
            return f"{player["firstname"]} {player["surname"]}"
        
    # If not found then search players.json
    try:
        f = open("data/players.json")
    except FileNotFoundError:
        load_participant_info()
        f = open("data/players.json")
    
    players = json.load(f)
    f.close()

    for player in players:
        if player["playerid"] == str(playerid):
            return f"{player["firstname"]} {player["surname"]}"

def load_shot_distance_data(playerid):

    # Scrape data from https://www.nba.com/stats/players/shooting by accessing the underlying API
    # Replicate the network request:
    url = "https://stats.nba.com/stats/leaguedashplayershotlocations"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
    }

    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "DistanceRange": "5ft Range",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "ISTRound": "",
        "LastNGames": "0",
        "Location": "",
        "MeasureType": "Base",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": "2024-25",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    rows = data['resultSets']['rowSet']

    # Rename columns
    shot_zones = ['<5ft', '5-9ft', '10-14ft', '15-19ft', '20-24ft', '25-29ft', '30-34ft', '35-39ft', '40+ft']
    base_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'NICKNAME']
    metrics = ['FGM', 'FGA', 'FG_PCT']
    zone_columns = [f"{metric}_{zone}" for zone in shot_zones for metric in metrics]
    all_columns = base_cols + zone_columns

    df = pd.DataFrame(rows, columns=all_columns)

    if not os.path.exists("data/shot_pc_by_dist.json"):
        with open("data/shot_pc_by_dist.json", "w") as f:
            json.dump({}, f)

    df = df[df['PLAYER_NAME'] == get_name_by_id(playerid)]
    new_data = {"fgm_20-24": df.at[df.index[0], 'FGM_20-24ft'],
                "fga_20-24": df.at[df.index[0], 'FGA_20-24ft'],
                "fg_pc_20-24": df.at[df.index[0], 'FG_PCT_20-24ft'],
                "fgm_25-29": df.at[df.index[0], 'FGM_25-29ft'],
                "fga_25-29": df.at[df.index[0], 'FGA_25-29ft'],
                "fg_pc_25-29": df.at[df.index[0], 'FG_PCT_25-29ft'],
                }
    with open("data/shot_pc_by_dist.json", "r") as f:
        data = json.load(f)
        data[str(playerid)] = new_data

    with open("data/shot_pc_by_dist.json", "w") as f:
        json.dump(data, f)
    
    return new_data

# Data collection from nba_api
def load_shot_chart(playerid):
    output_path = f"data/shot_charts/{playerid}.json"

    player_name = get_name_by_id(playerid).split(" ")
    data = playerindex.PlayerIndex().get_data_frames()[0]
    data = data[data['PLAYER_FIRST_NAME']==player_name[0]]
    data = data[data['PLAYER_LAST_NAME']==player_name[1]]
    nba_api_id = data.at[data.index[0], 'PERSON_ID']

    shots = []
    
    start_year = 2021
    end_year = 2024
    for year in range(start_year, end_year+1):
        season = f"{year}-{str(year + 1)[-2:]}"
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=nba_api_id,
            season_type_all_star='Regular Season',
            season_nullable=season,
            context_measure_simple='FG3A'
        )
        time.sleep(1)
        
        shots.append(shotchart.get_data_frames()[0])

    shot_data = pd.concat(shots)

    shot_data = shot_data[shot_data['ACTION_TYPE'] == 'Jump Shot']
    # shot_data = shot_data[['LOC_X', 'LOC_Y', 'SHOT_ZONE_AREA', 'SHOT_MADE_FLAG']]

    new_shots = shot_data.to_dict(orient='records')

    with open(output_path, "w") as f:
        json.dump(new_shots, f, indent=2)

    return new_shots

