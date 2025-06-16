# How likely is each player to advance from the qualifying round?
import glob
import json
import os
from data_collection import load_participant_info
from simulate import sim_qualifiers


def chance_of_qualifying(playerid, model="bayesian", n=1000):
    try:
        f = open("data/participants.json")
    except FileNotFoundError:
        load_participant_info()
        f = open("data/participants.json")
    
    participants = json.load(f)
    f.close()
    # Delete any old stored logistic regression models/data
    if model == "log_reg":
        shot_charts = glob.glob("data/shot_charts/*.json")
        models = glob.glob("data/models/*.joblib")
        for file in shot_charts + models:
            os.remove(file)

    qualified = 0

    for _ in range(n):
        finalists = sim_qualifiers(participants, model)
        # print(finalists)
        if str(playerid) in [finalist[0] for finalist in finalists]:
            qualified += 1

    return qualified/n

print(chance_of_qualifying(1815))