import glob
import json
import os
from data_collection import load_participant_info
from simulate import get_participants, sim_finals, sim_qualifiers, simulate

# How likely is each player to advance from the qualifying round?
def chance_of_qualifying(playerid, model="bayesian", n=1000):
    participants = get_participants()
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

# print(chance_of_qualifying(1815))

# How likely is each player to score 25+ points in the qualifying round?
def chance_of_scoring_25(playerid, model="bayesian", n=1000):
    successes = 0
    for i in range(n):
        score = simulate(playerid, model=model)[0]
        successes += (score >= 25)
    return successes/n

# print(chance_of_scoring_25(1815))

# How likely is it that the final round record score of 29 will be broken?
def chance_of_finalist_exceeding_29(model="bayesian", n=1000):
    participants = get_participants()
    successes = 0

    # Delete any old stored logistic regression models/data
    if model == "log_reg":
        shot_charts = glob.glob("data/shot_charts/*.json")
        models = glob.glob("data/models/*.joblib")
        for file in shot_charts + models:
            os.remove(file)

    for i in range(n):
        finalists = sim_qualifiers(participants, model)
        winner = sim_finals(finalists, model)
        successes += (winner["score"] > 29)

    return successes/n

print(chance_of_finalist_exceeding_29())