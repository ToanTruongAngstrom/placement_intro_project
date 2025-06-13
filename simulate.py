from collections import Counter
import glob
import heapq
import os
import scipy.stats
import numpy as np
import json
import matplotlib.pyplot as plt
from bayesian import get_probabilities as get_bayesian_probs
from logistic_regression import get_probabilities_by_location
from data_collection import load_participant_info, get_name_by_id

# By default, the last rack is the money rack and dew balls are shot after the second and third racks.
def simulate(playerid, model="bayesian", n=1, commentary=False, money_balls=[4, 9, 15, 21, 22, 23, 24, 25, 26], dew_balls=[10,16]):
    if model == "bayesian":
        scores = simulate_bayesian(playerid, n, commentary, money_balls, dew_balls)
    elif model == "log_reg":
        scores = simulate_log_reg(playerid, n, commentary, money_balls, dew_balls)
    else:
        print("Choose an existing model")
        return []
    return scores

def simulate_bayesian(playerid, n, commentary, money_balls, dew_balls):
    # n separate realisations of theta_reg and theta_dew according to beta posterior distribution
    thetas_reg, thetas_dew = get_bayesian_probs(n, playerid)
    scores = []

    if commentary:
        print(f"Next participant: {get_name_by_id(playerid)}")

    # For each simulation, take a value of theta_dew and theta_reg to be the shooting percentage for regular balls and dew balls.
    for i in range(n):
        thetas = [thetas_dew[i] if j in dew_balls else thetas_reg[i] for j in range(27)]
        scores.append(sim_round(thetas, commentary, money_balls, dew_balls))

    return scores

def simulate_log_reg(playerid, n, commentary, money_balls, dew_balls):
    scores = []
    shot_pc_by_location = get_probabilities_by_location(playerid)
    thetas = [shot_pc_by_location[0]]*5 + [shot_pc_by_location[1]] * 5 + [shot_pc_by_location[2]] * 5 + [shot_pc_by_location[3]] * 5 + [shot_pc_by_location[4]] * 5
    thetas.insert(dew_balls[0], shot_pc_by_location[5])
    thetas.insert(dew_balls[1], shot_pc_by_location[6])
    for i in range(n):
        scores.append(sim_round(thetas, commentary, money_balls, dew_balls))

    return scores

def sim_round(thetas, commentary, money_balls, dew_balls):
    '''
    Simulate 27 separate Bernoulli trials. 
    In future extensions, percentages may change throughout contest,
    so position of money balls and dew balls is important.
    By default, last rack is the money rack and that dew balls are shot after the
    second and third racks.
    '''
    # Scale up thetas somehow? Both models currently have players underperforming.
    score = 0
    for i in range(27):
        if i in dew_balls:
            x = scipy.stats.bernoulli.rvs(thetas[i])
            score += 3 * x
            if commentary:
                print(f"Dew ball: {"scores" if x else "misses"}")
        elif i in money_balls:
            x = scipy.stats.bernoulli.rvs(thetas[i])
            score += 2 * x
            if commentary:
                print(f"Money ball: {"scores" if x else "misses"}")
        else:
            x = scipy.stats.bernoulli.rvs(thetas[i])
            score += x
            if commentary:
                print(f"Regular ball: {"scores" if x else "misses"}")
    if commentary:
        print(f"Total score: {score}")
    return score

def simulate_contest(model="bayesian"):
    try:
        f = open("data/participants.json")
    except FileNotFoundError:
        load_participant_info()
        f = open("data/participants.json")
    
    n = 1000
    participants = json.load(f)
    f.close()

    winners = []

    # Delete any old stored logistic regression models/data
    if model == "log_reg":
        shot_charts = glob.glob("data/shot_charts/*.json")
        models = glob.glob("data/models/*.joblib")
        for file in shot_charts + models:
            os.remove(file)

    for i in range(n):
        scores = dict()
        for participant in participants:
            name = participant["firstname"] + " " + participant["surname"]
            id = participant["playerid"]
            scores[id] = {"name": name, "score": simulate(id, model=model)}
        
        # Take the 3 highest scorers
        finalists = heapq.nlargest(3, scores.items(), key=lambda i: i[1]["score"])
        final_scores = dict()
        # For the 3 finalists, simulate another round and store their scores
        for participant in finalists:
            id = participant[0]
            final_scores[id] = {"name": participant[1]["name"], "score": simulate(id, model=model)}
        # Determines the winner of the contest
        winner = max(final_scores.values(), key=lambda x:x["score"])["name"]
        winners.append(winner)

    # Calculate the implied probabilities of each player winning
    counts = Counter(winners)
    implied_probs = {key: 100 * counts[key] / n for key in counts.keys()}
    decimal_odds = {key: round(100 / implied_probs[key], 1) for key in counts.keys()}
    print(implied_probs)
    print(decimal_odds)
    
# scores = simulate(1314, model="log_reg", n=1000)
# counts = Counter(scores)
# plt.bar(counts.keys(), counts.values())
# plt.show()

simulate_contest("log_reg")