from collections import Counter, defaultdict
import glob
import heapq
import os
import pandas as pd
import scipy.stats
import numpy as np
import json
import matplotlib.pyplot as plt
from bayesian import get_probabilities as get_bayesian_probs
from logistic_regression import get_probabilities_by_location
from data_collection import load_participant_info, get_name_by_id

# By default, the last rack is the money rack and dew balls are shot after the second and third racks.
def simulate(playerid, model="bayesian", n=1, w=0.2, k=1.3, commentary=False, money_balls=[4, 9, 15, 21, 22, 23, 24, 25, 26], dew_balls=[10,16]):
    if model == "bayesian":
        scores = simulate_bayesian(playerid, n, w, k, commentary, money_balls, dew_balls)
    elif model == "log_reg":
        scores = simulate_log_reg(playerid, n, commentary, money_balls, dew_balls)
    else:
        print("Choose an existing model")
        return []
    return scores

def simulate_bayesian(playerid, n, w, k, commentary, money_balls, dew_balls):
    # n separate realisations of theta_reg and theta_dew according to beta posterior distribution
    thetas_reg, thetas_dew = get_bayesian_probs(n, w, k, playerid)
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

def get_participants():
    try:
        f = open("data/participants.json")
    except FileNotFoundError:
        load_participant_info()
        f = open("data/participants.json")
    
    participants = json.load(f)
    f.close()
    return participants

def simulate_contest(model="bayesian", n=1000):
    participants = get_participants()
    winners = []

    # Delete any old stored logistic regression models/data
    if model == "log_reg":
        shot_charts = glob.glob("data/shot_charts/*.json")
        models = glob.glob("data/models/*.joblib")
        for file in shot_charts + models:
            os.remove(file)

    for i in range(n):
        finalists = sim_qualifiers(participants, model)
        winner = sim_finals(finalists, model)
        winners.append(winner["name"])

    # Calculate the implied probabilities of each player winning
    counts = Counter(winners)
    implied_probs = {key: 100 * counts[key] / n for key in counts.keys()}
    decimal_odds = {key: round(100 / implied_probs[key], 1) for key in counts.keys()}
    return implied_probs, decimal_odds

def sim_qualifiers(participants, model="bayesian"):
    scores = dict()
    for participant in participants:
        name = participant["firstname"] + " " + participant["surname"]
        id = participant["playerid"]
        scores[id] = {"name": name, "score": simulate(id, model=model)[0]}
    
    finalists = top_3_with_tiebreak(scores, model)
    # finalists = heapq.nlargest(3, scores.items(), key=lambda i: i[1]["score"])
    return finalists

    
def top_3_with_tiebreak(scores, model="bayesian"):
    score_groups = defaultdict(list)
    for id, data in scores.items():
        score_groups[data["score"]].append((id, data))
    
    sorted_scores = sorted(score_groups.keys(), reverse=True)
    finalists = []
    for score in sorted_scores:
        remaining_slots = 3 - len(finalists)
        if remaining_slots == 0:
            break

        group = score_groups[score]

        if len(group) <= remaining_slots:
            finalists.extend(group)
        else:
            # Tie-break needed
            selected = sim_tiebreak(group, remaining_slots, model=model)
            finalists.extend(selected)
    return finalists
    
def sim_tiebreak(participants, num_needed, model="bayesian"):
    assert num_needed <= len(participants)
    
    current_group = participants
    result = []
    while True:
        # Simulate new scores
        scores = {
            id: {"name": data["name"], "score": simulate(id, model=model)[0]}
            for id, data in current_group
        }
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        ranked_participants = [(id, {"name": scores[id]["name"], "score": scores[id]["score"]}) for id, _ in ranked]

        # Check if there's a clean cutoff
        top = ranked_participants[:num_needed]
        if len(ranked_participants) <= num_needed:
            return ranked_participants
        # Check for tie at cutoff point
        cutoff_score = scores[top[-1][0]]["score"]
        tied_group = [p for p in ranked_participants if scores[p[0]]["score"] == cutoff_score]
        qualified = [p for p in ranked_participants if scores[p[0]]["score"] > cutoff_score]
        if len(tied_group) == num_needed - len(qualified):
            result.extend(ranked_participants[:num_needed])
            return result
        
        # Tie still exists, simulate again for tied group
        num_needed = num_needed - len(qualified)
        result.extend(qualified)
        current_group = tied_group

def sim_finals(finalists, model="bayesian"):
    # First simulation
    final_scores = {
        id: {"name": data["name"], "score": simulate(id, model=model)[0]}
        for id, data in finalists
    }
    # Find the top score
    max_score = max(p["score"] for p in final_scores.values())
    top_ids = [id for id, p in final_scores.items() if p["score"] == max_score]

    if len(top_ids) == 1:
        return final_scores[top_ids[0]]

    # Tie detected: build tie group
    tie_group = [(id, {"name": final_scores[id]["name"]}) for id in top_ids]

    # Break the tie until one winner remains
    winner = sim_tiebreak(tie_group, num_needed=1, model=model)[0]
    return winner[1]

# Perform a grid search for the scaling factors in the Bayesian model.
def grid_search():
    # k represents the how much we value in-game 3pt data compared to contest data.
    # c represents how much we scale in-game data to reflect increased percentages in the contests.
    w_values = [x / 10.0 for x in range(2, 5)]
    k_values = [x / 20.0 for x in range(20, 30)]
    with open("data/past_scores.json") as f:
        past_scores = json.load(f)

    min_error = float('inf')
    best_params = None
    
    for w in w_values:
        for k in k_values:
            squared_err = 0
            total_count = 0
            for player_id, real_scores in past_scores.items():
                sim_scores = simulate(int(player_id), w=w, k=k, n=1000)
                sim_mean = sum(sim_scores) / len(sim_scores)
                for score in real_scores:
                    squared_err += (score - sim_mean) ** 2
                    total_count += 1

            mse = squared_err / total_count

            if mse < min_error:
                min_error = mse
                best_params = (w, k)
            
            print(f"w = {w}, k = {k}, mse = {mse}")

    return best_params

# print(grid_search())

# scores = simulate(1050, n=1000)
# counts = Counter(scores)
# plt.bar(counts.keys(), counts.values())
# plt.show()

# implied_probs, decimal_odds = simulate_contest()
# print(f"Implied probabilities: {implied_probs}")
# print(f"Decimal odds: {decimal_odds}")

bookies_odds = {'Damian Lillard': 4.7, 'Tyrese Haliburton': 5.4, 'Trae Young': 6.5, 'Malik Beasley': 7.5, 'Karl-Anthony Towns': 8, 'Lauri Markkanen': 9, 'Jalen Brunson': 8, 'Donovan Mitchell': 10.5}
bookies_probabilities = {key: 100/bookies_odds[key] for key in bookies_odds.keys()}

implied_probs, decimal_odds = simulate_contest()

df = pd.DataFrame({
    'Bookies Implied %': bookies_probabilities,
    'Model Implied %': {name: implied_probs[name] for name in bookies_odds}
})

df = df.round(2).sort_values('Model Implied %', ascending=False)

print(df)
print(df.corr())
print(decimal_odds)