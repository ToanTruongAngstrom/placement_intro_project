import json
import os
from data_collection import load_results, load_3ptfg_last_100, load_shot_distance_data
import scipy.stats

def get_3ptfg_last_100(playerid):
    if not os.path.exists("data/3ptfg_last_100.json"):
        with open("data/3ptfg_last_100.json", "w") as f:
            json.dump({}, f)

    with open("data/3ptfg_last_100.json", "r") as f:
        data = json.load(f)

    if str(playerid) in data.keys():
        return data[str(playerid)]

    return load_3ptfg_last_100(playerid)

def get_shot_distance_data(playerid):
    if not os.path.exists("data/shot_pc_by_dist.json"):
        with open("data/shot_pc_by_dist.json", "w") as f:
            json.dump({}, f)

    with open("data/shot_pc_by_dist.json", "r") as f:
        data = json.load(f)

    if str(playerid) in data.keys():
        return data[str(playerid)]

    return load_shot_distance_data(playerid)

'''
Use a beta prior distribution with averages corresponding to the player's average percentages.
Intuition for the priors: "If a player took 500 threes in the last 100 games, 
and 40% of them were from 25-29 ft, then let's use 500, scaled down, to construct a pseudo-count prior.
For the dew balls, only use those 40% of long threes."
'''
def get_priors(playerid):
    last_100_data = get_3ptfg_last_100(int(playerid)) # made, att, 3pt%
    shot_distance_data = get_shot_distance_data(int(playerid)) # fgm_20-24, fga_20-24, fg_pc_20-24, fgm_25-29, fga_25-29, fg_pc_25-29

    epsilon = 0.001 # To avoid 0s in extreme cases

    k = 0.5 # Since we want shots from historical 3pt contests to count for more than in-game shots, we need a scale factor k.

    # Avoid division by zero errors, give low percentages for low volume shooters 
    if (shot_distance_data["fga_20-24"] + shot_distance_data["fga_25-29"]) == 0:
        return (3, 7, 1, 4)
    
    pc_long_threes = shot_distance_data["fga_25-29"] / (shot_distance_data["fga_20-24"] + shot_distance_data["fga_25-29"])

    # alpha prior represents assumed number of makes in the data
    alpha_reg_prior = shot_distance_data["fg_pc_20-24"] * last_100_data["att"] * k + epsilon
    # beta prior represents assumed number of misses in the data
    beta_reg_prior = (1-shot_distance_data["fg_pc_20-24"]) * last_100_data["att"] * k + epsilon

    # Prior distribution for dew shots
    alpha_dew_prior = shot_distance_data["fg_pc_25-29"] * last_100_data["att"] * pc_long_threes * k + epsilon
    beta_dew_prior = (1-shot_distance_data["fg_pc_25-29"]) * last_100_data["att"] * pc_long_threes * k + epsilon

    return (alpha_reg_prior, beta_reg_prior, alpha_dew_prior, beta_dew_prior)

# Get previous contest data
def get_results(playerid):
    try:
        f = open("data/results.json")
    except FileNotFoundError:
        load_results()
        f = open("data/results.json")

    results = json.load(f)
    f.close()

    for player in results:
        if player["id"] == str(playerid):
            made = player["made"]
            att = player["att"]
            dewmade = player["dewmade"]
            dewatt = player["dewatt"]

            return int(made), int(att), int(dewmade), int(dewatt)
    
    return

# Perform Bayesian updating of the shot percentage distribution based on previous contest data
def update(playerid):
    (alpha_reg_prior, beta_reg_prior, alpha_dew_prior, beta_dew_prior) = get_priors(playerid)
    (made, att, dewmade, dewatt) = get_results(playerid)
    alpha_reg_post = alpha_reg_prior + made
    beta_reg_post = beta_reg_prior + (att - made)
    alpha_dew_post = alpha_dew_prior + dewmade
    beta_dew_post = beta_dew_prior + (dewatt - dewmade)

    return (alpha_reg_post, beta_reg_post, alpha_dew_post, beta_dew_post)

def get_probabilities(n, playerid):
    (alpha_reg, beta_reg, alpha_dew, beta_dew) = update(playerid)
    theta_reg_dist = scipy.stats.beta(alpha_reg, beta_reg)
    theta_dew_dist = scipy.stats.beta(alpha_dew, beta_dew)
    samples_reg = theta_reg_dist.rvs(n)
    samples_dew = theta_dew_dist.rvs(n)

    return samples_reg, samples_dew