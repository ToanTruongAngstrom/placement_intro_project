# NBA 3-Point Contest Modelling
## Data Collection

### Sources
- **Angstrom dataset**  
  https://d2c6afifpk.execute-api.eu-west-2.amazonaws.com/dev

- **NBA Stats (Shooting Percentages by Distance to Basket)**  
  https://www.nba.com/stats/players/shooting  

  - **Underlying API** (raw data):  
    https://stats.nba.com/stats/leaguedashplayershotlocations

- **Shot-by-Shot Data via `nba_api`**  
  - Docs: https://github.com/swar/nba_api/blob/master/docs/table_of_contents.md  
  - Endpoint used: `ShotChartDetail`

---

## Modelling

### Logistic Regression

- Collected data for over 1000 individual 3pt attempts per participant using `nba_api`.
- Spanning the last *n* seasons (configurable).
- Model trained using `sklearn`.
- **Input features**:
  - `loc_x`: x coordinate of the shot
  - `loc_y`: y coordinate of the shot
  - `shot_zone_area`: categorized shot zone
  - `shot_type`: type of shot (e.g., fadeaway, pull-up jump shot)
  - `season`: models the recency of the shot taken

- **Post-processing**:  
  Applied a constant multiplicative scaling to account for the ~25% higher shooting percentage in 3pt contests compared to in-game shots.

---

### Bayesian Modelling

- Shot percentage modelled via a **Beta distribution**.
- Separated into `ϑ_reg` and `ϑ_dew`, each with its own distribution.
- **Prior**: Estimated from in-game 3pt makes/misses over past 100 games.
  - These are weighted and scaled according to hyper-parameters w, k
  - Parameters are tuned via grid search
- **Bayesian Updating**: If available, used past 3pt contest data to perform a Beta-Binomial update.
- **Posterior** reflects both game performance and historical contest results.

---

## Model Comparison

### Logistic Regression

**Advantages**:
- Incorporates granular, shot-by-shot data (location and type).
- Flexible with potential to incorporate more features.

**Disadvantages**:
- Doesn't account for "open-ness" of shots — hurts accuracy for tightly-defended players.
- Requires assumptions for adjusting contest performance.
- Does not account for previous contest data.

> Example issue: Bias against players like Damian Lillard who are typically closely defended.

**Potential Improvements**:
- Main issue is a lack of more granular data.
- The ML model would be improved given defender proximity data which is not publically accessible.
- Given shot-by-shot data from past 3pt contests would have allowed the model to account for increased performance in 3pt contests.

---

### Bayesian Model

**Advantages**:
- Incorporates prior 3pt contest history (valuable signal).
- Models uncertainty in shooting performance via posterior.
- Easily tuned when more data is made available.

**Disadvantages**:
- Less granular — treats all shots in same rack category equally.
  - Does not account for location-based shot variation (e.g., corner vs top-of-key).
- Prior distribution is important to model those with no data for updating.

---

## Simulation

Inferences are drawn via Monte Carlo simulation using the model-generated probabilities.

### `simulate(playerid, model="bayesian, n=1000)`
- Simulates an individual round for one player (`playerid`).
- Runs `n` simulations and returns score distribution.
- Can be used to answer questions like:
  - "How likely is a player to make all money balls?"
  - (see extension)

### `simulate_contest(model="bayesian", n=1000)`
- Simulates the full 3pt contest (first + final rounds) across 8 participants.
- Runs `n` simulations and outputs implied win probabilities for each player.
- Tie breaks are dealt with iteratively to ensure there is exactly one winner.

---

## Testing and Evaluation

The implied probabilities from my model were compared against the odds offered by Bet365: https://news.bet365.com/en-us/article/nba-3-point-contest-outright-odds/2025021120214058187

| Player          | Bookies Implied % (includes margin) | Model Implied % |
| --------------- | ------------------------------------| ----------------|
| Damian Lillard  | 23.53                               | 22.5            |
| Norman Powell   | 14.29                               | 19.6            |
| Buddy Hield     | 17.39                               | 18.7            |
| Darius Garland  | 16.67                               | 12.9            |
| Cameron Johnson | 11.76                               | 10.5            |
| Jalen Brunson   | 11.11                               | 7.5             |
| Tyler Herro     | 12.50                               | 4.9             |
| Cade Cunningham | 8.33                                | 3.4             |

This gives a correlation coefficient of 0.85.

To check if this is overfitted, I ran the same model with the participants of the 2024 and 2023 3pt contests.
- 2024: Correlation coefficient = 0.45
- 2023: Correlation coefficient = 0.68
- This might suggest my model was over-tuned. 
  - However, it may just be because the model uses future data that was not available at the time of those contests, so it makes sense that the probabilities generated are different.