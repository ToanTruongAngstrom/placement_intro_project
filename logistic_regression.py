import os
import json
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_collection import load_shot_chart

def get_shot_chart(playerid):
    path = f"data/shot_charts/{playerid}.json"
    if not os.path.exists(path):
        return pd.DataFrame(load_shot_chart(playerid))
    
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)

def train(playerid, print_eval=False):
    df = get_shot_chart(playerid)
    x = df[['LOC_X', 'LOC_Y', 'SHOT_ZONE_AREA', 'SHOT_DISTANCE', 'SHOT_TYPE']]
    y = df['SHOT_MADE_FLAG']

    # One-Hot encoding for categorical columns
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['SHOT_ZONE_AREA', 'SHOT_TYPE'])], remainder='passthrough')

    pipeline = Pipeline([('preprocess', transformer), ('clf', LogisticRegression(max_iter=5000))])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    pipeline.fit(x_train, y_train)
    # print(pipeline.predict_proba(x)[:, 1])
    if print_eval:
        evaluate(pipeline, x_test, y_test)

    joblib.dump(pipeline, f"data/models/{playerid}.joblib")
    return pipeline

def evaluate(pipeline, x_test, y_test):
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

'''
Upon inspecting the confusion matrix, we notice that there are lots of false negatives.
This is expected, since the probabilities for most shots will come out around 0.4.
Using the default threshold of 0.5 for classification, this will naturally result in 
the model predicting lots of misses.

However, we are going to use this model not for classification, but for a Monte-Carlo simulation,
so this is fine, as we are using the probabilities directly rather than for classification under a threshold.
'''

def get_probabilities_by_location(playerid):
    path = f"data/models/{playerid}.joblib"
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = train(playerid)

    # Get prediction of probability in each spot
    three_point_locations = [
        {"LOC_X": -220, "LOC_Y": 0,   "SHOT_ZONE_AREA": "Left Side(L)",          "SHOT_DISTANCE": 22, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": -150, "LOC_Y": 220, "SHOT_ZONE_AREA": "Left Side Center(LC)",  "SHOT_DISTANCE": 24, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 0,    "LOC_Y": 260, "SHOT_ZONE_AREA": "Center(C)",             "SHOT_DISTANCE": 24, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 150,  "LOC_Y": 220, "SHOT_ZONE_AREA": "Right Side Center(RC)", "SHOT_DISTANCE": 24, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 220,  "LOC_Y": 0,   "SHOT_ZONE_AREA": "Right Side(R)",         "SHOT_DISTANCE": 22, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": -92,  "LOC_Y": 290, "SHOT_ZONE_AREA": "Left Side Center(LC)",  "SHOT_DISTANCE": 30, "SHOT_TYPE": "3PT Field Goal"},
        {"LOC_X": 92,   "LOC_Y": 290, "SHOT_ZONE_AREA": "Right Side Center(RC)", "SHOT_DISTANCE": 30, "SHOT_TYPE": "3PT Field Goal"}
    ]

    df = pd.DataFrame(three_point_locations)
    thetas = model.predict_proba(df)[:, 1]
    # Constant multiplicative scaling by the league average difference shooting percentage between in game and 3pt contest 3s.
    # Can we now weight this based on the percentage of in-game 3s are wide open?
    k = 5/4
    return [theta * k for theta in thetas]

# print(get_probabilities_by_location(1050))