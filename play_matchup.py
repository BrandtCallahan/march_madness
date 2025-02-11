from datetime import datetime
import os
import pandas as pd
import numpy as np
from logzero import logger
from math import sqrt
from tqdm import tqdm
from scipy.stats import norm
import random

from get_data import *
from team_dict import *

pd.set_option("future.no_silent_downcasting", True)

def matchup(away_tm, home_tm, season, today, tm_df, neutral, conf_game, n=1):

    team_df = get_teamnm()
    home_url = team_df[team_df["Tm Name"] == home_tm]["Ref Name"].reset_index(
        drop=True
    )[0]
    away_url = team_df[team_df["Tm Name"] == away_tm]["Ref Name"].reset_index(
        drop=True
    )[0]
    hm_rating = tm_df[tm_df['Tm'] == home_tm].reset_index(drop=True)['Tm Rating'][0]
    aw_rating = tm_df[tm_df["Tm"] == away_tm].reset_index(drop=True)["Tm Rating"][0]

    mean = (hm_rating) - (aw_rating)
    std = 10

    results_df = pd.DataFrame()
    for i in range(n):
        game = norm.ppf(random.random(), loc=mean, scale=std)

        # return winner
        if game > 0:
            winner = home_tm
        else:
            winner = away_tm

        # build dataframe with results
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    data={
                        "game": [i + 1],
                        "winner": [winner],
                        "point_diff": [abs(game)],
                        "true_point_diff": [game],
                    }
                ),
            ]
        ).reset_index(drop=True)

    # find winner from monte carlo
    home_tm_w = results_df["winner"].str.count(f"{home_tm}").sum()
    away_tm_w = results_df["winner"].str.count(f"{away_tm}").sum()

    if home_tm_w > away_tm_w:
        game_winner = home_tm
        win_pct = home_tm_w / n
    else:
        game_winner = away_tm
        win_pct = away_tm_w / n

    # find average score diff
    point_diff = results_df[results_df["winner"] == game_winner].point_diff.mean()
    true_point_diff = results_df.point_diff.mean()

    return [game_winner, pt_diff.astype(float), win_pct.astype(float)]


# next_game = matchup(away_tm, home_tm, season, today, neutral, conf_game, n=10001)
