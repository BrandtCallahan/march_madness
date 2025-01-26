from datetime import datetime
import os

"""
Python Predictive Model imports
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import GridSearchCV
from math import sqrt
from tqdm import tqdm
from scipy.stats import norm
import random

from get_data import *

away_tm = "vanderbilt"
home_tm = "alabama"
neutral = False
season = 2025
today = pd.to_datetime("2025-01-20")


def matchup(away_tm, home_tm, season, today, neutral, n=501):

    # matchup data
    h_stats = get_team_stats(season, home_tm)[
        ["Tm", "Opp", "Date", "Poss", "Off Eff", "Def Eff", "Streak +/-"]
    ]
    a_stats = get_team_stats(season, away_tm)[
        ["Tm", "Opp", "Date", "Poss", "Off Eff", "Def Eff", "Streak +/-"]
    ]

    # weighted avg. efficiencies
    #   last 5 games (3, 3, 1.5, 1.5, 1)
    home_stats_dt = h_stats[h_stats["Date"].astype("datetime64[ns]") < today]
    home_stats = home_stats_dt.iloc[-5:].reset_index(drop=True)
    home_weight_off = (
        (home_stats.iloc[4]["Off Eff"] * 3)
        + (home_stats.iloc[3]["Off Eff"] * 3)
        + (home_stats.iloc[2]["Off Eff"] * 1.5)
        + (home_stats.iloc[1]["Off Eff"] * 1.5)
        + (home_stats.iloc[0]["Off Eff"] * 1)
    ) / 10
    home_weight_def = (
        (home_stats.iloc[4]["Def Eff"] * 3)
        + (home_stats.iloc[3]["Def Eff"] * 3)
        + (home_stats.iloc[2]["Def Eff"] * 1.5)
        + (home_stats.iloc[1]["Def Eff"] * 1.5)
        + (home_stats.iloc[0]["Def Eff"] * 1)
    ) / 10

    away_stats_dt = a_stats[a_stats["Date"].astype("datetime64[ns]") < today]
    away_stats = away_stats_dt.iloc[-5:]
    away_weight_off = (
        (away_stats.iloc[4]["Off Eff"] * 3)
        + (away_stats.iloc[3]["Off Eff"] * 3)
        + (away_stats.iloc[2]["Off Eff"] * 1.5)
        + (away_stats.iloc[1]["Off Eff"] * 1.5)
        + (away_stats.iloc[0]["Off Eff"] * 1)
    ) / 10
    away_weight_def = (
        (away_stats.iloc[4]["Def Eff"] * 3)
        + (away_stats.iloc[3]["Def Eff"] * 3)
        + (away_stats.iloc[2]["Def Eff"] * 1.5)
        + (away_stats.iloc[1]["Def Eff"] * 1.5)
        + (away_stats.iloc[0]["Def Eff"] * 1)
    ) / 10

    # Home vs. Away adjustments
    home_ha = home_away_adj(season, home_tm, today)
    if neutral:
        home_ha_off_adj = home_ha[home_ha["Location"] == "Neutral"].reset_index(
            drop=True
        )["Loc Off Eff"][0]
        home_ha_def_adj = home_ha[home_ha["Location"] == "Neutral"].reset_index(
            drop=True
        )["Loc Def Eff"][0]
    else:
        home_ha_off_adj = home_ha[home_ha["Location"] == "Home"].reset_index(drop=True)[
            "Loc Off Eff"
        ][0]
        home_ha_def_adj = home_ha[home_ha["Location"] == "Home"].reset_index(drop=True)[
            "Loc Def Eff"
        ][0]

    away_ha = home_away_adj(season, away_tm, today)
    if neutral:
        away_ha_off_adj = away_ha[away_ha["Location"] == "Neutral"].reset_index(
            drop=True
        )["Loc Off Eff"][0]
        away_ha_def_adj = away_ha[away_ha["Location"] == "Neutral"].reset_index(
            drop=True
        )["Loc Def Eff"][0]
    else:
        away_ha_off_adj = away_ha[away_ha["Location"] == "Away"].reset_index(drop=True)[
            "Loc Off Eff"
        ][0]
        away_ha_def_adj = away_ha[away_ha["Location"] == "Away"].reset_index(drop=True)[
            "Loc Def Eff"
        ][0]

    # winning/losing streak adjustments

    # Home Team Win/Loss Streak Adjustment
    home_w_streak = heat_check(season_year, home_tm, today)
    home_l_streak = cool_down(season_year, home_tm, today)

    home_streak = home_stats["Streak +/-"][-1:].reset_index(drop=True)[0]

    if home_streak >= 3:
        home_off_stk_adj = home_w_streak["Off Eff"][0]
        home_def_stk_adj = home_w_streak["Def Eff"][0]
    elif home_streak <= -3:
        home_off_stk_adj = home_l_streak["Off Eff"][0]
        home_def_stk_adj = home_l_streak["Def Eff"][0]
    else:
        home_off_stk_adj = 0
        home_def_stk_adj = 0

    # Away Team Win/Loss Streak Adjustment
    away_w_streak = heat_check(season_year, away_tm, today)
    away_l_streak = cool_down(season_year, away_tm, today)

    away_streak = away_stats["Streak +/-"][-1:].reset_index(drop=True)[0]

    if away_streak >= 3:
        away_off_stk_adj = away_w_streak["Off Eff"][0]
        away_def_stk_adj = away_w_streak["Def Eff"][0]
    elif away_streak <= -3:
        away_off_stk_adj = away_l_streak["Off Eff"][0]
        away_def_stk_adj = away_l_streak["Def Eff"][0]
    else:
        away_off_stk_adj = 0
        away_def_stk_adj = 0

    h_tm_off = (home_weight_off + home_off_stk_adj + home_ha_off_adj) - (
        away_weight_def + away_def_stk_adj + away_ha_def_adj
    )
    a_tm_off = (away_weight_off + away_off_stk_adj + away_ha_off_adj) - (
        home_weight_def + home_def_stk_adj + home_ha_def_adj
    )
    mean = h_tm_off - a_tm_off
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
    true_point_diff = results_df.true_point_diff.mean()

    return [game_winner, point_diff.astype(float), win_pct.astype(float)]


next_game = matchup(away_tm, home_tm, season, today, neutral, n=501)
