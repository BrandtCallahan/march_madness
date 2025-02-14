from datetime import datetime
import os

os.chdir(
    f"C:/Users/bcallahan/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness"
)

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
import math
from scipy.special import erf
import pylab as p

from march_madness_setup.get_data import *
from march_madness_setup.team_dict import *

pd.set_option("future.no_silent_downcasting", True)

# away_tm = "tennessee"
# home_tm = "vanderbilt"
# neutral = False
# conf_game = True
# season = 2025
# today = pd.to_datetime("2025-01-18")


def matchup(away_tm, home_tm, tm_df, n=5):
    team_df = get_teamnm()
    home_url = team_df[team_df["Tm Name"] == home_tm]["Ref Name"].reset_index(
        drop=True
    )[0]
    away_url = team_df[team_df["Tm Name"] == away_tm]["Ref Name"].reset_index(
        drop=True
    )[0]

    """
    # matchup data
    h_stats = tm_stats(season, home_url, home_tm)[
        [
            "Tm",
            "Opp",
            "Home/Away",
            "Conf Game",
            "Date",
            "Poss",
            "Poss Ext",
            "Opp Poss",
            "Opp Poss Ext",
            "Off Eff",
            "Def Eff",
            "Shoot Eff",
            "AST TOV Eff",
            "Opp Shoot Eff",
            "Opp AST TOV Eff",
            "W/L Flag",
            "Margin Victory",
            "Streak +/-",
            "Opp W Pct",
        ]
    ]
    a_stats = tm_stats(season, away_url, away_tm)[
        [
            "Tm",
            "Opp",
            "Home/Away",
            "Conf Game",
            "Date",
            "Poss",
            "Poss Ext",
            "Opp Poss",
            "Opp Poss Ext",
            "Off Eff",
            "Def Eff",
            "Shoot Eff",
            "AST TOV Eff",
            "Opp Shoot Eff",
            "Opp AST TOV Eff",
            "W/L Flag",
            "Margin Victory",
            "Streak +/-",
            "Opp W Pct",
        ]
    ]

    # +/- possessions per game
    h_poss = h_stats[
        (h_stats["Date"].astype("datetime64[ns]") < today)
        & (h_stats["Home/Away"] == "Home")
    ]
    h_poss = (h_poss["Poss Ext"] - h_poss["Opp Poss Ext"]).mean()
    h_w_poss = h_stats[
        (h_stats["Date"].astype("datetime64[ns]") < today) & (h_stats["W/L Flag"])
    ]
    h_w_poss = (h_w_poss["Poss Ext"] - h_w_poss["Opp Poss Ext"]).mean()
    a_poss = a_stats[
        (a_stats["Date"].astype("datetime64[ns]") < today)
        & (a_stats["Home/Away"] == "Away")
    ]
    a_poss = (a_poss["Poss Ext"] - a_poss["Opp Poss Ext"]).mean()
    a_w_poss = a_stats[
        (a_stats["Date"].astype("datetime64[ns]") < today) & (a_stats["W/L Flag"])
    ]
    a_w_poss = (a_w_poss["Poss Ext"] - a_w_poss["Opp Poss Ext"]).mean()

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
    home_weight_off = home_stats_dt["Off Eff"].mean()
    home_weight_def = home_stats_dt["Def Eff"].mean()

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
    away_weight_off = away_stats_dt["Off Eff"].mean()
    away_weight_def = away_stats_dt["Def Eff"].mean()

    # Home vs. Away adjustments
    home_ha = home_away_adj(season, home_tm, home_stats_dt, today, conf_game)
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

    away_ha = home_away_adj(season, away_tm, away_stats_dt, today, conf_game)
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
    home_w_streak = heat_check(season, home_tm, today, home_stats_dt)
    home_l_streak = cool_down(season, home_tm, today, home_stats_dt)

    home_streak = home_stats_dt["Streak +/-"][-1:].reset_index(drop=True)[0]

    if home_streak >= 3:
        home_off_stk_adj = home_w_streak["W Off Eff"][0]
        home_def_stk_adj = home_w_streak["W Def Eff"][0]
    elif home_streak <= -3:
        home_off_stk_adj = home_l_streak["L Off Eff"][0]
        home_def_stk_adj = home_l_streak["L Def Eff"][0]
    else:
        home_off_stk_adj = 0
        home_def_stk_adj = 0

    # Away Team Win/Loss Streak Adjustment
    away_w_streak = heat_check(season, away_tm, today, away_stats_dt)
    away_l_streak = cool_down(season, away_tm, today, away_stats_dt)

    away_streak = away_stats_dt["Streak +/-"][-1:].reset_index(drop=True)[0]

    if away_streak >= 3:
        away_off_stk_adj = away_w_streak["W Off Eff"][0]
        away_def_stk_adj = away_w_streak["W Def Eff"][0]
    elif away_streak <= -3:
        away_off_stk_adj = away_l_streak["L Off Eff"][0]
        away_def_stk_adj = away_l_streak["L Def Eff"][0]
    else:
        away_off_stk_adj = 0
        away_def_stk_adj = 0

    # additional offensive ratings (Shot Eff and A/TO Eff)
    home_off_adj = (
        home_stats_dt["Shoot Eff"] * home_stats_dt["AST TOV Eff"]
    ).mean() - (
        home_stats_dt["Opp Shoot Eff"] * home_stats_dt["Opp AST TOV Eff"]
    ).mean()
    away_off_adj = (
        away_stats_dt["Shoot Eff"] * away_stats_dt["AST TOV Eff"]
    ).mean() - (
        away_stats_dt["Opp Shoot Eff"] * away_stats_dt["Opp AST TOV Eff"]
    ).mean()

    # team's opponent W%
    home_opp_w = home_stats_dt["Opp W Pct"].mean()
    away_opp_w = away_stats_dt["Opp W Pct"].mean()

    off_tot = ((home_weight_off + home_off_adj)) - ((away_weight_off + away_off_adj))
    def_tot = ((home_weight_def)) - ((away_weight_def))
    """
    hm_rating = tm_df[tm_df["Tm"] == home_tm].reset_index(drop=True)["Net Tm Rating"][0]
    aw_rating = tm_df[tm_df["Tm"] == away_tm].reset_index(drop=True)["Net Tm Rating"][0]

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

    return [game_winner, true_point_diff.astype(float), win_pct.astype(float)]


def single_matchup(away_tm, home_tm, tm_df, neutral_gm):
    team_df = get_teamnm()
    home_url = team_df[team_df["Tm Name"] == home_tm]["Ref Name"].reset_index(
        drop=True
    )[0]
    away_url = team_df[team_df["Tm Name"] == away_tm]["Ref Name"].reset_index(
        drop=True
    )[0]

    h_tm = tm_df[tm_df["Tm"] == home_tm].reset_index(drop=True)
    a_tm = tm_df[tm_df["Tm"] == away_tm].reset_index(drop=True)

    # Game Point Differential
    if neutral_gm:
        point_diff = (
            (a_tm["TmEffAdj"][0] - h_tm["TmEffAdj"][0])
            * ((h_tm["Poss"] + a_tm["Poss"]) / 200)
        )[0]
    else:
        # Home Court Advantage (Adjustment)
        point_diff = (
            (
                a_tm["TmEffAdj"][0]
                - (h_tm["TmEffAdj"][0] )
            )
            * (h_tm["Poss"] + a_tm["Poss"])
            / 200
        )[0] - home_adv

    sigma, stdev = 10, 10
    x = 0
    home_adv = 3.5
    u, kpEMdiff = point_diff, point_diff

    # W Probability
    win_prob = norm.cdf(0, kpEMdiff, stdev)

    game_df = pd.DataFrame(
        data={
            "Tm": [away_tm, home_tm],
            "Win Prob.": [1 - win_prob, win_prob],
            "Point Diff": [kpEMdiff, kpEMdiff * -1],
        }
    ).reset_index(drop=True)

    return game_df


def graph_win_prob(away_tm, home_tm, point_diff, neutral, home_adv, stdev):

    if neutral:
        home_adv = 0
    else:
        home_adv = home_adv

    # Graph W Probability
    x = np.arange(
        (point_diff - home_adv) - 3.5 * stdev,
        (point_diff - home_adv) + 3.5 * stdev,
        0.01,
    )
    y = norm.pdf(x, (point_diff - home_adv), stdev)
    p.plot(x, y, color="k", lw=2)
    p.fill_between(
        x, 0, y, where=x >= 0, facecolor="blue", alpha=0.2, label=f"{away_tm} Wins"
    )
    p.fill_between(
        x, 0, y, where=x <= 0, facecolor="red", alpha=0.2, label=f"{home_tm} Wins"
    )
    p.axvline((point_diff - home_adv), c="k", ls="--")
    p.annotate(
        "{0:5.2f}".format((point_diff - home_adv)),
        xy=(1.05 * (point_diff - home_adv), 1.04 * y.max()),
    )
    p.legend(loc=2)
    p.show()


# next_game = matchup(away_tm, home_tm, season, today, neutral, conf_game, n=101)
