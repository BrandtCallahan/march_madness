from datetime import datetime
import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
from scipy.stats import norm
import random
import math
from scipy.special import erf
import pylab as p

from utils.get_data import *
from utils.team_dict import *

pd.set_option("future.no_silent_downcasting", True)


def matchup(away_tm, home_tm, tm_df, n=5):
    team_df = get_teamnm()
    home_url = team_df[team_df["Tm Name"] == home_tm]["Ref Name"].reset_index(
        drop=True
    )[0]
    away_url = team_df[team_df["Tm Name"] == away_tm]["Ref Name"].reset_index(
        drop=True
    )[0]

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


"""
The single game win probability formula and visualization code were originally taken from @bjade and then modified to fit my data.
https://github.com/bjade/ncaabbtools/blob/master/examples/kenpom_gamepredict.ipynb
"""
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
    return p.show()
