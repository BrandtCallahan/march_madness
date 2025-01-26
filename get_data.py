from datetime import datetime
import os

os.chdir(
    "~/Documents/Python/professional_portfolio/march_madness"
)

"""
Python Predictive Model imports
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logzero import logger
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

from webscrape_utils import (
    findTables,
    pullTable,
    # get_game_matchups,
    # get_game_lineups,
    # get_game_gambling,
)

"""
Team stats by game
"""
season_year = 2025
team = "vanderbilt"


def get_team_stats(season_year, team):
    url = f"https://www.sports-reference.com/cbb/schools/{team}/men/{season_year}-gamelogs.html"
    # findTables(url)

    # team statistics
    team_gamelog = pullTable(url, tableID="sgl-basic_NCAAM")
    team_gamelog = team_gamelog[
        ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
    ].reset_index(drop=True)

    # drop games with no data
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]"))
        < datetime.now().strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # label team
    team_gamelog["Tm"] = team

    # friendly home/away/neutral
    team_gamelog.loc[team_gamelog["Location"] == "@", "Home/Away"] = "Away"
    team_gamelog.loc[team_gamelog["Location"] == "N", "Home/Away"] = "Neutral"
    team_gamelog.loc[~(team_gamelog["Location"].isin(["@", "N"])), "Home/Away"] = "Home"

    # overtime flag
    team_gamelog.loc[team_gamelog["W/L"].str.contains("OT"), "OT Flag"] = True
    team_gamelog["OT Flag"] = team_gamelog["OT Flag"].fillna(False)

    # friendly win/loss
    team_gamelog.loc[team_gamelog["W/L"].str.contains("W"), "W/L Flag"] = True
    team_gamelog["W/L Flag"] = team_gamelog["W/L Flag"].fillna(False)

    team_gamelog["W/L"] = team_gamelog["W/L"].str[0]

    # running sum W
    team_gamelog["Total W"] = team_gamelog["W/L Flag"].cumsum().replace({True: 1})
    team_gamelog["Total W%"] = team_gamelog["Total W"] / team_gamelog["G"].astype(int)

    # streaks
    team_gamelog.loc[(team_gamelog["W/L"] == "L"), "Streak Value"] = -1
    team_gamelog["Streak Value"] = team_gamelog["Streak Value"].fillna(1)

    team_gamelog["Start Streak"] = team_gamelog["Streak Value"].ne(
        team_gamelog["Streak Value"].shift()
    )
    team_gamelog["Streak Id"] = team_gamelog["Start Streak"].cumsum()
    team_gamelog["Running Streak"] = team_gamelog.groupby("Streak Id").cumcount() + 1

    # W Streak == +, L Streak == -
    team_gamelog.loc[team_gamelog["W/L"] == "W", "Streak +/-"] = team_gamelog[
        "Running Streak"
    ]
    team_gamelog.loc[team_gamelog["W/L"] == "L", "Streak +/-"] = (
        team_gamelog["Running Streak"] * -1
    )
    team_gamelog["Streak +/-"] = team_gamelog["Streak +/-"].astype(int)

    # possessions
    team_gamelog["Poss"] = (
        team_gamelog["FGA"].astype(int)
        + (0.44 * team_gamelog["FTA"].astype(int))
        - team_gamelog["ORB"].astype(int)
        + team_gamelog["TOV"].astype(int)
    ) / 2
    team_gamelog["Opp Poss"] = (
        team_gamelog["Opp FGA"].astype(int)
        + (0.44 * team_gamelog["Opp FTA"].astype(int))
        - team_gamelog["Opp ORB"].astype(int)
        + team_gamelog["Opp TOV"].astype(int)
    ) / 2

    # offense ratings
    team_gamelog["Off Eff"] = (
        team_gamelog["Tm Score"].astype(int) / team_gamelog["Poss"]
    ) * 100
    team_gamelog["Shoot Eff"] = (
        team_gamelog["FG"].astype(int) + (0.5 * team_gamelog["3P"].astype(int))
    ) / team_gamelog["FGA"].astype(int)
    team_gamelog["AST TOV Eff"] = team_gamelog["AST"].astype(int) / team_gamelog[
        "TOV"
    ].astype(int)

    # defense ratings
    team_gamelog["Def Eff"] = (
        team_gamelog["Opp Score"].astype(int) / team_gamelog["Opp Poss"]
    ) * 100
    team_gamelog["Opp Shoot Eff"] = (
        team_gamelog["Opp FG"].astype(int) + (0.5 * team_gamelog["Opp 3P"].astype(int))
    ) / team_gamelog["Opp FGA"].astype(int)
    team_gamelog["Opp AST TOV Eff"] = team_gamelog["Opp AST"].astype(
        int
    ) / team_gamelog["Opp TOV"].astype(int)

    team_df = team_gamelog

    return team_df


def heat_check(season_year, team, today_date):

    # team stats
    team_df = get_team_stats(season_year, team)
    team_df = team_df[
        (team_df["Date"].astype("datetime64[ns]")) < today_date.strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # non streak efficiency
    nonstreak = team_df[(team_df["Streak +/-"] < 3) & (team_df["Streak +/-"] > -3)]
    nonstreak_tm = (
        nonstreak.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    # winning streak efficiency
    winstreak = team_df[team_df["Streak +/-"] >= 3]
    winstreak_tm = (
        winstreak.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    try:
        winstreak_off = winstreak_tm["Off Eff"][0]
    except KeyError:
        winstreak_off = nonstreak_tm["Off Eff"][0]

    try:
        winstreak_def = winstreak_tm["Def Eff"][0]
    except KeyError:
        winstreak_def = nonstreak_tm["Def Eff"][0]

    # compare
    winstreak_df = pd.DataFrame(
        data={
            "Tm": [team],
            "W Off Eff": [winstreak_off - nonstreak_tm["Off Eff"][0]],
            "W Def Eff": [winstreak_def - nonstreak_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    return winstreak_df


def cool_down(season_year, team, today_date):

    # team stats
    team_df = get_team_stats(season_year, team)
    team_df = team_df[
        (team_df["Date"].astype("datetime64[ns]")) < today_date.strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # non streak efficiency
    nonstreak = team_df[(team_df["Streak +/-"] < 3) & (team_df["Streak +/-"] > -3)]
    nonstreak_tm = (
        nonstreak.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    # losing streak efficiency
    lossstreak = team_df[team_df["Streak +/-"] <= -3]
    lossstreak_tm = (
        lossstreak.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    try:
        lossstreak_off = lossstreak_tm["Off Eff"][0]
    except KeyError:
        lossstreak_off = nonstreak_tm["Off Eff"][0]

    try:
        lossstreak_def = lossstreak_tm["Def Eff"][0]
    except KeyError:
        lossstreak_def = nonstreak_tm["Def Eff"][0]

    # compare
    lossstreak_df = pd.DataFrame(
        data={
            "Tm": [team],
            "L Off Eff": [lossstreak_off - nonstreak_tm["Off Eff"][0]],
            "L Def Eff": [lossstreak_def - nonstreak_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    return lossstreak_df


def home_away_adj(season_year, team, today_date):

    # team stats
    team_df = get_team_stats(season_year, team)
    team_df = team_df[
        (team_df["Date"].astype("datetime64[ns]")) < today_date.strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # avg. game efficiency (all games)
    allgame_tm = (
        team_df.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    # home game efficiency
    hm_game = team_df[team_df["Home/Away"] == "Home"]
    hm_tm = (
        hm_game.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    try:
        hm_off = hm_tm["Off Eff"][0]
    except KeyError:
        hm_off = allgame_tm["Off Eff"][0]

    try:
        hm_def = hm_tm["Def Eff"][0]
    except KeyError:
        hm_def = allgame_tm["Def Eff"][0]

    # compare
    hm_df = pd.DataFrame(
        data={
            "Tm": [team],
            "Location": ["Home"],
            "Loc Off Eff": [hm_off - allgame_tm["Off Eff"][0]],
            "Loc Def Eff": [hm_def - allgame_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    # away game efficiency
    aw_game = team_df[team_df["Home/Away"] == "Away"]
    aw_tm = (
        aw_game.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    try:
        aw_off = aw_tm["Off Eff"][0]
    except KeyError:
        aw_off = allgame_tm["Off Eff"][0]

    try:
        aw_def = aw_tm["Def Eff"][0]
    except KeyError:
        aw_def = allgame_tm["Def Eff"][0]

    # compare
    aw_df = pd.DataFrame(
        data={
            "Tm": [team],
            "Location": ["Away"],
            "Loc Off Eff": [aw_off - allgame_tm["Off Eff"][0]],
            "Loc Def Eff": [aw_def - allgame_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    # away game efficiency
    ne_game = team_df[team_df["Home/Away"] == "Neutral"]
    ne_tm = (
        ne_game.groupby(["Tm"])
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
        )
        .reset_index()
    ).rename(
        columns={
            "OffEff": "Off Eff",
            "DefEff": "Def Eff",
        }
    )

    try:
        ne_off = ne_tm["Off Eff"][0]
    except KeyError:
        ne_off = allgame_tm["Off Eff"][0]

    try:
        ne_def = ne_tm["Def Eff"][0]
    except KeyError:
        ne_def = allgame_tm["Def Eff"][0]

    # compare
    ne_df = pd.DataFrame(
        data={
            "Tm": [team],
            "Location": ["Neutral"],
            "Loc Off Eff": [ne_off - allgame_tm["Off Eff"][0]],
            "Loc Def Eff": [ne_def - allgame_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    location_df = pd.concat([hm_df, aw_df, ne_df]).reset_index(drop=True)

    return location_df


"""
Can we build our own ranking system?
"""
