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
from team_dict import *

"""
Team stats by game
"""
# season_year = 2025
# team = "vanderbilt"
# team_url = reference_dict[team]
# team_name = tmname_dict[team]


def get_team_stats(season_year, team_url, team_name):
    url = f"https://www.sports-reference.com/cbb/schools/{team_url}/men/{season_year}-gamelogs.html"
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
    team_gamelog["Tm"] = team_name

    # add conference
    tm_conf = conference_dict[f"{team_name}"]
    team_gamelog["Conference"] = team_gamelog.Opp.map(conference_dict)
    team_gamelog["Conf Game"] = team_gamelog["Conference"] == tm_conf

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
        + team_gamelog["Opp TOV"].astype(int)
    ) / 2
    team_gamelog["Opp Poss"] = (
        team_gamelog["Opp FGA"].astype(int)
        + (0.44 * team_gamelog["Opp FTA"].astype(int))
        - team_gamelog["Opp ORB"].astype(int)
        + team_gamelog["TOV"].astype(int)
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
    team_df = get_team_stats(season_year, team, tmname_dict[team])
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
    w_off = winstreak.nlargest(int(len(winstreak)/2), "Off Eff")["Off Eff"].mean()
    w_def = winstreak.nlargest(int(len(winstreak)/2), "Def Eff")["Def Eff"].mean()
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
        # winstreak_off = winstreak_tm["Off Eff"][0]
        winstreak_off = w_off
    except KeyError:
        winstreak_off = nonstreak_tm["Off Eff"][0]

    try:
        # winstreak_def = winstreak_tm["Def Eff"][0]
        winstreak_def = w_def
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
    team_df = get_team_stats(season_year, team, tmname_dict[team])
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
    l_off = lossstreak.nlargest(int(len(lossstreak)/2), "Off Eff")["Off Eff"].mean()
    l_def = lossstreak.nlargest(int(len(lossstreak)/2), "Def Eff")["Def Eff"].mean()
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
        # lossstreak_off = lossstreak_tm["Off Eff"][0]
        lossstreak_off = l_off
    except KeyError:
        lossstreak_off = nonstreak_tm["Off Eff"][0]

    try:
        # lossstreak_def = lossstreak_tm["Def Eff"][0]
        lossstreak_def = l_def
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


def eff_compare(team, team_df, allgame_tm, location, conf_game):
    if conf_game:
        game = team_df[(team_df["Home/Away"] == f"{location}") & (team_df['Conf Game'] == conf_game)]
    else:
        game = team_df[
            (team_df["Home/Away"] == f"{location}")
    ]

    tm = (
        game.groupby(["Tm"])
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
        off_eff = tm["Off Eff"][0]
    except KeyError:
        off_eff = allgame_tm["Off Eff"][0]

    try:
        def_eff = tm["Def Eff"][0]
    except KeyError:
        def_eff = allgame_tm["Def Eff"][0]

    # compare
    eff_df = pd.DataFrame(
        data={
            "Tm": [team],
            "Location": [f"{location}"],
            "Loc Off Eff": [off_eff - allgame_tm["Off Eff"][0]],
            "Loc Def Eff": [def_eff - allgame_tm["Def Eff"][0]],
        }
    ).reset_index(drop=True)

    return eff_df


def home_away_adj(season_year, team, today_date, conf_game):

    # team stats
    team_df = get_team_stats(season_year, team, tmname_dict[team])
    team_df = team_df[
        (team_df["Date"].astype("datetime64[ns]")) < today_date.strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # avg. game efficiency (all games)
    if conf_game:
        allgame_tm = (
            team_df[team_df["Conf Game"]]
            .groupby(["Tm"])
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
    else:
        allgame_tm = (
            team_df
            .groupby(["Tm"])
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
    hm_df = eff_compare(tmname_dict[team], team_df, allgame_tm, "Home", conf_game)

    # away game efficiency
    aw_df = eff_compare(tmname_dict[team], team_df, allgame_tm, "Away", conf_game)

    # neutral game efficiency
    ne_df = eff_compare(tmname_dict[team], team_df, allgame_tm, "Neutral", conf_game)


    location_df = pd.concat([hm_df, aw_df, ne_df]).reset_index(drop=True)

    return location_df


def tm_rating(season_year, today):

    # read out gamelog
    team_gamelog = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/season{season_year}_tm_boxscores.csv"
    )
    # only games less than "today"
    team_gamelog = team_gamelog[(team_gamelog["Date"].astype("datetime64[ns]") < today)]

    # friendly win/loss
    team_gamelog.loc[team_gamelog["W/L"].str.contains("W"), "W/L Flag"] = True
    team_gamelog["W/L Flag"] = team_gamelog["W/L Flag"].fillna(False)

    team_gamelog["W/L"] = team_gamelog["W/L"].str[0]

    # running sum W
    team_gamelog["Total W"] = (
        team_gamelog["W/L Flag"].cumsum().replace({True: 1, False: 0})
    )
    team_gamelog["Total W%"] = team_gamelog["Total W"] / team_gamelog["G"].astype(int)

    # possessions
    team_gamelog["Poss"] = (
        team_gamelog["FGA"].astype(int)
        + (0.44 * team_gamelog["FTA"].astype(int))
        - team_gamelog["ORB"].astype(int)
        + team_gamelog["TOV"].astype(int)
    )
    team_gamelog["Poss Ext"] = (
        team_gamelog["FGA"].astype(int)
        + (0.4 * team_gamelog["FTA"].astype(int))
        - (
            1.07
            * (
                team_gamelog["ORB"].astype(int)
                / (
                    team_gamelog["ORB"].astype(int)
                    + (
                        team_gamelog["Opp TRB"].astype(int)
                        - team_gamelog["Opp ORB"].astype(int)
                    )
                )
            )
        )
        * (team_gamelog["FGA"].astype(int) - team_gamelog["FG"].astype(int))
        + team_gamelog["TOV"].astype(int)
    )
    team_gamelog["Opp Poss"] = (
        team_gamelog["Opp FGA"].astype(int)
        + (0.44 * team_gamelog["Opp FTA"].astype(int))
        - team_gamelog["Opp ORB"].astype(int)
        + team_gamelog["Opp TOV"].astype(int)
    )
    team_gamelog["Opp Poss Ext"] = (
        team_gamelog["Opp FGA"].astype(int)
        + (0.4 * (team_gamelog["Opp FTA"].astype(int)))
        - (
            1.07
            * (team_gamelog["Opp ORB"].astype(int))
            / (
                team_gamelog["Opp ORB"].astype(int)
                + (team_gamelog["TRB"].astype(int) - team_gamelog["ORB"].astype(int))
            )
        )
        * (team_gamelog["Opp FGA"].astype(int) - team_gamelog["Opp FG"].astype(int))
        + team_gamelog["Opp TOV"].astype(int)
    )

    # offense ratings
    team_gamelog["Off Eff"] = (
        team_gamelog["Tm Score"].astype(int) / team_gamelog["Poss Ext"]
    ) * 100
    team_gamelog["Shoot Eff"] = (
        team_gamelog["FG"].astype(int) + (0.5 * team_gamelog["3P"].astype(int))
    ) / team_gamelog["FGA"].astype(int)
    team_gamelog["AST TOV Eff"] = team_gamelog["AST"].astype(int) / team_gamelog[
        "TOV"
    ].astype(int)

    # defense ratings
    team_gamelog["Def Eff"] = (
        team_gamelog["Opp Score"].astype(int) / team_gamelog["Opp Poss Ext"]
    ) * 100
    team_gamelog["Opp Shoot Eff"] = (
        team_gamelog["Opp FG"].astype(int) + (0.5 * team_gamelog["Opp 3P"].astype(int))
    ) / team_gamelog["Opp FGA"].astype(int)
    team_gamelog["Opp AST TOV Eff"] = team_gamelog["Opp AST"].astype(
        int
    ) / team_gamelog["Opp TOV"].astype(int)

    # margin of victory
    team_gamelog["Margin Victory"] = team_gamelog["Tm Score"].astype(
        int
    ) - team_gamelog["Opp Score"].astype(int)

    # opponent win %
    #   Can we also rank by opponent rank??
    opp_w = pd.DataFrame()
    for n, opp_tm in enumerate(team_gamelog["Opp"].to_list()):
        # print(opp_tm)
        logger.info(f"Game {n+1} of {len(team_gamelog["Opp"].to_list())}")

        # manually adjust tm names
        if opp_tm == "USC Upstate":
            opp_tmnm = "South Carolina Upstate"
        elif opp_tm == "UConn":
            opp_tmnm = "Connecticut"
        elif opp_tm == "Arkansas-Pine Bluff":
            opp_tmnm = "Arkansas-Pine-Bluff"
        elif opp_tm == "USC":
            opp_tmnm = "Southern California"
        elif opp_tm == "Pitt":
            opp_tmnm == "Pittsburgh"
        elif opp_tm == "Southern Miss":
            opp_tmnm = "Southern Mississippi"
        elif opp_tm == "UT-Martin":
            opp_tmnm = "Tennessee-Martin"
        elif opp_tm == "VCU":
            opp_tmnm = "Virginia Commonwealth"
        elif opp_tm == "Penn":
            opp_tmnm = "Pennsylvania"
        elif opp_tm == "St. Joseph's":
            opp_tmnm = "Saint Joseph's"
        elif opp_tm == "SIU-Edwardsville":
            opp_tmnm = "Southern Illinois-Edwardsville"
        elif opp_tm == "VMI":
            opp_tmnm = "Virginia Military Institute"
        elif opp_tm == "SMU":
            opp_tmnm = "Southern Methodist"
        elif opp_tm == "UC-San Diego":
            opp_tmnm = "UC San Diego"
        elif opp_tm == "UC-Davis":
            opp_tmnm = "UC Davis"
        elif opp_tm == "UC-Irvine":
            opp_tmnm = "UC Irvine"
        elif opp_tm == "UC-Riverside":
            opp_tmnm = "UC Riverside"
        elif opp_tm == "UNLV":
            opp_tmnm = "Nevada-Las Vegas"
        elif opp_tm == "UIC":
            opp_tmnm = "Illinois-Chicago"
        elif opp_tm == "UMass":
            opp_tmnm = "Massachusetts"
        elif opp_tm == "St. Peter's":
            opp_tmnm = "Saint Peter's"
        elif opp_tm == "UCSB":
            opp_tmnm = "UC Santa Barbara"
        elif opp_tm == "LIU":
            opp_tmnm = "Long Island University"
        elif opp_tm == "Central Connecticut":
            opp_tmnm = "Central Connecticut State"
        elif opp_tm == "UMBC":
            opp_tmnm = "Maryland-Baltimore County"
        elif opp_tm == "UMass-Lowell":
            opp_tmnm = "Massachusetts-Lowell"
        elif opp_tm == "IU Indianapolis":
            opp_tmnm = "IU Indy"
        elif opp_tm == "ETSU":
            opp_tmnm = "East Tennessee State"
        else:
            opp_tmnm = opp_tm

        # get opponent W's
        opp_gamelog = pd.read_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/season{season_year}_tm_boxscores.csv"
        )
        opp_gamelog = opp_gamelog[
            (opp_gamelog["Tm"] == opp_tmnm)
            & (
                (opp_gamelog["Date"].astype("datetime64[ns]"))
                < pd.to_datetime(today).strftime("%Y-%m-%d")
            )
        ]

        opp_wins = opp_gamelog[(opp_gamelog["W/L"].str.contains("W"))]["W/L"].count()
        opp_losses = opp_gamelog[(opp_gamelog["W/L"].str.contains("L"))]["W/L"].count()
        opp_w_pct = opp_wins / (opp_wins + opp_losses)

        opp_w = pd.concat(
            [opp_w, pd.DataFrame(data={"Opp": [opp_tm], "Opp W Pct": [opp_w_pct]})]
        ).reset_index(drop=True)

    # join opp w pct back to gamelog
    # team_df = team_gamelog.merge(opp_w, how="inner", on="Opp", validate="1:1")
    # can't merge because you can play a team multiple times (merge on more than just tm name is likely needed)
    team_df = pd.concat(
        [team_gamelog.reset_index(drop=True), opp_w[["Opp W Pct"]]], axis=1
    )

    # group by Tm (average)
    tm_df = (
        team_df.groupby(["Tm"], observed=True)
        .agg(
            OffEff=("Off Eff", "mean"),
            DefEff=("Def Eff", "mean"),
            OppWpct=("Opp W Pct", "mean"),
        )
        .reset_index()
    )

    # rank order (Off Eff, Def Eff, Opp W %)
    tm_df["Off Eff Rnk"] = tm_df["OffEff"].rank(method="max", ascending=True)
    tm_df["Def Eff Rnk"] = tm_df["DefEff"].rank(method="max", ascending=False)
    tm_df["Opp W Rnk"] = tm_df["OppWpct"].rank(method="max", ascending=True)

    # divide by max to get 'Ranking Points'
    tm_df['Off Eff Pts'] = tm_df['Off Eff Rnk'] / tm_df['Off Eff Rnk'].max()
    tm_df["Def Eff Pts"] = tm_df["Def Eff Rnk"] / tm_df["Def Eff Rnk"].max()
    tm_df["Opp W Pts"] = tm_df["Opp W Rnk"] / tm_df["Opp W Rnk"].max()

    # Ratings
    tm_df['Tm Rating'] = ((tm_df['Off Eff Pts']*1.3) + (tm_df["Def Eff Pts"]*1.45) + (tm_df['Opp W Pts']*1.25)) / 4
    tm_df["Tm Rating"] = (tm_df['Tm Rating'].apply(lambda x: (x-tm_df['Tm Rating'].min()) / (tm_df["Tm Rating"].max() - tm_df["Tm Rating"].min()) * 100))
    tm_df["Tm Rank"] = tm_df["Tm Rating"].rank(method="max", ascending=False)

    return tm_df
