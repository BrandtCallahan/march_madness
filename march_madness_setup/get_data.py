from datetime import datetime
import os

os.chdir(
    "C:/Users/bcallahan/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness"
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
import time

from utils.webscrape_utils import (
    findTables,
    pullTable,
)

from march_madness_setup.team_dict import get_teamnm

"""
Team stats by game
"""
# season_year = 2024
# team = "vanderbilt"
# team_url = reference_dict[team]
# team_name = tmname_dict[team]

# team_df = get_teamnm()


def get_team_stats(season_year, team_url, team_name):
    team_df = get_teamnm()
    conference_dict = (
        team_df[(team_df["Season"] == season_year)]
        .set_index("Tm Name")["Conference Abbr"]
        .to_dict()
    )
    team_df = team_df[
        (team_df["Season"] == season_year) & (team_df["Ref Name"] == team_url)
    ]
    url = f"https://www.sports-reference.com/cbb/schools/{team_url}/men/{season_year}-gamelogs.html"
    # findTables(url)

    # team statistics
    team_gamelog = pullTable(url, tableID="sgl-basic_NCAAM")
    team_gamelog = team_gamelog[
        ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
        & ~(team_gamelog["W/L"].isin([""]))
        & ~(team_gamelog["FG"].isin([""]))
    ].reset_index(drop=True)

    # drop games with no data
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]"))
        < datetime.now().strftime("%Y-%m-%d")
    ].reset_index(drop=True)

    # label team
    team_gamelog["Tm"] = team_name

    return team_gamelog


def tm_stats(
    season_year,
    team_url,
    team_name,
    today=datetime.now().strftime("%Y-%m-%d"),
):
    team_df = get_teamnm()
    conference_dict = (
        team_df[(team_df["Season"] == season_year)]
        .set_index("Tm Name")["Conference Abbr"]
        .to_dict()
    )
    team_df = team_df[
        (team_df["Season"] == season_year) & (team_df["Ref Name"] == team_url)
    ]
    team_gamelog = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
    )
    team_gamelog = team_gamelog[team_gamelog["Tm"] == team_name]

    # add conference
    # tm_conf = conference_dict[f"{team_name}"]
    tm_conf = team_df["Conference Abbr"].reset_index(drop=True)[0]
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
    team_gamelog["Total W"] = (
        team_gamelog["W/L Flag"].cumsum().replace({True: 1, False: 0})
    )
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
    opp_w = pd.DataFrame()
    for opp_tm in team_gamelog["Opp"].to_list():
        # print(opp_tm)

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
        else:
            opp_tmnm = opp_tm

        # get opponent W's
        opp_gamelog = pd.read_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
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

    # team_df = team_gamelog

    return team_df


def save_team_stats(season_year, team_name_list, today):
    teamnm_df = get_teamnm()
    conference_dict = (
        teamnm_df[(teamnm_df["Season"] == season_year)]
        .set_index("Tm Name")["Conference Abbr"]
        .to_dict()
    )
    tmref_dict = (
        teamnm_df[(teamnm_df["Season"] == season_year)]
        .set_index("Tm Name")["Ref Name"]
        .to_dict()
    )

    for tm in team_name_list:
        logger.info(f"Adding {tm}")
        team_df = teamnm_df[
            (teamnm_df["Season"] == season_year) & (teamnm_df["Tm Name"] == tm)
        ].reset_index(drop=True)

        # pull season gamelog
        team_gamelog = get_team_stats(season_year, team_df["Ref Name"][0], tm)
        team_gamelog = team_gamelog[
            (team_gamelog["Date"].astype("datetime64[ns]")) < today.strftime("%Y-%m-%d")
        ]

        # pull out .csv
        try:
            season_boxscores = pd.read_csv(
                f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv",
            )
        except:
            season_boxscores = pd.DataFrame()

        add_tm = (
            pd.concat(
                [
                    season_boxscores.replace({"": np.nan}),
                    team_gamelog.replace({"": np.nan}).astype(season_boxscores.dtypes),
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        add_tm.to_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv",
            index=False,
        )

        # sleep 10 seconds after each data pull
        time.sleep(10)

    return print(f"{season_year} boxscores saved to .csv")


def heat_check(season_year, team, today_date, team_df):
    # team_df = get_teamnm()
    # conference_dict = (
    #     team_df[(team_df["Season"] == season_year)]
    #     .set_index("Tm Name")["Conference Abbr"]
    #     .to_dict()
    # )
    # tmref_dict = (
    #     team_df[(team_df["Season"] == season_year)]
    #     .set_index("Tm Name")["Ref Name"]
    #     .to_dict()
    # )
    # team_df = team_df[(team_df["Season"] == season_year) & (team_df["Tm Name"] == team)]

    # team stats
    # team_df = tm_stats(season_year, tmref_dict[team], team)
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
    w_off = winstreak.nlargest(int(len(winstreak) / 2), "Off Eff")["Off Eff"].mean()
    w_def = winstreak.nlargest(int(len(winstreak) / 2), "Def Eff")["Def Eff"].mean()
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


def cool_down(season_year, team, today_date, team_df):
    # team_df = get_teamnm()
    # conference_dict = (
    #     team_df[(team_df["Season"] == season_year)]
    #     .set_index("Tm Name")["Conference Abbr"]
    #     .to_dict()
    # )
    # tmref_dict = (
    #     team_df[(team_df["Season"] == season_year)]
    #     .set_index("Tm Name")["Ref Name"]
    #     .to_dict()
    # )
    # team_df = team_df[(team_df["Season"] == season_year) & (team_df["Tm Name"] == team)]

    # # team stats
    # team_df = tm_stats(season_year, tmref_dict[team], team)
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
    l_off = lossstreak.nlargest(int(len(lossstreak) / 2), "Off Eff")["Off Eff"].mean()
    l_def = lossstreak.nlargest(int(len(lossstreak) / 2), "Def Eff")["Def Eff"].mean()
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
        game = team_df[
            (team_df["Home/Away"] == f"{location}")
            & (team_df["Conf Game"] == conf_game)
        ]
    else:
        game = team_df[(team_df["Home/Away"] == f"{location}")]

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


def home_away_adj(season_year, team, team_df, today_date, conf_game):
    teamnm_df = get_teamnm()
    conference_dict = (
        teamnm_df[(teamnm_df["Season"] == season_year)]
        .set_index("Tm Name")["Conference Abbr"]
        .to_dict()
    )
    tmref_dict = (
        teamnm_df[(teamnm_df["Season"] == season_year)]
        .set_index("Tm Name")["Ref Name"]
        .to_dict()
    )
    ## team_df = team_df[(team_df["Season"] == season_year) & (team_df["Tm Name"] == team)]

    # # team stats
    # team_df = tm_stats(season_year, tmref_dict[team], team)
    # team_df = team_df[
    #     (team_df["Date"].astype("datetime64[ns]")) < today_date.strftime("%Y-%m-%d")
    # ].reset_index(drop=True)

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
    hm_df = eff_compare(team, team_df, allgame_tm, "Home", conf_game)

    # away game efficiency
    aw_df = eff_compare(team, team_df, allgame_tm, "Away", conf_game)

    # neutral game efficiency
    ne_df = eff_compare(team, team_df, allgame_tm, "Neutral", conf_game)

    location_df = pd.concat([hm_df, aw_df, ne_df]).reset_index(drop=True)

    return location_df


def tm_rating(season_year, today):

    # read out gamelog
    team_gamelog = pd.read_csv(
        f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
    )
    # only games less than "today"
    team_gamelog = team_gamelog[
        (team_gamelog["Date"].astype("datetime64[ns]") < today)
    ].reset_index(drop=True)

    # friendly win/loss
    team_gamelog.loc[team_gamelog["W/L"].str.contains("W"), "W/L Flag"] = 1
    team_gamelog["W/L Flag"] = team_gamelog["W/L Flag"].fillna(0)

    team_gamelog["W/L"] = team_gamelog["W/L"].str[0]

    # running sum W
    team_gamelog["Total W"] = team_gamelog.groupby(["Tm"])["W/L Flag"].cumsum()
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

    # tm effieciency rating
    team_gamelog["Tm Eff"] = team_gamelog["Off Eff"] - team_gamelog["Def Eff"]

    # margin of victory
    team_gamelog["Margin Victory"] = team_gamelog["Tm Score"].astype(
        int
    ) - team_gamelog["Opp Score"].astype(int)

    # game luck
    team_gamelog["Tm Luck"] = (
        (team_gamelog["Tm Score"] / team_gamelog["Opp Score"]) ** 10.5
    ) / (((team_gamelog["Tm Score"] / team_gamelog["Opp Score"]) ** 10.5) + 1)

    # tm W %
    team_w = team_gamelog.copy()
    team_w.loc[(team_w["W/L"].str.contains("W")), "W"] = 1
    team_w.loc[team_w["W/L"].str.contains("L"), "L"] = 1
    team_w = (
        team_w.groupby(["Tm"], observed=True)
        .agg(
            W=("W", "sum"),
            L=("L", "sum"),
            OppCt=("Opp", "count"),
            OffEff=("Off Eff", "median"),
            DefEff=("Def Eff", "median"),
        )
        .reset_index()
    )
    team_w["W Pct"] = team_w["W"] / team_w["OppCt"]

    team = get_teamnm()
    team = team[team["Season"] == season_year]

    team_w = team_w.merge(team, how="left", left_on="Tm", right_on="Tm Name")

    # join opp w pct back to gamelog
    team_df = team_gamelog.merge(
        team_w[["Gamelog Name", "W Pct", "OffEff", "DefEff"]].rename(
            columns={
                "W Pct": "Opp W Pct",
                "OffEff": "Opp Off Eff",
                "DefEff": "Opp Def Eff",
            }
        ),
        how="left",
        left_on="Opp",
        right_on="Gamelog Name",
    )

    # drop games against non Basketball Ref teams
    team_df = team_df[team_df["Gamelog Name"].notnull()]

    team_df["Tm Off Eff Adj"] = (
        team_df["Off Eff"] * team_df["Off Eff"].mean()
    ) / team_df["Opp Def Eff"]

    team_df["Tm Def Eff Adj"] = (
        team_df["Def Eff"] * team_df["Def Eff"].mean()
    ) / team_df["Opp Off Eff"]
    team_df["Tm Eff Adj"] = team_df["Tm Off Eff Adj"] - team_df["Tm Def Eff Adj"]

    # group by Tm (average)
    tm_df = (
        team_df.groupby(["Tm"], observed=True)
        .agg(
            OffEff=("Off Eff", "median"),
            OffEffAdj=("Tm Off Eff Adj", "median"),
            DefEff=("Def Eff", "median"),
            DefEffAdj=("Tm Def Eff Adj", "median"),
            OppOffEff=("Opp Off Eff", "median"),
            OppDefEff=("Opp Def Eff", "median"),
            TmEff=("Tm Eff", "median"),
            TmEffAdj=("Tm Eff Adj", "median"),
            TmLuck=("Tm Luck", "sum"),
            OppWpct=("Opp W Pct", "median"),
            TmW=("Total W", "last"),
            Wpct=("Total W%", "last"),
            Poss=("Poss Ext", "median"),
        )
        .reset_index()
    )
    tm_df["LuckFactor"] = tm_df["TmW"] - tm_df["TmLuck"]

    # Tm Eff (compared to Lg Avg)
    tm_df["Tm Net Rating"] = (tm_df["OffEffAdj"] - tm_df["OffEffAdj"].mean()) - (
        (tm_df["DefEffAdj"] - tm_df["DefEffAdj"].mean())
    )
    tm_df["Opp Net Rating"] = (tm_df["OppOffEff"] - tm_df["OppOffEff"].mean()) - (
        (tm_df["OppDefEff"] - tm_df["OppDefEff"].mean())
    )
    tm_df["Tm KP Rating"] = tm_df["Tm Net Rating"] + tm_df["Opp Net Rating"]

    # adjust for "Lucky" wins
    tm_df["Tm KP Rating"] = tm_df["Tm KP Rating"] - tm_df["LuckFactor"]

    # rank order (Off Eff, Def Eff, Opp W %)
    tm_df["Luck Rnk"] = tm_df["LuckFactor"].rank(method="max", ascending=True)
    tm_df["Off Eff Rnk"] = tm_df["OffEff"].rank(method="max", ascending=True)
    tm_df["Def Eff Rnk"] = tm_df["DefEff"].rank(method="max", ascending=False)
    tm_df["W Rnk"] = tm_df["Wpct"].rank(method="max", ascending=True)
    tm_df["Opp W Rnk"] = tm_df["OppWpct"].rank(method="max", ascending=True)

    # divide by max to get 'Ranking Points'
    tm_df["Off Eff Pts"] = tm_df["Off Eff Rnk"] / tm_df["Off Eff Rnk"].max()
    tm_df["Def Eff Pts"] = tm_df["Def Eff Rnk"] / tm_df["Def Eff Rnk"].max()
    tm_df["W Pts"] = (tm_df["W Rnk"] + tm_df["Opp W Rnk"]) / (
        tm_df["W Rnk"].max() + tm_df["Opp W Rnk"].max()
    )

    # Ratings
    tm_df["Tm Rating"] = (
        (tm_df["Off Eff Pts"] * 1.35)
        + (tm_df["Def Eff Pts"] * 1.45)
        + (tm_df["W Pts"] * 1.2)
    ) / 4
    tm_df["Net Tm Rating"] = tm_df["Tm KP Rating"].apply(
        lambda x: (x - tm_df["Tm KP Rating"].min())
        / (tm_df["Tm KP Rating"].max() - tm_df["Tm KP Rating"].min())
        * 100
    )
    tm_df["Tm Rank"] = tm_df["Net Tm Rating"].rank(method="max", ascending=False)

    tm_df = tm_df[
        [
            "Tm",
            "Tm Rank",
            "OffEffAdj",
            "DefEffAdj",
            "TmEffAdj",
            "Poss",
            "TmW",
            "Wpct",
            "Tm KP Rating",
            "Net Tm Rating",
        ]
    ]

    return tm_df


"""
Can we build our own ranking system?
"""
