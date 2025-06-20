from datetime import datetime
import os

os.chdir(
    "C:/Users/bcallahan/Documents/Python/professional_portfolio/march_madness"
)

"""
Python Predictive Model imports
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logzero import logger
from math import sqrt
from tqdm import tqdm
import time

from utils.webscrape_utils import (
    findTables,
    pullTable,
    read_gamelog,
)

from utils.team_dict import get_teamnm

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
    ## team_gamelog = pullTable(url, tableID="sgl-basic_NCAAM")
    team_gamelog = read_gamelog(url)
    team_gamelog = team_gamelog[
        ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
        & ~(team_gamelog["W/L"].isin(["", np.nan, pd.NA]))
        & ~(team_gamelog["FG"].isin(["", np.nan, pd.NA]))
        & ~(team_gamelog["G"].isin(["", np.nan, pd.NA]))
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
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
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
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
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

    for n, tm in enumerate(team_name_list):
        logger.info(f"Adding {n+1}/{len(team_name_list)}: {tm}")
        team_df = teamnm_df[
            (teamnm_df["Season"] == season_year) & (teamnm_df["Tm Name"] == tm)
        ].reset_index(drop=True)

        # pull season gamelog
        try:
            team_gamelog = get_team_stats(season_year, team_df["Ref Name"][0], tm)
            team_gamelog = team_gamelog[
                (team_gamelog["Date"].astype("datetime64[ns]"))
                < today.strftime("%Y-%m-%d")
            ]
            team_gamelog[" "] = np.nan
            team_gamelog["Opp"] = team_gamelog["Opp"].replace(
                {
                    "Louisiana State": "LSU",
                    "North Carolina": "UNC",
                    "Maryland Eastern Shore": "Maryland-Eastern Shore",
                }
            )

            team_gamelog = team_gamelog[
                [
                    "G",
                    "Date",
                    "Location",
                    "Opp",
                    "Game Type",
                    "W/L",
                    "Tm Score",
                    "Opp Score",
                    "FG",
                    "FGA",
                    "FG%",
                    "3P",
                    "3PA",
                    "3P%",
                    "2P",
                    "2PA",
                    "2P%",
                    "eFG%",
                    "FT",
                    "FTA",
                    "FT%",
                    "ORB",
                    "DRB",
                    "TRB",
                    "AST",
                    "STL",
                    "BLK",
                    "TOV",
                    "PF",
                    "Opp FG",
                    "Opp FGA",
                    "Opp FG%",
                    "Opp 3P",
                    "Opp 3PA",
                    "Opp 3P%",
                    "Opp 2P",
                    "Opp 2PA",
                    "Opp 2P%",
                    "Opp eFG%",
                    "Opp FT",
                    "Opp FTA",
                    "Opp FT%",
                    "Opp ORB",
                    "Opp DRB",
                    "Opp TRB",
                    "Opp AST",
                    "Opp STL",
                    "Opp BLK",
                    "Opp TOV",
                    "Opp PF",
                    "Tm",
                ]
            ].astype(
                {
                    "Tm Score": int,
                    "Opp Score": int,
                    "FG": int,
                    "FGA": int,
                    "FG%": float,
                    "3P": int,
                    "3PA": int,
                    "3P%": float,
                    "2P": int,
                    "2PA": int,
                    "2P%": float,
                    "eFG%": float,
                    "FT": int,
                    "FTA": int,
                    "FT%": float,
                    "ORB": int,
                    "DRB": int,
                    "TRB": int,
                    "AST": int,
                    "STL": int,
                    "BLK": int,
                    "TOV": int,
                    "PF": int,
                    "Opp FG": int,
                    "Opp FGA": int,
                    "Opp FG%": float,
                    "Opp 3P": int,
                    "Opp 3PA": int,
                    "Opp 3P%": float,
                    "Opp 2P": int,
                    "Opp 2PA": int,
                    "Opp 2P%": float,
                    "Opp eFG%": float,
                    "Opp FT": int,
                    "Opp FTA": int,
                    "Opp FT%": float,
                    "Opp ORB": int,
                    "Opp DRB": int,
                    "Opp TRB": int,
                    "Opp AST": int,
                    "Opp STL": int,
                    "Opp BLK": int,
                    "Opp TOV": int,
                    "Opp PF": int,
                }
            )
        except:
            team_gamelog = pd.DataFrame(
                columns=[
                    "G",
                    "Date",
                    "Location",
                    "Opp",
                    "Game Type",
                    "W/L",
                    "Tm Score",
                    "Opp Score",
                    "FG",
                    "FGA",
                    "FG%",
                    "3P",
                    "3PA",
                    "3P%",
                    "2P",
                    "2PA",
                    "2P%",
                    "eFG%",
                    "FT",
                    "FTA",
                    "FT%",
                    "ORB",
                    "DRB",
                    "TRB",
                    "AST",
                    "STL",
                    "BLK",
                    "TOV",
                    "PF",
                    "Opp FG",
                    "Opp FGA",
                    "Opp FG%",
                    "Opp 3P",
                    "Opp 3PA",
                    "Opp 3P%",
                    "Opp 2P",
                    "Opp 2PA",
                    "Opp 2P%",
                    "Opp eFG%",
                    "Opp FT",
                    "Opp FTA",
                    "Opp FT%",
                    "Opp ORB",
                    "Opp DRB",
                    "Opp TRB",
                    "Opp AST",
                    "Opp STL",
                    "Opp BLK",
                    "Opp TOV",
                    "Opp PF",
                    "Tm",
                ]
            )

        # pull out .csv
        try:
            season_boxscores = pd.read_csv(
                f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv",
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
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv",
            index=False,
        )

        # sleep 10 seconds after each data pull
        time.sleep(10)

    return print(f"{season_year} boxscores saved to .csv")


def game_results(season, save=False):

    season_gm_results = pd.DataFrame(
        columns=[
            "Game Date",
            "Location",
            "Neutral Game",
            "Conference Game",
            "Postseason Game",
            "NCAA Tourney Game",
            "Matchup",
            "Home Team",
            "Home Rk",
            "Home Score",
            "Away Team",
            "Away Rk",
            "Away Score",
            "Home W",
            "Home Pt Diff",
        ]
    )

    # all teams
    team_df = get_teamnm()
    team_df = team_df[team_df["Season"] == season].reset_index(drop=True)

    # run through each team to get the results and compile to df
    for n, tm_url in enumerate(team_df["Ref Name"].unique().tolist()):

        logger.info(f'Running {n+1}/{len(team_df)}: {team_df["Tm Name"][n]}')

        # url = f"https://www.sports-reference.com/cbb/schools/{tm_url}/men/{season}-gamelogs.html"
        # tm_gmlog = read_gamelog(url)

        # temporarily read from saved 2025 boxscore .csv
        gmlog = pd.read_csv(
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_tm_boxscores.csv",
        )
        tm_gmlog = gmlog[gmlog["Tm"] == team_df["Tm Name"][n]].rename(
            columns={"Tm": "Tm Name"}
        )

        tm_gmlog = tm_gmlog[
            (
                ~(tm_gmlog["Opp"].isin(["", "Opponent", "Opp", pd.NA, np.nan]))
                & ~(tm_gmlog["W/L"].isin(["", np.nan, pd.NA]))
                & ~(tm_gmlog["FG"].isin(["", np.nan, pd.NA]))
                & ~(tm_gmlog["G"].isin(["", np.nan, pd.NA]))
            )
        ].reset_index(drop=True)

        tm_gmlog = tm_gmlog.astype(
            {
                "Tm Score": int,
                "Opp Score": int,
            }
        )

        for game in tm_gmlog.index:

            try:
                tmp_game = tm_gmlog.iloc[game]

                # conference game flag
                if "(Conf)" in tmp_game["Game Type"]:
                    conference = 1
                else:
                    conference = 0

                # postseason game flag
                if "REG" in tmp_game["Game Type"]:
                    postseason = 0
                else:
                    postseason = 1

                # ncaa tourney game flag
                if (
                    (not "REG" in tmp_game["Game Type"])
                    & (not "CTOURN" in tmp_game["Game Type"])
                    & (not "NIT" in tmp_game["Game Type"])
                ):
                    ncaa = 1
                else:
                    ncaa = 0

                # check if game is neutral
                if tmp_game["Location"] == "N":
                    # order by alphabetical order for both teams
                    tm_list = [tmp_game["Tm Name"], tmp_game["Opp"]]
                    tm_list.sort()

                    ratings = tm_rating(
                        season,
                        tmp_game["Date"],
                    )
                    try:
                        tm_ratings = ratings[ratings["Tm"].isin(tm_list)].reset_index(
                            drop=True
                        )
                        tm1 = tm_ratings["Tm"][0]
                        tm2 = tm_ratings["Tm"][1]
                    except KeyError:
                        tm2 = "Unknown"

                    if tm_ratings.empty:
                        hm_rnk = pd.NA
                        aw_rnk = pd.NA

                    if tm2 != "Unknown" and not tm_ratings.empty:

                        if ncaa == 1:
                            seeding_df = pd.read_csv(
                                f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season_bracketology.csv",
                            )
                            season_seed = seeding_df[seeding_df["season"] == season]

                            # tm1 seed
                            tm1_seed = season_seed[
                                season_seed["tm"] == tmp_game["Tm Name"]
                            ].reset_index(drop=True)["seed"][0]

                            # tm2 seed
                            tm2_seed = season_seed[
                                season_seed["tm"] == tmp_game["Opp"]
                            ].reset_index(drop=True)["seed"][0]

                            if tm1_seed <= tm2_seed:
                                hm_tm = tmp_game["Tm Name"]
                                hm_rnk = tm_ratings[
                                    tm_ratings["Tm"] == hm_tm
                                ].reset_index(drop=True)["Tm Rank"][0]
                                aw_tm = tmp_game["Opp"]
                                aw_rnk = tm_ratings[
                                    tm_ratings["Tm"] == aw_tm
                                ].reset_index(drop=True)["Tm Rank"][0]
                            else:
                                hm_tm = tmp_game["Opp"]
                                hm_rnk = tm_ratings[
                                    tm_ratings["Tm"] == hm_tm
                                ].reset_index(drop=True)["Tm Rank"][0]
                                aw_tm = tmp_game["Tm Name"]
                                aw_rnk = tm_ratings[
                                    tm_ratings["Tm"] == aw_tm
                                ].reset_index(drop=True)["Tm Rank"][0]
                        else:

                            if tm_ratings["Tm Rank"][0] < tm_ratings["Tm Rank"][1]:
                                hm_tm = tm_ratings["Tm"][0]
                                aw_tm = tm_ratings["Tm"][1]
                                hm_rnk = tm_ratings["Tm Rank"][0]
                                aw_rnk = tm_ratings["Tm Rank"][1]
                            else:
                                hm_tm = tm_ratings["Tm"][1]
                                aw_tm = tm_ratings["Tm"][0]
                                hm_rnk = tm_ratings["Tm Rank"][1]
                                aw_rnk = tm_ratings["Tm Rank"][0]

                    elif not tm_ratings.empty:
                        hm_tm = tm_ratings["Tm"][0]
                        hm_rnk = tm_ratings["Tm Rank"][0]
                        aw_tm = tmp_game["Opp"]
                        aw_rnk = ratings["Tm Rank"].max()
                    else:
                        hm_tm = tm_ratings["Tm"][0]
                        aw_tm = tmp_game["Opp"]

                    if hm_tm == tmp_game["Tm Name"]:
                        hm_score = tmp_game["Tm Score"]
                        aw_score = tmp_game["Opp Score"]
                    else:
                        hm_score = tmp_game["Opp Score"]
                        aw_score = tmp_game["Tm Score"]

                    hm_pt_diff = hm_score - aw_score
                    if hm_pt_diff > 0:
                        hm_w = 1
                    else:
                        hm_w = 0
                    matchup = f"{aw_tm} vs. {hm_tm}"

                    tmp_df = pd.DataFrame(
                        data={
                            "Game Date": [tmp_game["Date"]],
                            "Location": ["Neutral"],
                            "Neutral Game": [1],
                            "Conference Game": [conference],
                            "Postseason Game": [postseason],
                            "NCAA Tourney Game": [ncaa],
                            "Matchup": [matchup],
                            "Home Team": [hm_tm],
                            "Home Rk": [hm_rnk],
                            "Home Score": [hm_score],
                            "Away Team": [aw_tm],
                            "Away Rk": [aw_rnk],
                            "Away Score": [aw_score],
                            "Home W": [hm_w],
                            "Home Pt Diff": [hm_pt_diff],
                        }
                    )

                elif tmp_game["Location"] == "@":
                    tm_list = [tmp_game["Tm Name"], tmp_game["Opp"]]

                    ratings = tm_rating(
                        season,
                        tmp_game["Date"],
                    )
                    try:
                        tm_ratings = ratings[ratings["Tm"].isin(tm_list)].reset_index(
                            drop=True
                        )
                        aw_rnk = tm_ratings[
                            tm_ratings["Tm"] == tmp_game["Tm Name"]
                        ].reset_index(drop=True)["Tm Rank"][0]
                        hm_rnk = tm_ratings[
                            tm_ratings["Tm"] == tmp_game["Opp"]
                        ].reset_index(drop=True)["Tm Rank"][0]
                    except KeyError:
                        hm_rnk = ratings["Tm Rank"].max()

                    if tm_ratings.empty:
                        hm_rnk = pd.NA
                        aw_rnk = pd.NA

                    # win check
                    if tmp_game["Tm Score"] > tmp_game["Opp Score"]:
                        hm_tm_w = 0
                    else:
                        hm_tm_w = 1

                    hm_pt_diff = tmp_game["Opp Score"] - tmp_game["Tm Score"]
                    matchup = f"{team_df["Tm Name"][n]} vs. {team_df[team_df['Tm Name'] == tmp_game["Opp"]].reset_index(drop=True)['Tm Name'][0]}"

                    tmp_df = pd.DataFrame(
                        data={
                            "Game Date": [tmp_game["Date"]],
                            "Location": [f"@ {tmp_game["Opp"]}"],
                            "Neutral Game": [0],
                            "Conference Game": [conference],
                            "Postseason Game": [postseason],
                            "NCAA Tourney Game": [ncaa],
                            "Matchup": [matchup],
                            "Home Team": [
                                team_df[
                                    team_df["Tm Name"] == tmp_game["Opp"]
                                ].reset_index(drop=True)["Tm Name"][0]
                            ],
                            "Home Rk": [hm_rnk],
                            "Home Score": [tmp_game["Opp Score"]],
                            "Away Team": [team_df["Tm Name"][n]],
                            "Away Rk": [aw_rnk],
                            "Away Score": [tmp_game["Tm Score"]],
                            "Home W": [hm_tm_w],
                            "Home Pt Diff": [hm_pt_diff],
                        }
                    )

                else:
                    tm_list = [tmp_game["Tm Name"], tmp_game["Opp"]]

                    ratings = tm_rating(
                        season,
                        tmp_game["Date"],
                    )
                    try:
                        tm_ratings = ratings[ratings["Tm"].isin(tm_list)].reset_index(
                            drop=True
                        )
                        hm_rnk = tm_ratings[
                            tm_ratings["Tm"] == tmp_game["Tm Name"]
                        ].reset_index(drop=True)["Tm Rank"][0]
                        aw_rnk = tm_ratings[
                            tm_ratings["Tm"] == tmp_game["Opp"]
                        ].reset_index(drop=True)["Tm Rank"][0]
                    except KeyError:
                        aw_rnk = ratings["Tm Rank"].max()

                    if tm_ratings.empty:
                        hm_rnk = pd.NA
                        aw_rnk = pd.NA

                    # win check
                    if tmp_game["Tm Score"] > tmp_game["Opp Score"]:
                        hm_tm_w = 1
                    else:
                        hm_tm_w = 0

                    hm_pt_diff = tmp_game["Tm Score"] - tmp_game["Opp Score"]
                    matchup = f"{team_df[team_df['Tm Name'] == tmp_game["Opp"]].reset_index(drop=True)['Tm Name'][0]} vs. {team_df['Tm Name'][n]}"

                    tmp_df = pd.DataFrame(
                        data={
                            "Game Date": [tmp_game["Date"]],
                            "Location": [f"@ {team_df["Tm Name"][n]}"],
                            "Neutral Game": [0],
                            "Conference Game": [conference],
                            "Postseason Game": [postseason],
                            "NCAA Tourney Game": [ncaa],
                            "Matchup": [matchup],
                            "Home Team": [team_df["Tm Name"][n]],
                            "Home Rk": [hm_rnk],
                            "Home Score": [tmp_game["Tm Score"]],
                            "Away Team": [
                                team_df[
                                    team_df["Tm Name"] == tmp_game["Opp"]
                                ].reset_index(drop=True)["Tm Name"][0]
                            ],
                            "Away Rk": [aw_rnk],
                            "Away Score": [tmp_game["Opp Score"]],
                            "Home W": [hm_tm_w],
                            "Home Pt Diff": [hm_pt_diff],
                        }
                    )

                season_gm_results = pd.concat([season_gm_results, tmp_df]).reset_index(
                    drop=True
                )
            except KeyError:
                pass
            except ValueError:
                pass

        # sleep 10 seconds between data pulls
        # time.sleep(10)

    season_gm_results = season_gm_results.drop_duplicates()

    if save:
        season_gm_results.to_csv(
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_results.csv",
            index=False,
        )

    return season_gm_results


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
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season_year}_tm_boxscores.csv"
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
