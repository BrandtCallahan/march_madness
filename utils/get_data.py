from datetime import datetime
import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
import time

from utils.webscrape_utils import (
    findTables,
    pullTable,
)

from utils.team_dict import get_teamnm


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
                "W/L",
                "Tm Score",
                "Opp Score",
                "FG",
                "FGA",
                "FG%",
                "3P",
                "3PA",
                "3P%",
                "FT",
                "FTA",
                "FT%",
                "ORB",
                "TRB",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "PF",
                " ",
                "Opp FG",
                "Opp FGA",
                "Opp FG%",
                "Opp 3P",
                "Opp 3PA",
                "Opp 3P%",
                "Opp FT",
                "Opp FTA",
                "Opp FT%",
                "Opp ORB",
                "Opp TRB",
                "Opp AST",
                "Opp STL",
                "Opp BLK",
                "Opp TOV",
                "Opp PF",
                "Tm",
            ]
        ]

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
