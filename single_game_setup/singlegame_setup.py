from matplotlib.pylab import norm
import pandas as pd
import numpy as np
import pylab as p
import random
import math
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from march_madness_setup.get_data import *
from march_madness_setup.team_dict import *
from utils.beautiful_soup_helper import *

pd.set_option("future.no_silent_downcasting", True)


def pregame(season, away_tm, home_tm, today):

    team_df = get_teamnm()
    team_df = team_df[team_df["Season"] == season]

    hm_df = team_df[team_df["Tm Name"] == home_tm].reset_index(drop=True)
    aw_df = team_df[team_df["Tm Name"] == away_tm].reset_index(drop=True)

    # # pull in team data
    # team_gamelog = pd.read_csv(
    #     f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_tm_boxscores.csv"
    # )
    # pull straight from college basketball reference
    aw_url = f"https://www.sports-reference.com/cbb/schools/{aw_df['Ref Name'][0]}/men/{season}-gamelogs.html"
    away_df = read_gamelog(aw_url)
    away_df['Tm'] = away_tm

    hm_url = f"https://www.sports-reference.com/cbb/schools/{hm_df['Ref Name'][0]}/men/{season}-gamelogs.html"
    home_df = read_gamelog(hm_url)
    home_df['Tm'] = home_tm

    team_gamelog = pd.concat([away_df, home_df]).reset_index(drop=True)
    team_gamelog =  team_gamelog[(~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
        & ~(team_gamelog["W/L"].isin(["", np.nan, pd.NA]))
        & ~(team_gamelog["FG"].isin(["", np.nan, pd.NA]))
        & ~(team_gamelog["G"].isin(["", np.nan, pd.NA])))].reset_index(drop=True)

    # set dtypes 
    team_gamelog = team_gamelog.astype(
        {
            "G": int,
            "Tm Score": int,
            "Opp Score": int,
            "FG": int,
            "FGA": int,
            "FG%": float,
            '3P': int,
            '3PA': int,
            '3P%': float,
            '2P': int,
            '2PA': int,
            '2P%': float,
            'eFG%': float,
            'FT': int,
            'FTA': int,
            'FT%': float,
            'ORB': int,
            'DRB': int,
            'TRB': int,
            'AST': int,
            'STL': int,
            'BLK': int,
            'TOV': int,
            'PF': int,
            "Opp FG": int,
            "Opp FGA": int,
            "Opp FG%": float,
            'Opp 3P': int,
            'Opp 3PA': int,
            'Opp 3P%': float,
            'Opp 2P': int,
            'Opp 2PA': int,
            'Opp 2P%': float,
            'Opp eFG%': float,
            'Opp FT': int,
            'Opp FTA': int,
            'Opp FT%': float,
            'Opp ORB': int,
            'Opp DRB': int,
            'Opp TRB': int,
            'Opp AST': int,
            'Opp STL': int,
            'Opp BLK': int,
            'Opp TOV': int,
            'Opp PF': int,
            },
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

    # filter out for select teams
    tm_gamelog = team_gamelog[team_gamelog["Tm"].isin([away_tm, home_tm])]

    # season group by
    tm_season = (
        tm_gamelog.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "count"),
            TmPts=("Tm Score", "sum"),
            OppPts=("Opp Score", "sum"),
            AvgPoss=("Poss Ext", "mean"),
            AvgShotEff=("Shoot Eff", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            FG=("FG", "sum"),
            FGA=("FGA", "sum"),
            TFG=("3P", "sum"),
            TFGA=("3PA", "sum"),
            FT=("FT", "sum"),
            FTA=("FTA", "sum"),
            ORB=("ORB", "sum"),
            TRB=("TRB", "sum"),
            AST=("AST", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            TOV=("TOV", "sum"),
            OppFG=("Opp FG", "sum"),
            OppFGA=("Opp FGA", "sum"),
            Opp3P=("Opp 3P", "sum"),
            OppFTA=("Opp FTA", "sum"),
            OppTOV=("Opp TOV", "sum"),
            OppBLK=("Opp BLK", "sum"),
            OppORB=("Opp ORB", "sum"),
            OppTRB=("Opp TRB", "sum"),
            W=("Total W", "max"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
    ).rename(
        columns={"TFG": "3P", "TFGA": "3PA"},
    )
    tm_season["FG%"] = tm_season["FG"] / tm_season["FGA"]
    tm_season["3P%"] = tm_season["3P"] / tm_season["3PA"]
    tm_season["FT%"] = tm_season["FT"] / tm_season["FTA"]
    tm_season["Poss"] = (
        tm_season["FGA"].astype(int)
        + (0.4 * tm_season["FTA"].astype(int))
        - (
            1.07
            * (
                tm_season["ORB"].astype(int)
                / (
                    tm_season["ORB"].astype(int)
                    + (
                        tm_season["OppTRB"].astype(int)
                        - tm_season["OppORB"].astype(int)
                    )
                )
            )
        )
        * (tm_season["FGA"].astype(int) - tm_season["FG"].astype(int))
        + tm_season["TOV"].astype(int)
    ) / tm_season["G"]
    tm_season["OppPoss"] = (
        tm_season["OppFGA"].astype(int)
        + (0.4 * (tm_season["OppFTA"].astype(int)))
        - (
            1.07
            * (tm_season["OppORB"].astype(int))
            / (
                tm_season["OppORB"].astype(int)
                + (tm_season["TRB"].astype(int) - tm_season["ORB"].astype(int))
            )
        )
        * (tm_season["OppFGA"].astype(int) - tm_season["OppFG"].astype(int))
        + tm_season["OppTOV"].astype(int)
    ) / tm_season["G"]
    tm_season["ShotEff"] = (
        tm_season["FG"].astype(int) + (0.5 * tm_season["3P"].astype(int))
    ) / tm_season["FGA"].astype(int)
    tm_season["OppShotEff"] = (
        tm_season["OppFG"].astype(int) + (0.5 * tm_season["Opp3P"].astype(int))
    ) / tm_season["OppFGA"].astype(int)
    tm_season["LuckFactor"] = tm_season["W"] - tm_season["Luck"]
    tm_season["OffEff"] = (
        (tm_season["TmPts"] / tm_season["Poss"]) / tm_season["G"]
    ) * (1 + (tm_season["LuckFactor"] / tm_season["G"]))
    tm_season["DefEff"] = (tm_season["OppPts"] / tm_season["OppPoss"]) / tm_season["G"]
    tm_season["TotShots"] = tm_season["FGA"] + tm_season["FTA"]
    tm_season["Make%"] = (tm_season["FG"] + tm_season["FT"]) / tm_season["TotShots"]
    tm_season["2Pt%"] = ((tm_season["FG"] - tm_season["3P"]) * 2) / tm_season["TmPts"]
    tm_season["3Pt%"] = (tm_season["3P"] * 3) / tm_season["TmPts"]
    tm_season["1Pt%"] = (tm_season["FT"]) / tm_season["TmPts"]
    # Four Factor Calculations
    tm_season["eFG%"] = (
        tm_season["FG"].astype(int) + (0.5 * tm_season["3P"].astype(int))
    ) / tm_season["FGA"].astype(int)
    tm_season["TOV%"] = tm_season["TOV"] / (
        tm_season["FGA"] + (tm_season["FTA"] * 0.44) + tm_season["TOV"]
    )
    tm_season["ORR"] = tm_season["ORB"] / (
        tm_season["ORB"] + (tm_season["TRB"] - tm_season["ORB"])
    )
    tm_season["FTR"] = tm_season["FT"] / tm_season["FGA"]

    # recent group by (7 game)
    tm_recent = pd.DataFrame()
    for tm in [away_tm, home_tm]:
        recent = (
            team_gamelog[team_gamelog["Tm"] == tm]
            .sort_values(by="G")
            .reset_index(drop=True)
        )
        recent = recent.iloc[-7:]

        tm_recent = pd.concat([tm_recent, recent]).reset_index(drop=True)

    tm_recency = (
        tm_recent.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "count"),
            TmPts=("Tm Score", "sum"),
            OppPts=("Opp Score", "sum"),
            AvgPoss=("Poss Ext", "mean"),
            AvgShotEff=("Shoot Eff", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            FG=("FG", "sum"),
            FGA=("FGA", "sum"),
            TFG=("3P", "sum"),
            TFGA=("3PA", "sum"),
            FT=("FT", "sum"),
            FTA=("FTA", "sum"),
            ORB=("ORB", "sum"),
            TRB=("TRB", "sum"),
            AST=("AST", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            TOV=("TOV", "sum"),
            OppFG=("Opp FG", "sum"),
            OppFGA=("Opp FGA", "sum"),
            Opp3P=("Opp 3P", "sum"),
            OppFTA=("Opp FTA", "sum"),
            OppTOV=("Opp TOV", "sum"),
            OppBLK=("Opp BLK", "sum"),
            OppORB=("Opp ORB", "sum"),
            OppTRB=("Opp TRB", "sum"),
            W=("W/L Flag", "sum"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
        .rename(
            columns={"TFG": "3P", "TFGA": "3PA"},
        )
    )
    tm_recency["FG%"] = tm_recency["FG"] / tm_recency["FGA"]
    tm_recency["3P%"] = tm_recency["3P"] / tm_recency["3PA"]
    tm_recency["FT%"] = tm_recency["FT"] / tm_recency["FTA"]
    tm_recency["Poss"] = (
        tm_recency["FGA"].astype(int)
        + (0.4 * tm_recency["FTA"].astype(int))
        - (
            1.07
            * (
                tm_recency["ORB"].astype(int)
                / (
                    tm_recency["ORB"].astype(int)
                    + (
                        tm_recency["OppTRB"].astype(int)
                        - tm_recency["OppORB"].astype(int)
                    )
                )
            )
        )
        * (tm_recency["FGA"].astype(int) - tm_recency["FG"].astype(int))
        + tm_recency["TOV"].astype(int)
    ) / tm_recency["G"]
    tm_recency["OppPoss"] = (
        tm_recency["OppFGA"].astype(int)
        + (0.4 * (tm_recency["OppFTA"].astype(int)))
        - (
            1.07
            * (tm_recency["OppORB"].astype(int))
            / (
                tm_recency["OppORB"].astype(int)
                + (tm_recency["TRB"].astype(int) - tm_recency["ORB"].astype(int))
            )
        )
        * (tm_recency["OppFGA"].astype(int) - tm_recency["OppFG"].astype(int))
        + tm_recency["OppTOV"].astype(int)
    ) / tm_recency["G"]
    tm_recency["ShotEff"] = (
        tm_recency["FG"].astype(int) + (0.5 * tm_recency["3P"].astype(int))
    ) / tm_recency["FGA"].astype(int)
    tm_recency["OppShotEff"] = (
        tm_recency["OppFG"].astype(int) + (0.5 * tm_recency["Opp3P"].astype(int))
    ) / tm_recency["OppFGA"].astype(int)
    tm_recency["LuckFactor"] = tm_recency["W"] - tm_recency["Luck"]
    tm_recency["OffEff"] = (
        (tm_recency["TmPts"] / tm_recency["Poss"]) / tm_recency["G"]
    ) * (1 + (tm_recency["LuckFactor"] / tm_recency["G"]))
    tm_recency["DefEff"] = (tm_recency["OppPts"] / tm_recency["OppPoss"]) / tm_recency[
        "G"
    ]
    tm_recency["TotShots"] = tm_recency["FGA"] + tm_recency["FTA"]
    tm_recency["Make%"] = (tm_recency["FG"] + tm_recency["FT"]) / tm_recency["TotShots"]
    tm_recency["2Pt%"] = ((tm_recency["FG"] - tm_recency["3P"]) * 2) / tm_recency[
        "TmPts"
    ]
    tm_recency["3Pt%"] = (tm_recency["3P"] * 3) / tm_recency["TmPts"]
    tm_recency["1Pt%"] = (tm_recency["FT"]) / tm_recency["TmPts"]
    # Four Factor Calculations
    tm_recency["eFG%"] = (
        tm_recency["FG"].astype(int) + (0.5 * tm_recency["3P"].astype(int))
    ) / tm_recency["FGA"].astype(int)
    tm_recency["TOV%"] = tm_recency["TOV"] / (
        tm_recency["FGA"] + (tm_recency["FTA"] * 0.44) + tm_recency["TOV"]
    )
    tm_recency["ORR"] = tm_recency["ORB"] / (
        tm_recency["ORB"] + (tm_recency["TRB"] - tm_recency["ORB"])
    )
    tm_recency["FTR"] = tm_recency["FT"] / tm_recency["FGA"]

    # combine season with recency (weighting recency slightly)
    tm_df = pd.DataFrame(
        data={
            "Tm": tm_season["Tm"],
            "Poss": ((tm_season["Poss"] * 2) + (tm_recency["Poss"] * 3)) / 5,
            "OppPoss": ((tm_season["OppPoss"] * 2) + (tm_recency["OppPoss"] * 3)) / 5,
            "ShotEff": ((tm_season["ShotEff"] * 2) + (tm_recency["ShotEff"] * 3)) / 5,
            "OppShotEff": (
                (tm_season["OppShotEff"] * 2) + (tm_recency["OppShotEff"] * 3)
            )
            / 5,
            "OffEff": ((tm_season["OffEff"] * 2) + (tm_recency["OffEff"] * 3)) / 5,
            "DefEff": ((tm_season["DefEff"] * 2) + (tm_recency["DefEff"] * 3)) / 5,
            "1Pt%": ((tm_season["1Pt%"] * 2) + (tm_recency["1Pt%"] * 3)) / 5,
            "2Pt%": ((tm_season["2Pt%"] * 2) + (tm_recency["2Pt%"] * 3)) / 5,
            "3Pt%": ((tm_season["3Pt%"] * 2) + (tm_recency["3Pt%"] * 3)) / 5,
            "Make%": ((tm_season["Make%"] * 2) + (tm_recency["Make%"] * 3)) / 5,
            "eFG%": ((tm_season["eFG%"] * 2) + (tm_recency["eFG%"] * 3)) / 5,
            "TOV%": ((tm_season["TOV%"] * 2) + (tm_recency["TOV%"] * 3)) / 5,
            "ORR": ((tm_season["ORR"] * 2) + (tm_recency["ORR"] * 3)) / 5,
            "FTR": ((tm_season["FTR"] * 2) + (tm_recency["FTR"] * 3)) / 5,
        }
    )

    # need to sort df with away team in [0] and home team in [1]
    tm_dict = {
        f"{away_tm}": 0,
        f"{home_tm}": 1,
    }
    tm_df["Sort Rnk"] = tm_df["Tm"].map(tm_dict)
    tm_df = (
        tm_df.sort_values(by="Sort Rnk")
        .drop(columns=["Sort Rnk"])
        .reset_index(drop=True)
    )

    return tm_df


def lg_stats(season, today):

    team_df = get_teamnm()
    team_df = team_df[team_df["Season"] == season]

    # pull in team data
    team_gamelog = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_tm_boxscores.csv"
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

    # season group by
    tm_season = (
        team_gamelog.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "count"),
            TmPts=("Tm Score", "sum"),
            OppPts=("Opp Score", "sum"),
            AvgPoss=("Poss Ext", "mean"),
            AvgShotEff=("Shoot Eff", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            FG=("FG", "sum"),
            FGA=("FGA", "sum"),
            TFG=("3P", "sum"),
            TFGA=("3PA", "sum"),
            FT=("FT", "sum"),
            FTA=("FTA", "sum"),
            ORB=("ORB", "sum"),
            TRB=("TRB", "sum"),
            AST=("AST", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            TOV=("TOV", "sum"),
            OppFG=("Opp FG", "sum"),
            OppFGA=("Opp FGA", "sum"),
            Opp3P=("Opp 3P", "sum"),
            OppFTA=("Opp FTA", "sum"),
            OppTOV=("Opp TOV", "sum"),
            OppBLK=("Opp BLK", "sum"),
            OppORB=("Opp ORB", "sum"),
            OppTRB=("Opp TRB", "sum"),
            W=("Total W", "max"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
    ).rename(
        columns={"TFG": "3P", "TFGA": "3PA"},
    )
    tm_season["FG%"] = tm_season["FG"] / tm_season["FGA"]
    tm_season["3P%"] = tm_season["3P"] / tm_season["3PA"]
    tm_season["FT%"] = tm_season["FT"] / tm_season["FTA"]
    tm_season["Poss"] = (
        tm_season["FGA"].astype(int)
        + (0.4 * tm_season["FTA"].astype(int))
        - (
            1.07
            * (
                tm_season["ORB"].astype(int)
                / (
                    tm_season["ORB"].astype(int)
                    + (
                        tm_season["OppTRB"].astype(int)
                        - tm_season["OppORB"].astype(int)
                    )
                )
            )
        )
        * (tm_season["FGA"].astype(int) - tm_season["FG"].astype(int))
        + tm_season["TOV"].astype(int)
    ) / tm_season["G"]
    tm_season["OppPoss"] = (
        tm_season["OppFGA"].astype(int)
        + (0.4 * (tm_season["OppFTA"].astype(int)))
        - (
            1.07
            * (tm_season["OppORB"].astype(int))
            / (
                tm_season["OppORB"].astype(int)
                + (tm_season["TRB"].astype(int) - tm_season["ORB"].astype(int))
            )
        )
        * (tm_season["OppFGA"].astype(int) - tm_season["OppFG"].astype(int))
        + tm_season["OppTOV"].astype(int)
    ) / tm_season["G"]
    tm_season["ShotEff"] = (
        tm_season["FG"].astype(int) + (0.5 * tm_season["3P"].astype(int))
    ) / tm_season["FGA"].astype(int)
    tm_season["OppShotEff"] = (
        tm_season["OppFG"].astype(int) + (0.5 * tm_season["Opp3P"].astype(int))
    ) / tm_season["OppFGA"].astype(int)
    tm_season["LuckFactor"] = tm_season["W"] - tm_season["Luck"]
    tm_season["OffEff"] = (
        (tm_season["TmPts"] / tm_season["Poss"]) / tm_season["G"]
    ) * (1 + (tm_season["LuckFactor"] / tm_season["G"]))
    tm_season["DefEff"] = (tm_season["OppPts"] / tm_season["OppPoss"]) / tm_season["G"]
    tm_season["TotShots"] = tm_season["FGA"] + tm_season["FTA"]
    tm_season["Make%"] = (tm_season["FG"] + tm_season["FT"]) / tm_season["TotShots"]
    tm_season["2Pt%"] = ((tm_season["FG"] - tm_season["3P"]) * 2) / tm_season["TmPts"]
    tm_season["3Pt%"] = (tm_season["3P"] * 3) / tm_season["TmPts"]
    tm_season["1Pt%"] = (tm_season["FT"]) / tm_season["TmPts"]
    # Four Factor Calculations
    tm_season["eFG%"] = (
        tm_season["FG"].astype(int) + (0.5 * tm_season["3P"].astype(int))
    ) / tm_season["FGA"].astype(int)
    tm_season["TOV%"] = tm_season["TOV"] / (
        tm_season["FGA"] + (tm_season["FTA"] * 0.44) + tm_season["TOV"]
    )
    tm_season["ORR"] = tm_season["ORB"] / (
        tm_season["ORB"] + (tm_season["TRB"] - tm_season["ORB"])
    )
    tm_season["FTR"] = tm_season["FT"] / tm_season["FGA"]

    # recent group by (7 game)
    tm_recent = pd.DataFrame()
    for tm in team_gamelog["Tm"].unique().tolist():
        recent = (
            team_gamelog[team_gamelog["Tm"] == tm]
            .sort_values(by="G")
            .reset_index(drop=True)
        )
        recent = recent.iloc[-7:]

        tm_recent = pd.concat([tm_recent, recent]).reset_index(drop=True)

    tm_recency = (
        tm_recent.groupby(["Tm"], observed=True)
        .agg(
            G=("G", "count"),
            TmPts=("Tm Score", "sum"),
            OppPts=("Opp Score", "sum"),
            AvgPoss=("Poss Ext", "mean"),
            AvgShotEff=("Shoot Eff", "mean"),
            AvgOffEff=("Off Eff", "mean"),
            AvgDefEff=("Def Eff", "mean"),
            FG=("FG", "sum"),
            FGA=("FGA", "sum"),
            TFG=("3P", "sum"),
            TFGA=("3PA", "sum"),
            FT=("FT", "sum"),
            FTA=("FTA", "sum"),
            ORB=("ORB", "sum"),
            TRB=("TRB", "sum"),
            AST=("AST", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            TOV=("TOV", "sum"),
            OppFG=("Opp FG", "sum"),
            OppFGA=("Opp FGA", "sum"),
            Opp3P=("Opp 3P", "sum"),
            OppFTA=("Opp FTA", "sum"),
            OppTOV=("Opp TOV", "sum"),
            OppBLK=("Opp BLK", "sum"),
            OppORB=("Opp ORB", "sum"),
            OppTRB=("Opp TRB", "sum"),
            W=("W/L Flag", "sum"),
            Luck=("Tm Luck", "sum"),
        )
        .reset_index()
        .rename(
            columns={"TFG": "3P", "TFGA": "3PA"},
        )
    )
    tm_recency["FG%"] = tm_recency["FG"] / tm_recency["FGA"]
    tm_recency["3P%"] = tm_recency["3P"] / tm_recency["3PA"]
    tm_recency["FT%"] = tm_recency["FT"] / tm_recency["FTA"]
    tm_recency["Poss"] = (
        tm_recency["FGA"].astype(int)
        + (0.4 * tm_recency["FTA"].astype(int))
        - (
            1.07
            * (
                tm_recency["ORB"].astype(int)
                / (
                    tm_recency["ORB"].astype(int)
                    + (
                        tm_recency["OppTRB"].astype(int)
                        - tm_recency["OppORB"].astype(int)
                    )
                )
            )
        )
        * (tm_recency["FGA"].astype(int) - tm_recency["FG"].astype(int))
        + tm_recency["TOV"].astype(int)
    ) / tm_recency["G"]
    tm_recency["OppPoss"] = (
        tm_recency["OppFGA"].astype(int)
        + (0.4 * (tm_recency["OppFTA"].astype(int)))
        - (
            1.07
            * (tm_recency["OppORB"].astype(int))
            / (
                tm_recency["OppORB"].astype(int)
                + (tm_recency["TRB"].astype(int) - tm_recency["ORB"].astype(int))
            )
        )
        * (tm_recency["OppFGA"].astype(int) - tm_recency["OppFG"].astype(int))
        + tm_recency["OppTOV"].astype(int)
    ) / tm_recency["G"]
    tm_recency["ShotEff"] = (
        tm_recency["FG"].astype(int) + (0.5 * tm_recency["3P"].astype(int))
    ) / tm_recency["FGA"].astype(int)
    tm_recency["OppShotEff"] = (
        tm_recency["OppFG"].astype(int) + (0.5 * tm_recency["Opp3P"].astype(int))
    ) / tm_recency["OppFGA"].astype(int)
    tm_recency["LuckFactor"] = tm_recency["W"] - tm_recency["Luck"]
    tm_recency["OffEff"] = (
        (tm_recency["TmPts"] / tm_recency["Poss"]) / tm_recency["G"]
    ) * (1 + (tm_recency["LuckFactor"] / tm_recency["G"]))
    tm_recency["DefEff"] = (tm_recency["OppPts"] / tm_recency["OppPoss"]) / tm_recency[
        "G"
    ]
    tm_recency["TotShots"] = tm_recency["FGA"] + tm_recency["FTA"]
    tm_recency["Make%"] = (tm_recency["FG"] + tm_recency["FT"]) / tm_recency["TotShots"]
    tm_recency["2Pt%"] = ((tm_recency["FG"] - tm_recency["3P"]) * 2) / tm_recency[
        "TmPts"
    ]
    tm_recency["3Pt%"] = (tm_recency["3P"] * 3) / tm_recency["TmPts"]
    tm_recency["1Pt%"] = (tm_recency["FT"]) / tm_recency["TmPts"]
    # Four Factor Calculations
    tm_recency["eFG%"] = (
        tm_recency["FG"].astype(int) + (0.5 * tm_recency["3P"].astype(int))
    ) / tm_recency["FGA"].astype(int)
    tm_recency["TOV%"] = tm_recency["TOV"] / (
        tm_recency["FGA"] + (tm_recency["FTA"] * 0.44) + tm_recency["TOV"]
    )
    tm_recency["ORR"] = tm_recency["ORB"] / (
        tm_recency["ORB"] + (tm_recency["TRB"] - tm_recency["ORB"])
    )
    tm_recency["FTR"] = tm_recency["FT"] / tm_recency["FGA"]

    # combine season with recency (weighting recency slightly)
    tm_df = pd.DataFrame(
        data={
            "Tm": tm_season["Tm"],
            "Poss": ((tm_season["Poss"] * 2) + (tm_recency["Poss"] * 3)) / 5,
            "OppPoss": ((tm_season["OppPoss"] * 2) + (tm_recency["OppPoss"] * 3)) / 5,
            "ShotEff": ((tm_season["ShotEff"] * 2) + (tm_recency["ShotEff"] * 3)) / 5,
            "OppShotEff": (
                (tm_season["OppShotEff"] * 2) + (tm_recency["OppShotEff"] * 3)
            )
            / 5,
            "OffEff": ((tm_season["OffEff"] * 2) + (tm_recency["OffEff"] * 3)) / 5,
            "DefEff": ((tm_season["DefEff"] * 2) + (tm_recency["DefEff"] * 3)) / 5,
            "1Pt%": ((tm_season["1Pt%"] * 2) + (tm_recency["1Pt%"] * 3)) / 5,
            "2Pt%": ((tm_season["2Pt%"] * 2) + (tm_recency["2Pt%"] * 3)) / 5,
            "3Pt%": ((tm_season["3Pt%"] * 2) + (tm_recency["3Pt%"] * 3)) / 5,
            "Make%": ((tm_season["Make%"] * 2) + (tm_recency["Make%"] * 3)) / 5,
            "eFG%": ((tm_season["eFG%"] * 2) + (tm_recency["eFG%"] * 3)) / 5,
            "TOV%": ((tm_season["TOV%"] * 2) + (tm_recency["TOV%"] * 3)) / 5,
            "ORR": ((tm_season["ORR"] * 2) + (tm_recency["ORR"] * 3)) / 5,
            "FTR": ((tm_season["FTR"] * 2) + (tm_recency["FTR"] * 3)) / 5,
        }
    )

    # Rank the Four Factor Metrics
    tm_df["eFG% Rnk"] = tm_df["eFG%"].rank(method="max", ascending=False)
    tm_df["TOV% Rnk"] = tm_df["TOV%"].rank(method="max", ascending=True)
    tm_df["ORR Rnk"] = tm_df["eFG%"].rank(method="max", ascending=False)
    tm_df["FTR Rnk"] = tm_df["eFG%"].rank(method="max", ascending=False)
    tm_df["Four Factors Rnk"] = round(
        (tm_df["eFG% Rnk"] + tm_df["TOV% Rnk"] + tm_df["ORR Rnk"] + tm_df["FTR Rnk"])
        / (
            tm_df["eFG% Rnk"].max()
            + tm_df["TOV% Rnk"].max()
            + tm_df["ORR Rnk"].max()
            + tm_df["FTR Rnk"].max()
        ),
        5,
    ).rank(method="max", ascending=True)

    # sort teams alphabetically for now
    tm_df = tm_df.sort_values(by="Tm").reset_index(drop=True)

    return tm_df


def game(tm_df, neutral):
    score_df = pd.DataFrame()
    home_tm = tm_df["Tm"][1]

    # simulate a game
    for tm in tm_df["Tm"]:
        tm_possessions = math.ceil(
            (tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["Poss"][0])
        )
        opp_possessions = math.ceil(
            (tm_df[~(tm_df["Tm"] == tm)].reset_index(drop=True)["OppPoss"][0])
        )

        possessions = math.ceil(((tm_possessions) + (opp_possessions)) / 2)
        # eff = 1 + ((
        #     tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["OffEff"][0]
        #     - (tm_df[~(tm_df["Tm"] == tm)].reset_index(drop=True)["DefEff"][0])
        # ))
        eff = 1

        score = 0
        while possessions > 0:
            # shot
            shot = random.random()

            # shot was made
            if shot <= (
                tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["ShotEff"][0]
                * tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["OffEff"][0]
            ):

                if shot <= (
                    tm_df[~(tm_df["Tm"] == tm)].reset_index(drop=True)["OppShotEff"][0]
                    * tm_df[~(tm_df["Tm"] == tm)].reset_index(drop=True)["DefEff"][0]
                ):
                    points = random.random()

                    # 2 Pt
                    if (
                        points
                        <= tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["2Pt%"][0]
                    ):
                        score += 2
                    # 3 Pt
                    elif (
                        points
                        > tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["2Pt%"][0]
                    ) & (
                        points
                        <= (
                            tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["3Pt%"][0]
                            + tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["2Pt%"][0]
                        )
                    ):
                        score += 3
                    # 1 Pt
                    elif points > (
                        1 - tm_df[tm_df["Tm"] == tm].reset_index(drop=True)["1Pt%"][0]
                    ):
                        score += 1
                else:
                    score += 0

            # shot was missed
            else:
                score += 0

            possessions += -1

        if not neutral:
            if tm == home_tm:
                score = (score * eff) + 2.5
        else:
            score = score * eff

        # input into final df
        score_df = pd.concat(
            [score_df, pd.DataFrame(data={"Tm": [tm], "Score": [math.ceil(score)]})]
        ).reset_index(drop=True)

    return score_df


def game_sim(season, away_tm, home_tm, today, neutral, n):

    pregame_df = pregame(season, away_tm, home_tm, today)

    results_df = pd.DataFrame()
    for i in range(n):

        # play the game
        game_df = game(pregame_df, neutral)

        # tally results
        if game_df["Score"][0] >= game_df["Score"][1]:
            winner = game_df["Tm"][0]  # home team wins
            pt_spread = game_df["Score"][0] - game_df["Score"][1]
        else:
            winner = game_df["Tm"][1]  # away team wins
            pt_spread = game_df["Score"][1] - game_df["Score"][0]

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    data={
                        "Game": [i + 1],
                        "Winner": [winner],
                        "Point Spread": [pt_spread],
                    }
                ),
            ]
        ).reset_index(drop=True)

    # find winner from monte carlo
    home_tm_w = results_df["Winner"].str.count(f"{home_tm.split("(")[0]}").sum()
    away_tm_w = results_df["Winner"].str.count(f"{away_tm.split("(")[0]}").sum()

    if home_tm_w > away_tm_w:
        game_winner = home_tm
        win_pct = home_tm_w / n
    else:
        game_winner = away_tm
        win_pct = away_tm_w / n

    # find average score diff
    point_diff = results_df[results_df["Winner"] == game_winner]["Point Spread"].mean()
    if results_df[~(results_df["Winner"] == game_winner)]["Point Spread"].empty:
        opp_point_diff = 0
    else:
        opp_point_diff = results_df[~(results_df["Winner"] == game_winner)][
            "Point Spread"
        ].mean()

    tot_point_diff = point_diff - opp_point_diff

    if game_winner == away_tm:
        aw_point_diff = tot_point_diff * -1
        hm_point_diff = tot_point_diff
    else:
        aw_point_diff = tot_point_diff
        hm_point_diff = tot_point_diff * -1

    gm_results = pd.DataFrame(
        data={
            "Tm": [away_tm, home_tm],
            "Win Prob.": [away_tm_w / n, home_tm_w / n],
            "Point Diff": [aw_point_diff, hm_point_diff],
        }
    )

    return gm_results


def sim_graph(season, away_tm, home_tm, sim_results_df, date_label):

    team_df = get_teamnm()

    away_abbr = team_df[
        (team_df["Tm Name"] == away_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Gamelog Name"][0]
    home_abbr = team_df[
        (team_df["Tm Name"] == home_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Gamelog Name"][0]

    away_tm_color = team_df[
        (team_df["Tm Name"] == away_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Tm Primary Color"][0]
    home_tm_color = team_df[
        (team_df["Tm Name"] == home_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Tm Primary Color"][0]

    stdev = 10

    if sim_results_df["Win Prob."][0] > sim_results_df["Win Prob."][1]:
        gm_winner = sim_results_df["Tm"][0]
        pt_spread = sim_results_df["Point Diff"][0] * -1
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]
    else:
        gm_winner = sim_results_df["Tm"][1]
        pt_spread = sim_results_df["Point Diff"][1] * -1
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]

    sim_results = [gm_winner, pt_spread, away_win_prob, home_win_prob]

    # Set style
    # plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Graph W Probability
    x = np.arange(
        sim_results[1] - (3.5 * stdev),
        sim_results[1] + (3.5 * stdev),
        0.01,
    )
    y = norm.pdf(x, sim_results[1], stdev)

    # Plot the distribution curve
    ax.plot(x, y, color="black", lw=2, alpha=0.7)

    if away_win_prob > home_win_prob:
        # Fill areas
        ax.fill_between(
            x,
            0,
            y,
            where=x <= 0,
            facecolor=home_tm_color,
            alpha=0.3,
            label=f"{home_abbr} Win Probability: {home_win_prob:.1%}",
        )
        ax.fill_between(
            x,
            0,
            y,
            where=x > 0,
            facecolor=away_tm_color,
            alpha=0.3,
            label=f"{away_abbr} Win Probability: {away_win_prob:.1%}",
        )
    else:
        # Fill areas
        ax.fill_between(
            x,
            0,
            y,
            where=x > 0,
            facecolor=home_tm_color,
            alpha=0.3,
            label=f"{home_abbr} Win Probability: {home_win_prob:.1%}",
        )
        ax.fill_between(
            x,
            0,
            y,
            where=x <= 0,
            facecolor=away_tm_color,
            alpha=0.3,
            label=f"{away_abbr} Win Probability: {away_win_prob:.1%}",
        )

    # Add mean line and annotation
    ax.axvline((sim_results[1]), c="black", ls="--", alpha=0.5)
    ax.annotate(
        f"Expected Point Differential: {sim_results[1]:+.1f}",
        xy=((sim_results[1]), y.max()),
        xytext=(10, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    # # Add standard deviation ranges
    # for i in [1, 2]:
    #     ax.axvline((sim_results[1]) + i * stdev, c="gray", ls=":", alpha=0.3)
    #     ax.axvline((sim_results[1]) - i * stdev, c="gray", ls=":", alpha=0.3)
    #     ax.annotate(
    #         f"{i}σ",
    #         xy=((sim_results[1]) + i * stdev, y.max() / 4),
    #         ha="center",
    #         color="gray",
    #     )
    #     ax.annotate(
    #         f"-{i}σ",
    #         xy=((sim_results[1]) - i * stdev, y.max() / 4),
    #         ha="center",
    #         color="gray",
    #     )

    # Add game information box
    if date_label:
        info_text = (
            f"Game Details:\n"
            f"Gameday: {input("What day is the game? ")}\n"
            f"Location: @ {home_tm}"
        )
    else:
        info_text = f"Game Details:\n" f"Location: @ {home_tm}"
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
    )

    # Customize plot
    ax.set_title(
        f"{away_tm} vs {home_tm}\n Win Probability Distribution", fontsize=14, pad=20
    )
    ax.set_xlabel("Point Differential", fontsize=12)
    # ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # hide y-axis ticks
    ax.tick_params(axis="y", labelleft=False)

    plt.tight_layout()

    # plt.show()
    return plt.show()


def sim_donut_graph(season, away_tm, home_tm, sim_results_df, hm_tm_prim, aw_tm_prim):

    team_df = get_teamnm()

    away_abbr = team_df[
        (team_df["Tm Name"] == away_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Gamelog Name"][0]
    home_abbr = team_df[
        (team_df["Tm Name"] == home_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Gamelog Name"][0]

    away_tm_color = team_df[
        (team_df["Tm Name"] == away_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Tm Primary Color"][0]
    home_tm_color = team_df[
        (team_df["Tm Name"] == home_tm) & (team_df["Season"] == season)
    ].reset_index(drop=True)["Tm Primary Color"][0]

    stdev = 10

    if sim_results_df["Win Prob."][0] > sim_results_df["Win Prob."][1]:
        gm_winner = sim_results_df["Tm"][0]
        pt_spread = sim_results_df["Point Diff"][0] * -1
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]
    else:
        gm_winner = sim_results_df["Tm"][1]
        pt_spread = sim_results_df["Point Diff"][1] * -1
        away_win_prob = sim_results_df["Win Prob."][0]
        home_win_prob = sim_results_df["Win Prob."][1]

    sim_results = [gm_winner, pt_spread, away_win_prob, home_win_prob]
    win_prob = [away_win_prob, home_win_prob]

    home_tm_color_prim = get_teamcolor_prim(home_tm, season)
    home_tm_color_sec = get_teamcolor_sec(home_tm, season)
    away_tm_color_prim = get_teamcolor_prim(away_tm, season)
    away_tm_color_sec = get_teamcolor_sec(away_tm, season)
    
    if hm_tm_prim:
        home_tm_color = home_tm_color_prim
    else:
        home_tm_color = home_tm_color_sec
    if aw_tm_prim:
        away_tm_color = away_tm_color_prim
    else:
        away_tm_color = away_tm_color_sec
    game_colors = [away_tm_color, home_tm_color]

    # explosions
    explode = [0.05, 0.05]

    # Pie Chart
    plt.pie(
        win_prob,
        colors=game_colors,
        textprops={'color': 'white', 'fontsize': 11,},
        autopct="%1.1f%%",
        pctdistance=0.82,
        explode=explode,
        wedgeprops={'linewidth': 1, 'edgecolor': '000000', 'width': 0.8,},
        startangle=90,
    )

    # draw circle
    center_circle = plt.Circle((0,0), 0.7, fc='white')
    fig = plt.gcf()

    # adding circle in Pie Chart
    fig.gca().add_artist(center_circle)

    # # Add game information to center of chart
    plt.text(0, 0, f"Location: @ {home_abbr}\n Point Spread: {round(sim_results[1], 1)}", ha="center", va="center", fontsize=10,)

    # Add Legends
    plt.legend([away_abbr, home_abbr], loc="upper right")

    # Ensuring circle proportion
    plt.axis('equal')

    # Title
    plt.title(f"{away_abbr} @ {home_abbr}\n", fontsize=14)
    plt.suptitle('Win Probability', x=0.5, y=0.92, fontsize=10)
    # plt.show()

    return plt.show()


def game_radar(season, away_tm, home_tm, today):

    tmname_df = get_teamnm()
    hm_color = tmname_df[tmname_df["Tm Name"] == home_tm].reset_index(drop=True)[
        "Tm Primary Color"
    ][0]
    aw_color = tmname_df[tmname_df["Tm Name"] == away_tm].reset_index(drop=True)[
        "Tm Primary Color"
    ][0]

    lg_stats_df = lg_stats(season, today)

    tm_df = lg_stats_df[lg_stats_df["Tm"].isin([away_tm, home_tm])].reset_index(
        drop=True
    )

    # need to sort df with away team in [0] and home team in [1]
    tm_dict = {
        f"{away_tm}": 0,
        f"{home_tm}": 1,
    }
    tm_df["Sort Rnk"] = tm_df["Tm"].map(tm_dict)
    tm_df = (
        tm_df.sort_values(by="Sort Rnk")
        .drop(columns=["Sort Rnk"])
        .reset_index(drop=True)
    )

    # quantile calculations for Four Factor stats
    # eFG%
    eFG_quan1 = np.quantile(tm_df["eFG% Rnk"], 0.2)
    eFG_quan2 = np.quantile(tm_df["eFG% Rnk"], 0.4)
    eFG_quan3 = np.quantile(tm_df["eFG% Rnk"], 0.6)
    eFG_quan4 = np.quantile(tm_df["eFG% Rnk"], 0.8)
    eFG_quan5 = np.quantile(tm_df["eFG% Rnk"], 1.0)
    # TOV%
    TOV_quan1 = np.quantile(tm_df["TOV% Rnk"], 0.2)
    TOV_quan2 = np.quantile(tm_df["TOV% Rnk"], 0.4)
    TOV_quan3 = np.quantile(tm_df["TOV% Rnk"], 0.6)
    TOV_quan4 = np.quantile(tm_df["TOV% Rnk"], 0.8)
    TOV_quan5 = np.quantile(tm_df["TOV% Rnk"], 1.0)
    # ORR
    ORR_quan1 = np.quantile(tm_df["ORR Rnk"], 0.2)
    ORR_quan2 = np.quantile(tm_df["ORR Rnk"], 0.4)
    ORR_quan3 = np.quantile(tm_df["ORR Rnk"], 0.6)
    ORR_quan4 = np.quantile(tm_df["ORR Rnk"], 0.8)
    ORR_quan5 = np.quantile(tm_df["ORR Rnk"], 1.0)
    # FTR
    FTR_quan1 = np.quantile(tm_df["FTR Rnk"], 0.2)
    FTR_quan2 = np.quantile(tm_df["FTR Rnk"], 0.4)
    FTR_quan3 = np.quantile(tm_df["FTR Rnk"], 0.6)
    FTR_quan4 = np.quantile(tm_df["FTR Rnk"], 0.8)
    FTR_quan5 = np.quantile(tm_df["FTR Rnk"], 1.0)
    # FF
    FF_quan1 = np.quantile(tm_df["Four Factors Rnk"], 0.2)
    FF_quan2 = np.quantile(tm_df["Four Factors Rnk"], 0.4)
    FF_quan3 = np.quantile(tm_df["Four Factors Rnk"], 0.6)
    FF_quan4 = np.quantile(tm_df["Four Factors Rnk"], 0.8)
    FF_quan5 = np.quantile(tm_df["Four Factors Rnk"], 1.0)

    away_stats_df = tm_df.iloc[[0]]
    home_stats_df = tm_df.iloc[[1]].reset_index(drop=True)

    categories = [
        "eFG%",
        "TOV%",
        "ORR",
        "FTR",
        "Four Factors",
    ]

    # loop through categories
    for category in categories:
        if category == "eFG%":
            quan1 = eFG_quan1
            quan2 = eFG_quan2
            quan3 = eFG_quan3
            quan4 = eFG_quan4
            quan5 = eFG_quan5
        elif category == "TOV%":
            quan1 = TOV_quan1
            quan2 = TOV_quan2
            quan3 = TOV_quan3
            quan4 = TOV_quan4
            quan5 = TOV_quan5
        elif category == "ORR":
            quan1 = ORR_quan1
            quan2 = ORR_quan2
            quan3 = ORR_quan3
            quan4 = ORR_quan4
            quan5 = ORR_quan5
        elif category == "FTR":
            quan1 = FTR_quan1
            quan2 = FTR_quan2
            quan3 = FTR_quan3
            quan4 = FTR_quan4
            quan5 = FTR_quan5
        elif category == "Four Factors":
            quan1 = FF_quan1
            quan2 = FF_quan2
            quan3 = FF_quan3
            quan4 = FF_quan4
            quan5 = FF_quan5

        if away_stats_df[f"{category} Rnk"][0] <= quan1:
            away = 5
        elif (away_stats_df[f"{category} Rnk"][0] > quan1) & (
            away_stats_df[f"{category} Rnk"][0] <= quan2
        ):
            away = 4
        elif (away_stats_df[f"{category} Rnk"][0] > quan2) & (
            away_stats_df[f"{category} Rnk"][0] <= quan3
        ):
            away = 3
        elif (away_stats_df[f"{category} Rnk"][0] > quan3) & (
            away_stats_df[f"{category} Rnk"][0] <= quan4
        ):
            away_eFG = 2
        elif away_stats_df[f"{category} Rnk"][0] > quan4:
            away = 1

        if home_stats_df[f"{category} Rnk"][0] <= quan1:
            home = 5
        elif (home_stats_df[f"{category} Rnk"][0] > quan1) & (
            home_stats_df[f"{category} Rnk"][0] <= quan2
        ):
            home = 4
        elif (home_stats_df[f"{category} Rnk"][0] > quan2) & (
            home_stats_df[f"{category} Rnk"][0] <= quan3
        ):
            home = 3
        elif (home_stats_df[f"{category} Rnk"][0] > quan3) & (
            home_stats_df[f"{category} Rnk"][0] <= quan4
        ):
            home = 2
        elif home_stats_df[f"{category} Rnk"][0] > quan4:
            home = 1

        if category == "eFG%":
            away_eFG = away
            home_eFG = home
        elif category == "TOV%":
            away_TOV = away
            home_TOV = home
        elif category == "ORR":
            away_ORR = away
            home_ORR = home
        elif category == "FTR":
            away_FTR = away
            home_FTR = home
        elif category == "Four Factors":
            away_FF = away
            home_FF = home

    away_stats = pd.DataFrame(
        data={
            "Tm": [away_tm],
            "eFG%": [away_eFG],
            # "eFG%": [len(lg_stats_df) - away_stats_df["eFG% Rnk"][0]],
            "TOV%": [away_TOV],
            # "TOV%": [len(lg_stats_df) - away_stats_df["TOV% Rnk"][0]],
            "ORR": [away_ORR],
            # "ORR": [len(lg_stats_df) - away_stats_df["ORR Rnk"][0]],
            "FTR": [away_FTR],
            # "FTR": [len(lg_stats_df) - away_stats_df["FTR Rnk"][0]],
            # "Four Factors": [away_FF],
        }
    )
    home_stats = pd.DataFrame(
        data={
            "Tm": [home_tm],
            "eFG%": [home_eFG],
            # "eFG%": [len(lg_stats_df) - home_stats_df["eFG% Rnk"][0]],
            "TOV%": [home_TOV],
            # "TOV%": [len(lg_stats_df) - home_stats_df["TOV% Rnk"][0]],
            "ORR": [home_ORR],
            # "ORR": [len(lg_stats_df) - home_stats_df["ORR Rnk"][0]],
            "FTR": [home_FTR],
            # "FTR": [len(lg_stats_df) - home_stats_df["FTR Rnk"][0]],
            # "Four Factors": [home_FF],
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[
                # away_stats[categories[4]][0],
                away_stats[categories[0]][0],
                away_stats[categories[1]][0],
                away_stats[categories[2]][0],
                away_stats[categories[3]][0],
            ],
            theta=categories,
            fill="toself",
            name=f"{away_tm}",
            marker=dict(size=5, color=f"{aw_color}"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                # home_stats[categories[4]][0],
                home_stats[categories[0]][0],
                home_stats[categories[1]][0],
                home_stats[categories[2]][0],
                home_stats[categories[3]][0],
            ],
            theta=categories,
            fill="toself",
            name=f"{home_tm}",
            marker=dict(size=5, color=f"{hm_color}"),
        )
    )

    fig.update_layout(
        title=dict(text=f"{away_tm} vs. {home_tm}"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        # polar=dict(radialaxis=dict(visible=True, range=[0, len(lg_stats_df)])),
        showlegend=True,
    )

    # fig.show()

    return fig.show()
