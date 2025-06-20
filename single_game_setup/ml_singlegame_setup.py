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
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils.get_data import *
from utils.team_dict import *
from utils.beautiful_soup_helper import *

pd.set_option("future.no_silent_downcasting", True)


def gamelog_setup(season, tm_name, gm_date):

    # make sure 'today' is a date and not string
    gm_date = pd.to_datetime(gm_date)

    team_df = get_teamnm()
    team_df = team_df[team_df["Season"] == season]

    tm_df = team_df[team_df["Tm Name"] == tm_name].reset_index(drop=True)

    # tm_url = f"https://www.sports-reference.com/cbb/schools/{tm_df['Ref Name'][0]}/men/{season}-gamelogs.html"
    # team_gamelog = read_gamelog(tm_url)
    # team_gamelog["Tm"] = tm_name

    # temporary workaround
    gamelog = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_tm_boxscores.csv",
    )
    team_gamelog = gamelog[gamelog["Tm"] == tm_df["Tm Name"][0]]

    team_gamelog = team_gamelog[
        (
            ~(team_gamelog["Opp"].isin(["", "Opponent", "Opp"]))
            & ~(team_gamelog["W/L"].isin(["", np.nan, pd.NA]))
            & ~(team_gamelog["FG"].isin(["", np.nan, pd.NA]))
            & ~(team_gamelog["G"].isin(["", np.nan, pd.NA]))
        )
    ].reset_index(drop=True)

    # add season column
    team_gamelog["Season"] = season

    # set dtypes
    team_gamelog = team_gamelog.astype(
        {
            "G": int,
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
        },
    )

    # friendly win/loss
    team_gamelog.loc[team_gamelog["W/L"].str.contains("W"), "W/L Flag"] = 1
    team_gamelog["W/L Flag"] = team_gamelog["W/L Flag"].fillna(0)

    team_gamelog["W/L"] = team_gamelog["W/L"].str[0]

    # running sum W
    team_gamelog["Total W"] = team_gamelog.groupby(["Tm"])["W/L Flag"].cumsum()
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
    team_gamelog = team_gamelog.drop(
        columns=["Streak Value", "Start Streak", "Streak Id", "Running Streak"]
    )

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

    # flag conference games (regular season)
    team_gamelog.loc[(team_gamelog["Game Type"] == "REG (Conf)"), "Conf Game"] = 1
    team_gamelog["Conf Game"] = team_gamelog["Conf Game"].fillna(0)

    # flag postseason games
    team_gamelog.loc[
        ~(team_gamelog["Game Type"].str.contains("REG")), "Postseason Game"
    ] = 1
    team_gamelog["Postseason Game"] = team_gamelog["Postseason Game"].fillna(0)

    # rename Home/Away/Neutral to 2/1/0
    location_dict = {
        "@": 1,
        "N": 0,
    }
    team_gamelog["Location"] = (team_gamelog["Location"].map(location_dict)).fillna(2)
    team_gamelog["Location"] = team_gamelog["Location"].astype(int)

    # limit data to only show up to defined date (not including)
    team_gamelog = (
        team_gamelog[
            (team_gamelog["Date"].astype("datetime64[ns]") < gm_date)
        ].reset_index(drop=True)
        # .drop(columns=["Rank"])
    )

    return team_gamelog


def rolling_gamedata(season, hm_tm, aw_tm, gm_date):

    # pull both team's gamelogs up to game
    hm_gm_log = gamelog_setup(season, hm_tm, gm_date)
    aw_gm_log = gamelog_setup(season, aw_tm, gm_date)

    gm_log = pd.concat([hm_gm_log, aw_gm_log])

    # group gamelog stats (season avg.)
    gm_df = (
        gm_log.groupby(["Tm"], observed=True)
        .agg(
            W=("W/L Flag", "sum"),
            Wpct=("Total W%", "last"),
            Poss=("Poss Ext", "median"),
            OppPoss=("Opp Poss Ext", "median"),
            OffEff=("Off Eff", "median"),
            ShootEff=("Shoot Eff", "median"),
            DefEff=("Def Eff", "median"),
            OppShootEff=("Opp Shoot Eff", "median"),
            TmLuckW=("Tm Luck", "sum"),
            Pts=("Tm Score", "median"),
            OppPts=("Opp Score", "median"),
            # start boxscore stats
            FGpct=("FG%", "median"),
            P3pct=("3P%", "median"),
            P2pct=("2P%", "median"),
            eFGpct=("eFG%", "median"),
            FTpct=("FT%", "median"),
            ORB=("ORB", "median"),
            DRB=("DRB", "median"),
            TRB=("TRB", "median"),
            AST=("AST", "median"),
            STL=("STL", "median"),
            BLK=("BLK", "median"),
            TOV=("TOV", "median"),
            PF=("PF", "median"),
            # opponent boxscore stats
            OppFGpct=("FG%", "median"),
            OppP3pct=("3P%", "median"),
            OppP2pct=("Opp 2P%", "median"),
            OppeFGpct=("eFG%", "median"),
            OppFTpct=("FT%", "median"),
            OppORB=("ORB", "median"),
            OppDRB=("DRB", "median"),
            OppTRB=("TRB", "median"),
            OppAST=("AST", "median"),
            OppSTL=("STL", "median"),
            OppBLK=("BLK", "median"),
            OppTOV=("TOV", "median"),
            OppPF=("PF", "median"),
        )
        .reset_index()
        .rename(
            columns={
                "P3pct": "3Ppct",
                "OppP3pct": "Opp3Ppct",
                "P2pct": "2Ppct",
                "OppP2pct": "Opp2Ppct",
            }
        )
    )
    gm_df["TmLuck"] = gm_df["W"] - gm_df["TmLuckW"]
    gm_df["Game Date"] = gm_date
    gm_df["Season"] = season

    # transform gm_df into single line game for game results df
    hm = (
        gm_df[gm_df["Tm"] == hm_tm]
        .add_prefix("Hm_")
        .rename(columns={"Hm_Game Date": "Game Date"})
    )
    aw = (
        gm_df[gm_df["Tm"] == aw_tm]
        .add_prefix("Aw_")
        .rename(columns={"Aw_Game Date": "Game Date"})
    )

    full_gm_df = (aw.merge(hm, how="outer", on=["Game Date"])).reset_index(drop=True)
    full_gm_df["Matchup"] = full_gm_df["Aw_Tm"] + " vs. " + full_gm_df["Hm_Tm"]

    return full_gm_df


def season_data(season):

    # read results df
    results_df = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_results.csv",
    )
    season_df = pd.DataFrame()

    # for each result, run rolling_gamedata()
    for n, matchup in enumerate(results_df["Matchup"].tolist()):
        logger.info(
            f"{n+1}/{len(results_df["Matchup"].tolist())}: {results_df["Matchup"].tolist()[n]}"
        )

        hm_tm = results_df["Home Team"][n]
        aw_tm = results_df["Away Team"][n]
        gm_date = results_df["Game Date"][n]

        gamelog_stats = rolling_gamedata(season, hm_tm, aw_tm, gm_date)

        gamelog_stats = gamelog_stats.merge(
            results_df[
                [
                    "Matchup",
                    "Home Team",
                    "Home Rk",
                    "Away Team",
                    "Away Rk",
                    "Home W",
                    "Home Pt Diff",
                    "Neutral Game",
                    "Conference Game",
                    "Postseason Game",
                    "NCAA Tourney Game",
                ]
            ],
            how="inner",
            left_on=["Matchup", "Aw_Tm", "Hm_Tm"],
            right_on=["Matchup", "Away Team", "Home Team"],
        )

        season_df = (
            pd.concat([season_df, gamelog_stats])
            .drop_duplicates(subset=["Matchup", "Game Date"])
            .reset_index(drop=True)
        )

    # save .csv file
    season_df.to_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_matchup_results.csv",
        index=False,
    )

    return season_df


def single_game_model(data_seasons, today, matchup, game_details):

    # load up matchup results for training
    matchup_df = pd.DataFrame()
    for season in data_seasons:
        tmp_df = pd.read_csv(
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_matchup_results.csv",
        )
        tmp_df = tmp_df.astype({"Game Date": "datetime64[ns]"})

        # concat all years of data into one df
        matchup_df = pd.concat([matchup_df, tmp_df]).reset_index(drop=True)

    # make sure only data from before "today"
    matchup_df = (
        matchup_df[matchup_df["Game Date"] < pd.to_datetime(today)]
        .reset_index(drop=True)
        .rename(columns={"Hm_Season": "Season"})
        .drop(columns=["Aw_Season"])
    )

    # seed for NCAA Tourney games
    ncaa_df = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season_bracketology.csv"
    )
    ncaa_df = ncaa_df[ncaa_df["season"].isin(data_seasons)]
    ncaa_df.loc[ncaa_df["season"].isin(data_seasons), "NCAA Tourney Game"] = 1
    matchup_df = matchup_df.merge(
        ncaa_df.rename(
            columns={
                "tm": "Aw_Tm",
                "seed": "Aw_Seed",
                "season": "Season",
            }
        )[["Season", "Aw_Tm", "NCAA Tourney Game", "Aw_Seed"]],
        how="left",
        on=["Season", "Aw_Tm", "NCAA Tourney Game"],
    ).merge(
        ncaa_df.rename(
            columns={
                "tm": "Hm_Tm",
                "seed": "Hm_Seed",
                "season": "Season",
            }
        )[["Season", "Hm_Tm", "NCAA Tourney Game", "Hm_Seed"]],
        how="left",
        on=["Season", "Hm_Tm", "NCAA Tourney Game"],
    )

    # data for the matchup
    hm_tm = matchup.split(" vs. ")[1]
    aw_tm = matchup.split(" vs. ")[0]
    matchup_data = rolling_gamedata(
        data_seasons[-1],
        hm_tm,
        aw_tm,
        pd.to_datetime(today),
    )

    # get team ratings
    tm_ratings = tm_rating(data_seasons[-1], today)
    aw_tm_rating = (
        tm_ratings[tm_ratings["Tm"] == aw_tm]
        .reset_index(drop=True)
        .rename(
            columns={
                "Tm": "Aw_Tm",
                "Tm Rank": "Aw_Rank",
            }
        )
    )
    hm_tm_rating = (
        tm_ratings[tm_ratings["Tm"] == hm_tm]
        .reset_index(drop=True)
        .rename(
            columns={
                "Tm": "Hm_Tm",
                "Tm Rank": "Hm_Rank",
            }
        )
    )

    # join to matchup_data
    matchup_data = matchup_data.merge(
        aw_tm_rating[["Aw_Tm", "Aw_Rank"]], how="inner", on=["Aw_Tm"]
    ).merge(hm_tm_rating[["Hm_Tm", "Hm_Rank"]], how="inner", on=["Hm_Tm"])
    matchup_data = matchup_data.rename(
        columns={"Hm_Rank": "Home Rk", "Aw_Rank": "Away Rk", "Hm_Season": "Season"}
    ).drop(columns=["Aw_Season"])

    # if NCAA Tourney game get seed
    if game_details["NCAA Tourney Game"]:
        seeding_df = pd.read_csv(
            f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season_bracketology.csv",
        )
        season_seeds = seeding_df[seeding_df["season"] == data_seasons[-1]]

        # away seed
        aw_seed = (
            season_seeds[season_seeds["tm"] == aw_tm]
            .reset_index(drop=True)
            .rename(
                columns={
                    "tm": "Aw_Tm",
                    "seed": "Aw_Seed",
                }
            )
        )
        # home seed
        hm_seed = (
            season_seeds[season_seeds["tm"] == hm_tm]
            .reset_index(drop=True)
            .rename(
                columns={
                    "tm": "Hm_Tm",
                    "seed": "Hm_Seed",
                }
            )
        )
        # join to matchup_data
        matchup_data = matchup_data.merge(
            aw_seed[["Aw_Tm", "Aw_Seed"]], how="inner", on=["Aw_Tm"]
        ).merge(hm_seed[["Hm_Tm", "Hm_Seed"]], how="inner", on=["Hm_Tm"])
    else:
        matchup_data["Aw_Seed"] = pd.NA
        matchup_data["Hm_Seed"] = pd.NA

    # additional matchup details
    matchup_data["Neutral Game"] = [game_details["Neutral Game"]]
    matchup_data["Conference Game"] = [game_details["Conference Game"]]
    matchup_data["Postseason Game"] = [game_details["Postseason Game"]]
    matchup_data["NCAA Tourney Game"] = [game_details["NCAA Tourney Game"]]

    # fix empty data in matchup
    matchup_data = matchup_data.fillna(np.nan)
    
    """
        Build Out Model
    """

    model_df = matchup_df.sort_values(by=["Game Date", "Matchup"]).reset_index(
        drop=True
    )[
        [
            "Aw_Tm",
            "Aw_W",
            "Aw_Wpct",
            "Aw_Poss",
            "Aw_OppPoss",
            "Aw_OffEff",
            "Aw_ShootEff",
            "Aw_DefEff",
            "Aw_OppShootEff",
            "Aw_TmLuckW",
            "Aw_Pts",
            "Aw_OppPts",
            "Aw_TmLuck",
            "Game Date",
            "Hm_Tm",
            "Hm_W",
            "Hm_Wpct",
            "Hm_Poss",
            "Hm_OppPoss",
            "Hm_OffEff",
            "Hm_ShootEff",
            "Hm_DefEff",
            "Hm_OppShootEff",
            "Hm_TmLuckW",
            "Hm_Pts",
            "Hm_OppPts",
            "Hm_TmLuck",
            "Season",
            "Matchup",
            "Home Team",
            "Away Team",
            "Home W",
            "Home Pt Diff",
            "Away Rk",
            "Home Rk",
            "Aw_Seed",
            "Hm_Seed",
            "Neutral Game",
            "Conference Game",
            "Postseason Game",
            "NCAA Tourney Game",
        ]
    ]

    """
        Set Target Variables/DFs
    """
    final_pred_df = pd.DataFrame()
    final_model_coef = pd.DataFrame()
    final_model_stats = pd.DataFrame()
    for target_variable in ["Home W", "Home Pt Diff"]:
        logger.info(f"data transformed: setting target variable - {target_variable}")
        # target variable
        y = model_df[
            [
                f"Matchup",
                f"{target_variable}",
            ]
        ]
        model_y = y.drop(columns=["Matchup"])
        X = model_df.drop(
            columns=[
                "Home W",
                "Home Team",
                "Hm_Tm",
                "Away Team",
                "Aw_Tm",
                "Game Date",
                "Home Pt Diff",
            ]
        )
        model_X = X.drop(columns=["Matchup"])

        """
            Run Model
        """
        logger.info(f"commence model run... NOW")
        # in order to predict probability of attendance use "model.predict_proba()"

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            model_X.iloc[1:], model_y.iloc[1:], test_size=0.2, random_state=14
        )

        # call model with parameters
        if target_variable in ["Home W"]:
            model = HistGradientBoostingClassifier(
                max_iter=1000,
                early_stopping=True,
                warm_start=True,
                random_state=14,
            )
        elif target_variable in ["Home Pt Diff"]:
            model = HistGradientBoostingRegressor(
                max_iter=1000,
                early_stopping=True,
                warm_start=True,
                random_state=14,
            )

        # logger.info(f"training model")
        model.fit(X_train, np.ravel(y_train))

        # suppress scientific notation
        # logger.info(f"predicting test set")
        np.set_printoptions(suppress=True)
        predictions = model.predict(X_test)
        if target_variable in ["Home W"]:
            predict_prob = model.predict_proba(X_test)

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(predict_prob)):
            tmp_pred = predictions[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W"]:
                tmp_prob = predict_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df (with error)
        tmp_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Away Team",
                "Home W",
                "Predict",
                "Predict Probability",
            ]
        )

        # dynamically get test data info for final df
        indexes = y_test.index.tolist()
        mylist = []
        for x in indexes:
            mylist += [x]

        tmp_df["Matchup"] = model_df.iloc[mylist]["Matchup"]
        tmp_df["Game Date"] = model_df.iloc[mylist]["Game Date"]
        tmp_df["Home Team"] = model_df.iloc[mylist]["Home Team"]
        tmp_df["Away Team"] = model_df.iloc[mylist]["Away Team"]
        tmp_df["Home W"] = model_df.iloc[mylist]["Home W"]

        tmp_df["Predict"] = w_pred
        if target_variable in ["Home W"]:
            tmp_df["Predict Probability"] = w_prob

        # model statistics
        if target_variable in ["Home W"]:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                    "Recall Score",
                    "Precision Score",
                    "ROC AUC",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test, y_test)]
            model_stats_df["Recall Score"] = [
                recall_score(y_test, predictions)
            ]  # tp / (tp + fn)
            model_stats_df["Precision Score"] = [
                precision_score(y_test, predictions)
            ]  # tp / (tp + fp)
            model_stats_df["ROC AUC"] = roc_auc_score(y_test, w_prob)
        else:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                ]
            )
            model_stats_df["Target Variable"] = [target_variable]
            model_stats_df["R^2"] = [model.score(X_test, y_test)]

        """
            Confusion Matrix for Train/Test
        """
        # matrix = confusion_matrix(y_test, predictions)
        # matrix_viz = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        # matrix_viz.plot()
        # plt.show()

        """
            Matchup Prediction (probability of home team winning)
        """
        logger.info(f"predicting {target_variable}: {matchup_data["Matchup"][0]}")
        prediction_df = matchup_data[
            [
                "Aw_W",
                "Aw_Wpct",
                "Aw_Poss",
                "Aw_OppPoss",
                "Aw_OffEff",
                "Aw_ShootEff",
                "Aw_DefEff",
                "Aw_OppShootEff",
                "Aw_TmLuckW",
                "Aw_Pts",
                "Aw_OppPts",
                "Aw_TmLuck",
                "Hm_W",
                "Hm_Wpct",
                "Hm_Poss",
                "Hm_OppPoss",
                "Hm_OffEff",
                "Hm_ShootEff",
                "Hm_DefEff",
                "Hm_OppShootEff",
                "Hm_TmLuckW",
                "Hm_Pts",
                "Hm_OppPts",
                "Hm_TmLuck",
                "Season",
                "Away Rk",
                "Home Rk",
                "Aw_Seed",
                "Hm_Seed",
                "Neutral Game",
                "Conference Game",
                "Postseason Game",
                "NCAA Tourney Game",
            ]
        ]
        # prediction_df = prediction_df[limit_X_train.feature.tolist()]

        if target_variable in ["Home W"]:
            prediction_prob = model.predict_proba(prediction_df)
        prediction = model.predict(prediction_df)

        # roll through prediction probabilities and give me the probability of attending the game
        #   i.e. the second number of the array for each index
        w_prob = []
        w_pred = []
        for i in range(len(prediction_prob)):
            tmp_pred = prediction[i]
            w_pred += [tmp_pred]

            if target_variable in ["Home W"]:
                tmp_prob = prediction_prob[i]
                w_prob += [tmp_prob[1]]

        # view results in df
        pred_df = pd.DataFrame(
            columns=[
                "Matchup",
                "Game Date",
                "Home Team",
                "Away Team",
                f"Predict",
                f"Predict Probability",
            ]
        )

        # dynamically get test data info for final df
        indexes = prediction_df.index.tolist()
        mylist = []
        for x in indexes:
            mylist += [x]

        pred_df["Matchup"] = matchup_data.iloc[mylist]["Matchup"]
        pred_df["Game Date"] = matchup_data.iloc[mylist]["Game Date"]
        pred_df["Home Team"] = matchup_data.iloc[mylist]["Hm_Tm"]
        pred_df["Away Team"] = matchup_data.iloc[mylist]["Aw_Tm"]

        pred_df[f"Predict"] = w_pred

        if target_variable in ["Home W"]:
            pred_df[f"Predict Probability"] = w_prob

            # round the probability variable
            pred_df["Predict Probability"] = np.round(pred_df["Predict Probability"], 4)

        if final_pred_df.empty:
            final_pred_df = pred_df.rename(
                columns={
                    "Predict": f"{target_variable}",
                    "Predict Probability": f"{target_variable} Probability",
                }
            )
        else:
            final_pred_df = final_pred_df.merge(
                pred_df[["Matchup", "Predict"]], how="inner", on="Matchup"
            ).rename(columns={"Predict": f"{target_variable}"})

        # concat model_stats_df
        final_model_stats = pd.concat([final_model_stats, model_stats_df]).reset_index(
            drop=True
        )

    # format a df to fit the donut chart
    sg_win = pd.DataFrame(
        data={
            "Tm": [final_pred_df["Away Team"][0], final_pred_df["Home Team"][0]],
            "Win Prob.": [
                1 - final_pred_df["Home W Probability"][0],
                final_pred_df["Home W Probability"][0],
            ],
            "Point Diff": [
                final_pred_df["Home Pt Diff"][0],
                final_pred_df["Home Pt Diff"][0] * -1,
            ],
        }
    )

    return [final_pred_df, final_model_stats, final_model_coef, sg_win]
