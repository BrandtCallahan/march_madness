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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from utils.get_data import *
from utils.team_dict import *
from utils.beautiful_soup_helper import *
from ml_single_game.game_data import *


def single_game_model(data_seasons, today, matchup):

    # load up matchup results for training
    matchup_df = pd.DataFrame()
    for season in data_seasons:
        tmp_df = pd.read_csv(
            f"~/OneDrive - Tennessee Titans/Documents/Python/professional_portfolio/march_madness/csv_files/season{season}_matchup_results.csv",
        )
        tmp_df = tmp_df.astype({"Game Date": "datetime64[ns]"})

        # concat all years of data into one df
        matchup_df = pd.concat([matchup_df, tmp_df]).reset_index(drop=True)

    # make sure only data from before "today"
    matchup_df = matchup_df[
        matchup_df["Game Date"].astype("datetime64[ns]") < pd.to_datetime(today)
    ].reset_index(drop=True)

    # data for the matchup
    hm_tm = matchup.split(" vs. ")[1]
    aw_tm = matchup.split(" vs. ")[0]
    matchup_data = rolling_gamedata(
        data_seasons[-1],
        hm_tm,
        aw_tm,
        pd.to_datetime(today),
    )

    """
        Build Out Model
    """

    model_df = matchup_df.sort_values(by=["Game Date", "Matchup"]).reset_index(
        drop=True
    )

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
            model = LogisticRegression(
                max_iter=1000,
                solver="newton-cholesky",
                random_state=14,
            )
        elif target_variable in ["Home Pt Diff"]:
            model = LinearRegression()

        logger.info(f"training model")
        model.fit(X_train, np.ravel(y_train))

        if target_variable in ["Home W"]:
            model_coef = model.coef_[0]
            abs_coef = [abs(model.intercept_[0])]
            reg_coef = [model.intercept_[0]]
        else:
            model_coef = model.coef_
            abs_coef = [abs(model.intercept_)]
            reg_coef = [model.intercept_]
        features = ["intercept"]

        for coef in model_coef:
            abs_coef += [abs(coef)]
            reg_coef += [coef]

        for name in model.feature_names_in_:
            features += [name]
        model_coef = pd.DataFrame(
            data={
                "target variable": target_variable,
                "feature": features,
                "coef": reg_coef,
                "abs_coef": abs_coef,
            }
        )

        # suppress scientific notation
        logger.info(f"predicting test set")
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
                f"Predict",
                f"Predict Probability",
                # "error",
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

        tmp_df[f"Predict"] = w_pred
        if target_variable in ["Home W"]:
            tmp_df[f"Predict Probability"] = w_prob

        # tmp_df["Error"] = tmp_df[f"{target_stat}"] - tmp_df[f"Pred. {target_stat}"]

        # model statistics
        if target_variable in ["Home W"]:
            model_stats_df = pd.DataFrame(
                columns=[
                    "Target Variable",
                    "R^2",
                    "Recall Score",
                    "Precision Score",
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
            Confusion Matrix for Training
        """
        # matrix = confusion_matrix(y_test, predictions)
        # matrix_viz = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
        # matrix_viz.plot()
        # plt.show()

        """
            Matchup Prediction (probability of home team winning)
        """
        logger.info(f"predicting: {matchup_data["Matchup"][0]}")
        prediction_df = matchup_data.drop(
            columns=["Aw_Tm", "Hm_Tm", "Matchup", "Game Date"]
        )
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

        # concat model_stats_df & model_coef
        final_model_stats = pd.concat([final_model_stats, model_stats_df]).reset_index(
            drop=True
        )
        final_model_coef = pd.concat([final_model_coef, model_coef]).reset_index(
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
