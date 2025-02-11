import pandas as pd
import numpy as np

from play_matchup import *
from team_dict import *


def bracketology(season_year):
    bracket_df = pd.read_csv(
        f"~/Documents/Python/professional_portfolio/march_madness/csv_files/season_bracketology.csv"
    )

    bracket_df = bracket_df[bracket_df["season"] == season_year]

    return bracket_df

## Need to add this feature
# def play_in_games():
#     winners = []
#     return winners


def round_64(season_year, today, tm_df, n):

    # Matchups (within regions)
    #   1 vs. 16
    #   2 vs. 15
    #   3 vs. 14
    #   4 vs. 13
    #   5 vs. 12
    #   6 vs. 11
    #   7 vs. 10
    #   8 vs. 9

    rnk_dict = {
        1: 1,
        8: 2,
        5: 3,
        4: 4,
        6: 5,
        3: 6,
        7: 7,
        2: 8,
        15: 9,
        10: 10,
        14: 11,
        11: 12,
        13: 13,
        12: 14,
        9: 15,
        16: 16,
    }
    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    bracket_df = bracketology(season_year)
    # reorder seeds for matchups
    bracket_df["seed_rnk"] = bracket_df["seed"].map(rnk_dict)
    bracket_df["region_rnk"] = bracket_df["region"].map(reg_rnk[season_year])
    bracket_df = bracket_df.sort_values(by=["region_rnk", "seed_rnk"]).reset_index(
        drop=True
    )

    # create list for tournament regions
    region_list = bracket_df["region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    for region in region_list:
        region_df = bracket_df[bracket_df["region"] == region]
        seed_list = region_df["seed"].unique().tolist()

        while not seed_list == []:
            tm1 = region_df[region_df["seed"] == seed_list.pop(0)]["tm"].reset_index(
                drop=True
            )[0]
            tm2 = region_df[region_df["seed"] == seed_list.pop()]["tm"].reset_index(
                drop=True
            )[0]

            # print(f"{tm2} vs. {tm1}")
            # logger.info(f"{region}: {tm2} vs. {tm1}")
            game = matchup(
                tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n
            )

            winner_df = pd.concat(
                [
                    winner_df,
                    pd.DataFrame(
                        data={
                            "Region": [region],
                            "Gm Winner": [game[0]],
                            "Pt Spread": [game[1]],
                        }
                    ),
                ]
            ).reset_index(drop=True)
            winner_list += [game[0]]

    return winner_df


def round_32(season_year, today, tm_df, winner64_df, n):

    # Matchups (within regions)
    #   1/16 vs. 8/9
    #   2/15 vs. 7/10
    #   3/14 vs. 5/12
    #   4/13 vs. 6/11

    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    # create list for tournament regions
    region_list = winner64_df["Region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    for region in region_list:
        region_df = winner64_df[winner64_df["Region"] == region].reset_index(drop=True).reset_index()
        region_df = region_df.rename(columns={"index": "seed"})
        
        seed_list = []
        for i in range(len(region_df)):
            seed_list += [i]

        # region_df.loc[region_df["Region"] == region, "seed"] = seed_list

        while not seed_list == []:
            tm1 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]
            tm2 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]

            # print(f"{tm2} vs. {tm1}")
            # logger.info(f"{region}: {tm2} vs. {tm1}")
            game = matchup(
                tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n
            )

            winner_df = pd.concat(
                [
                    winner_df,
                    pd.DataFrame(
                        data={
                            "Region": [region],
                            "Gm Winner": [game[0]],
                            "Pt Spread": [game[1]],
                        }
                    ),
                ]
            ).reset_index(drop=True)
            winner_list += [game[0]]

    return winner_df


def sweet_16(season_year, today, tm_df, winner32_df, n):

    # Sweet Sixteen (within regions)
    #   1/16/8/9 vs. 2/15/7/10
    #   3/14/5/12 vs. 4/13/6/11

    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    # create list for tournament regions
    region_list = winner32_df["Region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    for region in region_list:
        region_df = (
            winner32_df[winner32_df["Region"] == region]
            .reset_index(drop=True)
            .reset_index()
        )
        region_df = region_df.rename(columns={"index": "seed"})

        seed_list = []
        for i in range(len(region_df)):
            seed_list += [i]

        # region_df.loc[region_df["Region"] == region, "seed"] = seed_list

        while not seed_list == []:
            tm1 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]
            tm2 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]

            # print(f"{tm2} vs. {tm1}")
            # logger.info(f"{region}: {tm2} vs. {tm1}")
            game = matchup(
                tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n
            )

            winner_df = pd.concat(
                [
                    winner_df,
                    pd.DataFrame(
                        data={
                            "Region": [region],
                            "Gm Winner": [game[0]],
                            "Pt Spread": [game[1]],
                        }
                    ),
                ]
            ).reset_index(drop=True)
            winner_list += [game[0]]

    return winner_df


def elite_8(season_year, today, tm_df, winner16_df, n):

    # Elite Eight (within regions)
    # Region Finals

    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    # create list for tournament regions
    region_list = winner16_df["Region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    for region in region_list:
        region_df = (
            winner16_df[winner16_df["Region"] == region]
            .reset_index(drop=True)
            .reset_index()
        )
        region_df = region_df.rename(columns={"index": "seed"})

        seed_list = []
        for i in range(len(region_df)):
            seed_list += [i]

        region_df.loc[region_df["Region"] == region, "seed"] = seed_list

        while not seed_list == []:
            tm1 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]
            tm2 = region_df[region_df["seed"] == seed_list.pop(0)][
                "Gm Winner"
            ].reset_index(drop=True)[0]

            # print(f"{tm2} vs. {tm1}")
            # logger.info(f"{region}: {tm2} vs. {tm1}")
            game = matchup(
                tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n
            )

            winner_df = pd.concat(
                [
                    winner_df,
                    pd.DataFrame(
                        data={
                            "Region": [region],
                            "Gm Winner": [game[0]],
                            "Pt Spread": [game[1]],
                        }
                    ),
                ]
            ).reset_index(drop=True)
            winner_list += [game[0]]

    return winner_df


def final_4(season_year, today, tm_df, winner8_df, n):

    # Final Four

    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    # create list for tournament regions
    region_list = winner8_df["Region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    # for region in region_list:
    # region_df = winner8_df[winner8_df["Region"] == region]
    
    seed_list = []
    for i in range(len(winner8_df)):
        seed_list += [i]

    winner8_df = winner8_df.reset_index(drop=True).reset_index().rename(columns={'index': 'seed'})
    # winner8_df["seed"] = seed_list

    while not seed_list == []:
        tm1 = winner8_df[winner8_df["seed"] == seed_list.pop(0)][
            "Gm Winner"
        ].reset_index(drop=True)[0]
        tm2 = winner8_df[winner8_df["seed"] == seed_list.pop(0)][
            "Gm Winner"
        ].reset_index(drop=True)[0]

        # print(f"{tm2} vs. {tm1}")
        # logger.info(f"Final Four: {tm2} vs. {tm1}")
        game = matchup(tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n)

        winner_df = pd.concat(
            [
                winner_df,
                pd.DataFrame(
                    data={
                        "Gm Winner": [game[0]],
                        "Pt Spread": [game[1]],
                    }
                ),
            ]
        ).reset_index(drop=True)
        winner_list += [game[0]]

    return winner_df


def final_2(season_year, today, tm_df, winner4_df, n):

    # Final Four

    reg_rnk = {
        2024: {
            "East": 1,
            "West": 2,
            "South": 3,
            "Midwest": 4,
        }
    }

    # create list for tournament regions
    # region_list = winner8_df["Region"].unique().tolist()

    # set up region matchups for Round of 64
    winner_list = []
    winner_df = pd.DataFrame()
    # for region in region_list:
    # region_df = winner8_df[winner8_df["Region"] == region]

    seed_list = []
    for i in range(len(winner4_df)):
        seed_list += [i]

    winner4_df = (
            winner4_df.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "seed"})
        )
    # winner4_df["seed"] = seed_list

    while not seed_list == []:
        tm1 = winner4_df[winner4_df["seed"] == seed_list.pop(0)][
            "Gm Winner"
        ].reset_index(drop=True)[0]
        tm2 = winner4_df[winner4_df["seed"] == seed_list.pop(0)][
            "Gm Winner"
        ].reset_index(drop=True)[0]

        # print(f"{tm2} vs. {tm1}")
        # logger.info(f"Championship: {tm2} vs. {tm1}")
        game = matchup(tm2, tm1, season_year, today, tm_df, neutral=True, conf_game=False, n=n)

        winner_df = pd.concat(
            [
                winner_df,
                pd.DataFrame(
                    data={
                        "Gm Winner": [game[0]],
                        "Pt Spread": [game[1]],
                    }
                ),
            ]
        ).reset_index(drop=True)
        winner_list += [game[0]]

    return winner_df
