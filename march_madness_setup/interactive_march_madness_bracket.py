import pandas as pd
import numpy as np

from march_madness_setup.bracketology import *
from utils.get_data import *


def interactive_march_madness(season, today, modeltype):

    n = int(input(f"How many brackets do you want to fill out? "))

    # tm_df = tm_rating(season, today)
    mm_bracket = bracketology(season)[["season", "region", "seed", "tm"]]

    results = []
    r64_arch = pd.DataFrame()
    r32_arch = pd.DataFrame()
    r16_arch = pd.DataFrame()
    r8_arch = pd.DataFrame()
    r4_arch = pd.DataFrame()
    r2_arch = pd.DataFrame()

    logger.info(f"Welcome to March Madness")

    # Round of 64
    r64 = interactive_round_64(season, today, modeltype)
    r64_arch = pd.concat([r64_arch, r64]).reset_index(drop=True)
    r64_hist = (
        r64_arch.groupby(["Region", "Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r64_hist["NumG"] = n
    r64_hist["Adv%"] = r64_hist["NumW"].fillna(0) / r64_hist["NumG"]

    # Round of 32
    r32 = interactive_round_32(season, today, modeltype, r64)
    r32_arch = pd.concat([r32_arch, r32]).reset_index(drop=True)
    r32_hist = (
        r32_arch.groupby(["Region", "Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r32_hist["NumG"] = n
    r32_hist["Adv%"] = r32_hist["NumW"].fillna(0) / r32_hist["NumG"]

    # Sweet Sixteen
    r16 = interactive_sweet_16(season, today, modeltype, r32)
    r16_arch = pd.concat([r16_arch, r16]).reset_index(drop=True)
    r16_hist = (
        r16_arch.groupby(["Region", "Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r16_hist["NumG"] = n
    r16_hist["Adv%"] = r16_hist["NumW"].fillna(0) / r16_hist["NumG"]

    # Elite Eight
    r8 = interactive_elite_8(season, today, modeltype, r16)
    r8_arch = pd.concat([r8_arch, r8]).reset_index(drop=True)
    r8_hist = (
        r8_arch.groupby(["Region", "Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r8_hist["NumG"] = n
    r8_hist["Adv%"] = r8_hist["NumW"].fillna(0) / r8_hist["NumG"]

    # Final Four
    r4 = interactive_final_4(season, today, modeltype, r8)
    r4_arch = pd.concat([r4_arch, r4]).reset_index(drop=True)
    r4_hist = (
        r4_arch.groupby(["Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r4_hist["NumG"] = n
    r4_hist["Adv%"] = r4_hist["NumW"].fillna(0) / r4_hist["NumG"]

    # Championship
    r2 = interactive_final_2(season, today, modeltype, r4)
    r2_arch = pd.concat([r2_arch, r2]).reset_index(drop=True)
    r2_hist = (
        r2_arch.groupby(["Gm Winner"], observed=True)
        .agg(NumW=("Gm Winner", "count"))
        .reset_index()
    )
    r2_hist["NumG"] = n
    r2_hist["Adv%"] = r2_hist["NumW"].fillna(0) / r2_hist["NumG"]

    results += [r2["Gm Winner"][0]]

    # value_counts = pd.Series([item for item in results]).value_counts()
    # print(value_counts)

    bracket_results = (
        mm_bracket.merge(
            r64_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Round 32"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )
    bracket_results = (
        bracket_results.merge(
            r32_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Sweet Sixteen"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )
    bracket_results = (
        bracket_results.merge(
            r16_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Elite Eight"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )
    bracket_results = (
        bracket_results.merge(
            r8_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Final Four"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )
    bracket_results = (
        bracket_results.merge(
            r4_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Championship"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )
    bracket_results = (
        bracket_results.merge(
            r2_hist[["Gm Winner", "Adv%"]].rename(columns={"Adv%": "Champion"}),
            how="left",
            left_on="tm",
            right_on="Gm Winner",
        )
        .drop(columns=["Gm Winner"])
        .fillna(0)
    )

    logger.info(f"Your bracket is complete.")

    return bracket_results
